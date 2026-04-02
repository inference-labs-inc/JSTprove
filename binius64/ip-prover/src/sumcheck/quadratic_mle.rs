// Copyright 2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{AsSlicesMut, FieldSliceMut, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;

use super::{common::SumcheckProver, error::Error, gruen32::Gruen32, round_evals::RoundEvals2};
use crate::sumcheck::common::MleCheckProver;

/// MLE-check prover for polynomials defined as quadratic compositions of N multilinear polynomials.
///
/// This prover implements the MLE (multilinear extension) check protocol. Given N multilinear
/// polynomials M₁, M₂, ..., Mₙ and a quadratic composition function C, it proves claims about
/// the multilinear extension of the composite polynomial C(M₁, M₂, ..., Mₙ).
///
/// The prover uses the sumcheck protocol to reduce claims about this multilinear extension
/// to claims about the individual multilinear evaluations, employing the Karatsuba optimization
/// for efficient degree-2 polynomial interpolation.
pub struct QuadraticMleCheckProver<P: PackedField, Composition, InfinityComposition, const N: usize>
{
    multilinears: Box<dyn AsSlicesMut<P, N> + Send>,
    composition: Composition,
    infinity_composition: InfinityComposition,
    last_coeffs_or_eval: RoundCoeffsOrEval<P::Scalar>,
    gruen32: Gruen32<P>,
}

impl<F, P, Composition, InfinityComposition, const N: usize>
    QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
    F: Field,
    P: PackedField<Scalar = F>,
    Composition: Fn([P; N]) -> P + Sync,
    InfinityComposition: Fn([P; N]) -> P + Sync,
{
    /// Creates a new prover for verifying quadratic composite polynomial evaluations.
    ///
    /// # Arguments
    ///
    /// * `multilinears` - Array of N multilinear polynomials that serve as inputs to the
    ///   composition. Each multilinear must have the same number of variables.
    ///
    /// * `composition` - Function for evaluating the N-variate quadratic composition over packed
    ///   field elements. Takes an array of N packed field elements (one from each multilinear) and
    ///   returns their composition value. For example, for a product of two multilinears, this
    ///   would compute `M₁(X) * M₂(X)`.
    ///
    /// * `infinity_composition` - Polynomial evaluator that computes only the highest-degree terms
    ///   of the composition. This is used for evaluation at the "infinity point" in the Karatsuba
    ///   optimization, where we take the limit of P(X)/X^d as X approaches infinity. This limit
    ///   equals the coefficient of the highest-degree term, effectively ignoring lower-degree
    ///   terms. For example, if composition is `a*b - c`, the infinity_composition would be just
    ///   `a*b`.
    ///
    /// * `eval_point` - The point at which the multilinear extension is being evaluated. Must have
    ///   length equal to the number of variables in the multilinears.
    ///
    /// * `eval_claim` - The claimed value of the multilinear extension of the composite polynomial
    ///   at the evaluation point. This is the multilinear extension of the function that maps v ∈
    ///   {0,1}ⁿ to C(M₁(v), M₂(v), ..., Mₙ(v)), evaluated at eval_point.
    ///
    /// # Returns
    ///
    /// A configured prover instance ready to execute the sumcheck protocol.
    ///
    /// # Errors
    ///
    /// Returns `Error::MultilinearSizeMismatch` if any multilinear has a different number of
    /// variables than the length of `eval_point`.
    pub fn new(
        mut multilinears: impl AsSlicesMut<P, N> + Send + 'static,
        composition: Composition,
        infinity_composition: InfinityComposition,
        eval_point: Vec<F>,
        eval_claim: F,
    ) -> Result<Self, Error> {
        let n_vars = eval_point.len();

        for multilinear in &multilinears.as_slices_mut() {
            if multilinear.log_len() != n_vars {
                return Err(Error::MultilinearSizeMismatch);
            }
        }

        let last_coeffs_or_eval = RoundCoeffsOrEval::Eval(eval_claim);
        let gruen32 = Gruen32::new(&eval_point);

        Ok(Self {
            multilinears: Box::new(multilinears),
            composition,
            infinity_composition,
            last_coeffs_or_eval,
            gruen32,
        })
    }

    /// Gets mutable slices of the multilinears, truncated to the current number of variables.
    fn multilinears_mut(&mut self) -> [FieldSliceMut<'_, P>; N] {
        let n_vars = self.gruen32.n_vars_remaining();
        let mut slices = self.multilinears.as_slices_mut();
        for slice in &mut slices {
            slice.truncate(n_vars);
        }
        slices
    }
}

impl<F, P, Composition, InfinityComposition, const N: usize> SumcheckProver<F>
    for QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
    F: Field,
    P: PackedField<Scalar = F>,
    Composition: Fn([P; N]) -> P + Sync,
    InfinityComposition: Fn([P; N]) -> P + Sync,
{
    fn n_vars(&self) -> usize {
        self.gruen32.n_vars_remaining()
    }

    fn n_claims(&self) -> usize {
        1
    }

    fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
        let last_eval = match &self.last_coeffs_or_eval {
            RoundCoeffsOrEval::Eval(eval) => *eval,
            RoundCoeffsOrEval::Coeffs(_) => return Err(Error::ExpectedFold),
        };

        let n_vars_remaining = self.gruen32.n_vars_remaining();
        assert!(n_vars_remaining > 0);

        let eq_expansion = self.gruen32.eq_expansion();
        assert_eq!(eq_expansion.log_len(), n_vars_remaining - 1);

        // Get references to compositions - these don't conflict with multilinears borrow
        let composition = &self.composition;
        let infinity_composition = &self.infinity_composition;

        // Get multilinear slices and truncate to current n_vars
        let mut multilinears = self.multilinears.as_slices_mut();
        for slice in &mut multilinears {
            slice.truncate(n_vars_remaining);
        }

        // Split each multilinear in half
        let (splits_0, splits_1) = multilinears
            .iter()
            .map(|multilinear| multilinear.split_half_ref())
            .unzip::<_, _, Vec<_>, Vec<_>>();

        // Compute F(1) and F(∞) where F = ∑_{v ∈ B} C(M_1(v || X), ..., M_N(v || X)) eq(v, z).
        // We need to iterate over all positions in parallel
        let round_evals = eq_expansion
            .as_ref()
            .into_par_iter()
            .enumerate()
            .map(|(i, &eq_i)| {
                // Collect evaluations at 1 and ∞ for each multilinear
                let mut evals_1 = [P::default(); N];
                let mut evals_inf = [P::default(); N];
                for j in 0..N {
                    evals_1[j] = splits_1[j].as_ref()[i];
                    evals_inf[j] = splits_0[j].as_ref()[i] + splits_1[j].as_ref()[i];
                }

                // Evaluate composition at X=1
                let y_1 = composition(evals_1) * eq_i;

                // Evaluate composition at X=∞ (where M(∞) = M(0) + M(1))
                let y_inf = infinity_composition(evals_inf) * eq_i;

                RoundEvals2 { y_1, y_inf }
            })
            .reduce(RoundEvals2::default, |lhs, rhs| lhs + &rhs)
            .sum_scalars(n_vars_remaining - 1);

        let alpha = self.gruen32.next_coordinate();
        let round_coeffs = round_evals.interpolate_eq(last_eval, alpha);

        self.last_coeffs_or_eval = RoundCoeffsOrEval::Coeffs(round_coeffs.clone());
        Ok(vec![round_coeffs])
    }

    fn fold(&mut self, challenge: F) -> Result<(), Error> {
        let RoundCoeffsOrEval::Coeffs(coeffs) = &self.last_coeffs_or_eval else {
            return Err(Error::ExpectedExecute);
        };

        assert!(
            self.n_vars() > 0,
            "n_vars is decremented in fold; \
			fold changes last_coeffs_or_eval to Eval variant; \
			fold only executes with Coeffs variant; \
			thus, n_vars should be > 0"
        );

        let eval = coeffs.evaluate(challenge);

        for multilinear in &mut self.multilinears_mut() {
            fold_highest_var_inplace(multilinear, challenge);
        }

        self.gruen32.fold(challenge);
        self.last_coeffs_or_eval = RoundCoeffsOrEval::Eval(eval);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<F>, Error> {
        if self.n_vars() > 0 {
            let error = match self.last_coeffs_or_eval {
                RoundCoeffsOrEval::Coeffs(_) => Error::ExpectedFold,
                RoundCoeffsOrEval::Eval(_) => Error::ExpectedExecute,
            };

            return Err(error);
        }

        let multilinear_evals = self
            .multilinears_mut()
            .into_iter()
            .map(|multilinear| multilinear.get(0))
            .collect();

        Ok(multilinear_evals)
    }
}

impl<F, P, Composition, InfinityComposition, const N: usize> MleCheckProver<F>
    for QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
    F: Field,
    P: PackedField<Scalar = F>,
    Composition: Fn([P; N]) -> P + Sync,
    InfinityComposition: Fn([P; N]) -> P + Sync,
{
    fn eval_point(&self) -> &[F] {
        &self.gruen32.eval_point()[..self.n_vars()]
    }
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEval<F: Field> {
    Coeffs(RoundCoeffs<F>),
    Eval(F),
}

#[cfg(test)]
mod tests {
    use std::{array, iter};

    use binius_field::{arch::OptimalPackedB128, field::FieldOps};
    use binius_ip::mlecheck;
    use binius_math::{
        FieldBuffer,
        multilinear::evaluate::evaluate,
        test_utils::{random_field_buffer, random_scalars},
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

    type StdChallenger = HasherChallenger<sha2::Sha256>;
    use itertools::{self, Itertools};
    use rand::{SeedableRng, prelude::StdRng};

    use super::*;
    use crate::sumcheck::prove_single_mlecheck;

    fn test_mlecheck_prove_verify<F, P, Composition, InfinityComposition, const N: usize>(
        prover: QuadraticMleCheckProver<P, Composition, InfinityComposition, N>,
        composition: Composition,
        eval_claim: F,
        eval_point: &[F],
        multilinears: Vec<FieldBuffer<P>>,
    ) where
        F: Field,
        P: PackedField<Scalar = F>,
        Composition: Fn([P; N]) -> P + Sync,
        InfinityComposition: Fn([P; N]) -> P + Sync,
    {
        // Run the proving protocol
        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let output = prove_single_mlecheck(prover, &mut prover_transcript).unwrap();

        // Write the multilinear evaluations to the transcript
        prover_transcript
            .message()
            .write_slice(&output.multilinear_evals);

        // Convert to verifier transcript and run verification
        let mut verifier_transcript = prover_transcript.into_verifier();
        let sumcheck_output = mlecheck::verify(
            eval_point,
            2, // degree 2 for composite polynomials
            eval_claim,
            &mut verifier_transcript,
        )
        .unwrap();

        let mut reduced_eval_point = sumcheck_output.challenges.clone();
        reduced_eval_point.reverse();

        // Read the multilinear evaluations from the transcript
        let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(N).unwrap();

        // Check that the composition of the evaluations equals the reduced evaluation
        let evals_packed: [P; N] = array::from_fn(|i| P::broadcast(multilinear_evals[i]));
        let composition_result = composition(evals_packed);
        assert_eq!(
            composition_result.iter().next().unwrap(),
            sumcheck_output.eval,
            "Composition of multilinear evaluations should equal the reduced evaluation"
        );

        // Check that the original multilinears evaluate to the claimed values at the challenge
        // point
        for (multilinear, claimed_eval) in iter::zip(&multilinears, multilinear_evals) {
            let actual_eval = evaluate(multilinear, &reduced_eval_point);
            assert_eq!(actual_eval, claimed_eval);
        }

        // Also verify the challenges match what the prover saw
        assert_eq!(
            output.challenges, sumcheck_output.challenges,
            "Prover and verifier challenges should match"
        );
    }

    fn test_quadratic_mlecheck_prove_verify<F, P, const N: usize>(
        composition: impl Fn([P; N]) -> P + Clone + Sync,
        infinity_composition: impl Fn([P; N]) -> P + Clone + Sync,
    ) where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        let n_vars = 8;
        let mut rng = StdRng::seed_from_u64(0);

        // Generate random multilinear polynomials
        let multilinears: [_; N] = array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));

        // Compute product multilinear
        let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
            .map(|i| {
                let vals = array::from_fn(|j| multilinears[j].as_ref()[i]);
                composition(vals)
            })
            .collect_vec();
        let composite_vals = FieldBuffer::new(n_vars, composite_vals);

        let eval_point = random_scalars::<F>(&mut rng, n_vars);
        let eval_claim = evaluate(&composite_vals, &eval_point);

        // Create the prover
        let mlecheck_prover = QuadraticMleCheckProver::new(
            multilinears.clone(),
            composition.clone(),
            infinity_composition,
            eval_point.clone(),
            eval_claim,
        )
        .unwrap();

        test_mlecheck_prove_verify(
            mlecheck_prover,
            composition,
            eval_claim,
            &eval_point,
            multilinears.to_vec(),
        );
    }

    // Test that quadratic MLE-check handles multilinears. It's not the most efficient strategy
    // for a multilinear MLE-check, but it's a good edge case.
    #[test]
    fn test_linear_mlecheck() {
        test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 2>(
            |[a, b]| a + b,
            |[_a, _b]| OptimalPackedB128::zero(), // coefficient on the quadratic term is 0
        );
    }

    #[test]
    fn test_bivariate_product_mlecheck() {
        test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 2>(
            |[a, b]| a * b,
            |[a, b]| a * b,
        );
    }

    #[test]
    fn test_mul_gate_mlecheck() {
        test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 3>(
            |[a, b, c]| a * b - c,
            |[a, b, _c]| a * b,
        );
    }

    #[test]
    fn test_4_variate_composition_mlecheck() {
        test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 4>(
            |[a, b, c, d]| (a + b) * (c + d),
            |[a, b, c, d]| (a + b) * (c + d),
        );
    }
}
