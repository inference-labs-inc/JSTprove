// Copyright 2023-2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_math::AsSlicesMut;

use super::{error::Error, quadratic_mle::QuadraticMleCheckProver};
use crate::sumcheck::common::MleCheckProver;

/// Creates an [`MleCheckProver`] that reduces an evaluation claim on a multilinear extension
/// of the product of two multilinears to evaluation claims on said multilinears.
///
/// ## Mathematical Definition
/// * $n \in N$ - number of variables in multilinear polynomials
/// * $A, B \in F\[x\], x = \(x_1, \ldots, x_n\)$ - multilinears being multiplied
/// * $(\widetilde{AB})\[x\] = y$ - evaluation claim on the product MLE
///
/// The claim is equivalent to $P(x) = \sum_{v \in B} \widetilde{eq}(v, x) A(v) B(v) = y$, and the
/// reduction can be achieved by sumchecking the latter degree-3 composition. The paper [Gruen24],
/// however, describes a way to partition the $\widetilde{eq}(v, x)$ into three parts in round $j
/// \in 1, \ldots, n$ during specialization of variable $v_{n-j+1}$, with $j-1$ challenges
/// $\alpha_i$ already sampled:
///
/// $$ \widetilde{eq}(x_{n-j+2}, \ldots, x_n; \alpha_{j-1}, \ldots, \alpha_{1}) \tag{1} $$
/// $$ \widetilde{eq}(x_{n-j+1}; v_{n-j+1}) \tag{2} $$
/// $$ \widetilde{eq}(x_1, \ldots, x_{n-j}; v_1, \ldots, v_{n-j}) \tag{3} $$
///
/// The following holds:
/// * (1) is a constant that can be incrementally updated in O(1) time,
/// * (2) is a linear polynomial that is easy to compute in monomial form specialized to either
///   variable
/// * (3) is a an equality indicator over the claim point suffix
///
/// These observations allow us to instead sumcheck:
/// $$
/// P'(x) = \sum_{v \in B} \widetilde{eq}(x_1, \ldots, x_{n-j}; v_1, \ldots, v_{n-j}) A(v) B(v)
/// $$
///
/// Which is simpler because:
/// * $P'(x)$ is degree-2 in $j$-th variable, requiring one less evaluation point
/// * Equality indicator expansion does not depend on $j$-th variable and thus doesn't need to be
///   interpolated
///
/// After computing the round polynomial for $P'(x)$ in monomial form, one can simply multiply by
/// (2) and (1) in polynomial form. For more details, see [`Gruen32`]('gruen32::Gruen32') struct and
/// [Gruen24] Section 3.2.
///
/// Note 1: as evident from the definition, this prover binds variables in high-to-low index order.
///
/// Note 2: evaluation points are 0 (implicit), 1 and Karatsuba infinity.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub fn new<F, P>(
    multilinears: impl AsSlicesMut<P, 2> + Send + 'static,
    eval_point: Vec<F>,
    eval_claim: F,
) -> Result<impl MleCheckProver<F>, Error>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    QuadraticMleCheckProver::new(
        multilinears,
        |[a, b]| a * b,
        |[a, b]| a * b,
        eval_point,
        eval_claim,
    )
}

#[cfg(test)]
mod tests {
    use binius_field::arch::{OptimalB128, OptimalPackedB128};
    use binius_ip::{mlecheck, sumcheck::verify};
    use binius_math::{
        FieldBuffer,
        multilinear::{eq::eq_ind, evaluate::evaluate},
        test_utils::{random_field_buffer, random_scalars},
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

    type StdChallenger = HasherChallenger<sha2::Sha256>;
    use itertools::{self, Itertools};
    use rand::{SeedableRng, prelude::StdRng};

    use super::*;
    use crate::sumcheck::{MleToSumCheckDecorator, prove::prove_single, prove_single_mlecheck};

    fn test_mlecheck_prove_verify<F, P>(
        prover: impl MleCheckProver<F>,
        eval_claim: F,
        eval_point: &[F],
        multilinear_a: FieldBuffer<P>,
        multilinear_b: FieldBuffer<P>,
    ) where
        F: Field,
        P: PackedField<Scalar = F>,
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
            2, // degree 2 for bivariate product
            eval_claim,
            &mut verifier_transcript,
        )
        .unwrap();

        let mut reduced_eval_point = sumcheck_output.challenges.clone();
        reduced_eval_point.reverse();

        // Read the multilinear evaluations from the transcript
        let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

        // Check that the product of the evaluations equals the reduced evaluation
        assert_eq!(
            multilinear_evals[0] * multilinear_evals[1],
            sumcheck_output.eval,
            "Product of multilinear evaluations should equal the reduced evaluation"
        );

        // Check that the original multilinears evaluate to the claimed values at the challenge
        // point The prover binds variables from high to low, but evaluate expects them from low
        // to high
        let eval_a = evaluate(&multilinear_a, &reduced_eval_point);
        let eval_b = evaluate(&multilinear_b, &reduced_eval_point);

        assert_eq!(
            eval_a, multilinear_evals[0],
            "Multilinear A should evaluate to the first claimed evaluation"
        );
        assert_eq!(
            eval_b, multilinear_evals[1],
            "Multilinear B should evaluate to the second claimed evaluation"
        );

        // Also verify the challenges match what the prover saw
        assert_eq!(
            output.challenges, sumcheck_output.challenges,
            "Prover and verifier challenges should match"
        );
    }

    fn test_wrapped_sumcheck_prove_verify<F, P>(
        mlecheck_prover: impl MleCheckProver<F>,
        eval_claim: F,
        eval_point: &[F],
        multilinear_a: FieldBuffer<P>,
        multilinear_b: FieldBuffer<P>,
    ) where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        let n_vars = mlecheck_prover.n_vars();
        let prover = MleToSumCheckDecorator::new(mlecheck_prover);

        // Run the proving protocol
        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let output = prove_single(prover, &mut prover_transcript).unwrap();

        // Write the multilinear evaluations to the transcript
        prover_transcript
            .message()
            .write_slice(&output.multilinear_evals);

        // Convert to verifier transcript and run verification
        let mut verifier_transcript = prover_transcript.into_verifier();
        let sumcheck_output = verify(
            n_vars,
            3, // degree 3 for trivariate product (bivariate by equality indicator)
            eval_claim,
            &mut verifier_transcript,
        )
        .unwrap();

        // The prover binds variables from high to low, but evaluate expects them from low
        // to high
        let mut reduced_eval_point = sumcheck_output.challenges.clone();
        reduced_eval_point.reverse();

        // Read the multilinear evaluations from the transcript
        let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

        // Evaluate the equality indicator
        let eq_ind_eval = eq_ind(eval_point, &reduced_eval_point);

        // Check that the product of the evaluations equals the reduced evaluation
        assert_eq!(
            multilinear_evals[0] * multilinear_evals[1] * eq_ind_eval,
            sumcheck_output.eval,
            "Product of multilinear evaluations should equal the reduced evaluation"
        );

        // Check that the original multilinears evaluate to the claimed values at the challenge
        // point
        let eval_a = evaluate(&multilinear_a, &reduced_eval_point);
        let eval_b = evaluate(&multilinear_b, &reduced_eval_point);

        assert_eq!(
            eval_a, multilinear_evals[0],
            "Multilinear A should evaluate to the first claimed evaluation"
        );
        assert_eq!(
            eval_b, multilinear_evals[1],
            "Multilinear B should evaluate to the second claimed evaluation"
        );

        // Also verify the challenges match what the prover saw
        assert_eq!(
            output.challenges, sumcheck_output.challenges,
            "Prover and verifier challenges should match"
        );
    }

    #[test]
    fn test_bivariate_product_mlecheck() {
        type F = OptimalB128;
        type P = OptimalPackedB128;

        let n_vars = 8;
        let mut rng = StdRng::seed_from_u64(0);

        // Generate two random multilinear polynomials
        let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
        let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

        // Compute product multilinear
        let product = itertools::zip_eq(multilinear_a.as_ref(), multilinear_b.as_ref())
            .map(|(&l, &r)| l * r)
            .collect_vec();
        let product_buffer = FieldBuffer::new(n_vars, product);

        let eval_point = random_scalars::<F>(&mut rng, n_vars);
        let eval_claim = evaluate(&product_buffer, &eval_point);

        // Create the prover
        let mlecheck_prover = new(
            [multilinear_a.clone(), multilinear_b.clone()],
            eval_point.clone(),
            eval_claim,
        )
        .unwrap();

        test_mlecheck_prove_verify(
            mlecheck_prover,
            eval_claim,
            &eval_point,
            multilinear_a.clone(),
            multilinear_b.clone(),
        );

        // Create another prover for the wrapped test
        let mlecheck_prover = new(
            [multilinear_a.clone(), multilinear_b.clone()],
            eval_point.clone(),
            eval_claim,
        )
        .unwrap();

        test_wrapped_sumcheck_prove_verify(
            mlecheck_prover,
            eval_claim,
            &eval_point,
            multilinear_a.clone(),
            multilinear_b.clone(),
        );
    }
}
