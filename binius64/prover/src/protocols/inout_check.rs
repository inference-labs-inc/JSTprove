// Copyright 2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_math::{
    FieldBuffer, FieldSlice, inner_product::inner_product_buffers, line::extrapolate_line_packed,
    multilinear::fold::fold_highest_var_inplace,
};
use binius_verifier::protocols::sumcheck::RoundCoeffs;

use crate::protocols::sumcheck::{
    Error,
    common::{MleCheckProver, SumcheckProver},
    gruen32::Gruen32,
};

/// An MLE-check prover instance for the argument of public input/witness consistency.
///
/// See [`binius_verifier::protocols::pubcheck::verify`] for protocol details.
pub struct InOutCheckProver<P: PackedField> {
    log_public: usize,
    witness: FieldBuffer<P>,
    last_coeffs_or_eval: RoundCoeffsOrEval<P::Scalar>,
    gruen32: Gruen32<P>,
    zero_padded_eval_point: Vec<P::Scalar>,
}

impl<F: Field, P: PackedField<Scalar = F>> InOutCheckProver<P> {
    /// Creates a new [`InOutCheckProver`] instance.
    ///
    /// # Arguments
    /// * `witness` - An ℓ-variate multilinear polynomial representing the witness
    /// * `inout` - An m-variate multilinear polynomial representing the input/output values, where
    ///   m ≤ ℓ
    /// * `eval_point` - The m-dimensional evaluation point for the inout polynomial (the non-zero
    ///   part of the full evaluation point). The full evaluation point is `eval_point || 0^{ℓ-m}`.
    ///
    /// # Preconditions
    /// * `witness.len() >= inout.len()` - The witness must have at least as many values as inout
    /// * `eval_point.len() == inout.log_len()` - The evaluation point dimension must match the
    ///   number of inout variables
    pub fn new(witness: FieldBuffer<P>, eval_point: &[F]) -> Self {
        assert!(eval_point.len() <= witness.log_len());

        let log_public = eval_point.len();
        let zero_padded_eval_point =
            [eval_point, &vec![F::ZERO; witness.log_len() - log_public]].concat();

        let gruen32 = Gruen32::new(eval_point);
        let public = witness.chunk(log_public, 0);
        let public_eval = Self::evaluate_multilinear_with_gruen32(&gruen32, public);

        Self {
            log_public,
            witness,
            last_coeffs_or_eval: RoundCoeffsOrEval::Eval(public_eval),
            gruen32,
            zero_padded_eval_point,
        }
    }

    /// Computes the round evaluation for the first ℓ-m rounds.
    ///
    /// In these rounds, when the folded witness is larger than the inout polynomial, the evaluation
    /// point coordinate is 0. We handle this case by only summing over 2^m terms. This returns the
    /// evaluation of the round polynomial at 1.
    ///
    /// # Preconditions
    /// * `witness.log_len() > inout.log_len()` - Must be in the early rounds where witness has more
    ///   variables than inout
    fn compute_round_eval_early_rounds(&self) -> F {
        let n_vars = self.log_public;

        // Get the first 2^n_vars values in the upper half of the witness.
        let truncated_witness = self
            .witness
            .chunk(n_vars, 1 << (self.witness.log_len() - n_vars - 1));

        Self::evaluate_multilinear_with_gruen32(&self.gruen32, truncated_witness)
    }

    /// Computes the round evaluation for the last m rounds.
    ///
    /// In these rounds, we run the algorithm for the regular multilinear MLE-check where the
    /// witness and inout polynomials have the same number of variables. This returns the evaluation
    /// of the round polynomial at 1.
    ///
    /// # Preconditions
    /// * `witness.log_len() > 0` - The witness must have at least one variable remaining
    /// * `witness.log_len() == eq_expansion.log_len()` - Must be in the later rounds where both
    ///   have the same number of variables
    fn compute_round_eval_later_rounds(&self) -> F {
        let eq_expansion = self.gruen32.eq_expansion();
        let (_, witness_1) = self.witness.split_half_ref();

        inner_product_buffers(&witness_1, eq_expansion)
    }

    fn evaluate_multilinear_with_gruen32(gruen32: &Gruen32<P>, multilin: FieldSlice<P>) -> F {
        assert_eq!(gruen32.n_vars_remaining(), multilin.log_len());

        if multilin.log_len() == 0 {
            return multilin.get(0);
        }

        // The Gruen32 structure doesn't fully expand the eq tensor because it omits the last
        // variable. In the early rounds, we do want an evaluation of the witness - inout values at
        // the full evaluation point. We work around this by computing the evaluation on the lower
        // and upper halves of the witness and inout vector separately, then extrapolating with the
        // last coordinate of the evaluation point.
        let eq_expansion = gruen32.eq_expansion();
        let (multilin_0, multilin_1) = multilin.split_half_ref();

        let lo = inner_product_buffers(&multilin_0, eq_expansion);
        let hi = inner_product_buffers(&multilin_1, eq_expansion);

        let alpha = gruen32.next_coordinate();
        extrapolate_line_packed(lo, hi, alpha)
    }
}

impl<F, P> SumcheckProver<F> for InOutCheckProver<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    fn n_vars(&self) -> usize {
        self.witness.log_len()
    }

    fn n_claims(&self) -> usize {
        1
    }

    fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
        let RoundCoeffsOrEval::Eval(last_eval) = &self.last_coeffs_or_eval else {
            return Err(Error::ExpectedFold);
        };

        let n_vars_remaining = self.n_vars();
        assert!(n_vars_remaining > 0);

        let (y_0, y_1) = if self.log_public < self.witness.log_len() {
            let y_1 = self.compute_round_eval_early_rounds();

            // The coordinate of the evaluation point in this round is 0, so R(0) = last_eval
            let y_0 = *last_eval;

            (y_0, y_1)
        } else {
            let y_1 = self.compute_round_eval_later_rounds();

            // Compute the round coefficients from the fact that
            // R(1) = y_1
            // R(α) = last_eval
            // ==> y_0 = (sum - y_1 * alpha) / (1 - alpha)
            let alpha = self.gruen32.next_coordinate();
            let y_0 = (*last_eval - y_1 * alpha) * (F::ONE - alpha).invert_or_zero();

            (y_0, y_1)
        };

        // Coefficients for degree 1 polynomial: c_0 + c_1*X
        let c_0 = y_0;
        let c_1 = y_1 - y_0;
        let round_coeffs = RoundCoeffs(vec![c_0, c_1]);

        self.last_coeffs_or_eval = RoundCoeffsOrEval::Coeffs(round_coeffs.clone());
        Ok(vec![round_coeffs])
    }

    fn fold(&mut self, challenge: F) -> Result<(), Error> {
        let RoundCoeffsOrEval::Coeffs(coeffs) = &self.last_coeffs_or_eval else {
            return Err(Error::ExpectedExecute);
        };

        let n_vars = self.n_vars();
        assert!(n_vars > 0);

        let eval = coeffs.evaluate(challenge);

        // Always fold the witness
        fold_highest_var_inplace(&mut self.witness, challenge);

        // Fold gruen32 in the last m rounds
        if n_vars <= self.log_public {
            self.gruen32.fold(challenge);
        }

        self.last_coeffs_or_eval = RoundCoeffsOrEval::Eval(eval);
        Ok(())
    }

    fn finish(self) -> Result<Vec<F>, Error> {
        if self.n_vars() > 0 {
            let error = match self.last_coeffs_or_eval {
                RoundCoeffsOrEval::Coeffs(_) => Error::ExpectedFold,
                RoundCoeffsOrEval::Eval(_) => Error::ExpectedExecute,
            };

            return Err(error);
        }

        // Return only the witness evaluation
        let witness_eval = self.witness.get(0);
        Ok(vec![witness_eval])
    }
}

impl<F, P> MleCheckProver<F> for InOutCheckProver<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    fn eval_point(&self) -> &[F] {
        &self.zero_padded_eval_point[..self.witness.log_len()]
    }
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEval<F: Field> {
    Coeffs(RoundCoeffs<F>),
    Eval(F),
}

#[cfg(test)]
mod tests {
    use binius_field::{
        PackedField,
        arch::{OptimalB128, OptimalPackedB128},
    };
    use binius_math::{
        FieldBuffer, FieldSlice,
        multilinear::evaluate::evaluate,
        test_utils::{random_field_buffer, random_scalars},
    };
    use binius_transcript::ProverTranscript;
    use binius_verifier::{config::StdChallenger, protocols::pubcheck};
    use rand::{SeedableRng, prelude::StdRng};

    use super::*;
    use crate::protocols::sumcheck::prove_single_mlecheck;

    #[test]
    fn test_pubcheck_prove_verify() {
        type F = OptimalB128;
        type P = OptimalPackedB128;

        let n_witness_vars = 8;
        let n_public_vars = 4;
        let mut rng = StdRng::seed_from_u64(0);

        // Generate inout multilinear
        let inout = random_field_buffer::<P>(&mut rng, n_public_vars);

        // Generate witness multilinear that agrees with inout on the first 2^m values
        let mut witness_vec = random_scalars::<F>(&mut rng, 1 << n_witness_vars);
        // Copy inout values to the first 2^m positions of witness (padded with zeros)
        for (i, val) in inout.as_ref().iter().flat_map(|p| p.iter()).enumerate() {
            witness_vec[i] = val;
        }
        let witness = FieldBuffer::<P>::from_values(&witness_vec);

        let eval_point = random_scalars::<F>(&mut rng, n_public_vars);
        let public = FieldSlice::from_slice(n_public_vars, &witness_vec[..1 << n_public_vars]);
        let public_eval = evaluate(&public, &eval_point);

        // Create the prover
        let prover = InOutCheckProver::new(witness.clone(), &eval_point);

        // Run the proving protocol
        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let output = prove_single_mlecheck(prover, &mut prover_transcript).unwrap();

        // Write the multilinear evaluations to the transcript
        prover_transcript
            .message()
            .write_slice(&output.multilinear_evals[..1]);

        // Convert to verifier transcript and run verification
        let mut verifier_transcript = prover_transcript.into_verifier();

        let pubcheck::VerifyOutput {
            eval,
            eval_point: reduced_eval_point,
        } = pubcheck::verify(
            n_witness_vars,
            public_eval,
            &eval_point,
            &mut verifier_transcript,
        )
        .unwrap();

        // Check that the original multilinears evaluate to the claimed values at the challenge.
        let expected_witness_eval = evaluate(&witness, &reduced_eval_point);
        assert_eq!(eval, expected_witness_eval);

        // Also verify the challenges match what the prover saw
        let verifier_challenges = reduced_eval_point.into_iter().rev().collect::<Vec<_>>();
        assert_eq!(
            output.challenges, verifier_challenges,
            "Prover and verifier challenges should match"
        );
    }
}
