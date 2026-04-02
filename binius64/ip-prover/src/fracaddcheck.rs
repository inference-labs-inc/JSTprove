// Copyright 2025-2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_ip::prodcheck::MultilinearEvalClaim;
use binius_math::{FieldBuffer, line::extrapolate_line_packed};
use binius_utils::rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    channel::IPProverChannel,
    sumcheck::{
        Error as SumcheckError,
        batch::batch_prove_mle_and_write_evals,
        common::MleCheckProver,
        frac_add_mle::{self, FractionalBuffer},
    },
};

/// Prover for the fractional addition protocol.
///
/// Each layer is a double of the numerator and denominator values of fractional terms. Each layer
/// represents the addition of siblings with respect to the fractional addition rule:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$
pub struct FracAddCheckProver<P: PackedField> {
    layers: Vec<(FieldBuffer<P>, FieldBuffer<P>)>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(
        "mismatched numerator/denominator lengths: numerator log_len {num_log_len}, denominator log_len {den_log_len}"
    )]
    MismatchedWitnessLengths {
        num_log_len: usize,
        den_log_len: usize,
    },
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumcheckError),
}

impl<F, P> FracAddCheckProver<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    /// Creates a new [`FracAddCheckProver`].
    ///
    /// Returns `(prover, sums)` where `sums` is the final layer containing the
    /// fractional additions over all `k` variables.
    ///
    /// # Arguments
    /// * `k` - The number of variables over which the reduction is taken. Each reduction step
    ///   reduces one variable by computing fractional additions of sibling terms.
    /// * `witness` - The witness numerator/denominator layers
    ///
    /// # Preconditions
    /// * `witness.0.log_len() >= k`
    pub fn new(k: usize, witness: FractionalBuffer<P>) -> (Self, FractionalBuffer<P>) {
        let (witness_num, witness_den) = witness;
        assert!(witness_num.log_len() == witness_den.log_len());
        assert!(witness_num.log_len() >= k);

        let mut layers = Vec::with_capacity(k + 1);
        layers.push((witness_num, witness_den));

        for _ in 0..k {
            let prev_layer = layers.last().expect("layers is non-empty");

            let (num, den) = prev_layer;
            let (num_0, num_1) = num.split_half_ref();
            let (den_0, den_1) = den.split_half_ref();

            let (next_layer_num, next_layer_den) = (
                num_0.as_ref(),
                den_0.as_ref(),
                num_1.as_ref(),
                den_1.as_ref(),
            )
                .into_par_iter()
                .map(|(&a_0, &b_0, &a_1, &b_1)| (a_0 * b_1 + a_1 * b_0, b_0 * b_1))
                .collect::<(Vec<_>, Vec<_>)>();

            let next_layer = (
                FieldBuffer::new(num.log_len() - 1, next_layer_num.into_boxed_slice()),
                FieldBuffer::new(den.log_len() - 1, next_layer_den.into_boxed_slice()),
            );

            layers.push(next_layer);
        }

        let sums = layers.pop().expect("layers has k+1 elements");
        (Self { layers }, sums)
    }

    /// Returns the number of remaining layers to prove.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Pops the last layer and returns a sumcheck prover for it.
    ///
    /// Returns `(layer_prover, remaining)` where:
    /// - `layer_prover` is a sumcheck prover for the popped layer
    /// - `remaining` is `Some(self)` if there are more layers, `None` otherwise
    pub fn layer_prover(
        mut self,
        claim: (MultilinearEvalClaim<F>, MultilinearEvalClaim<F>),
    ) -> Result<(impl MleCheckProver<F>, Option<Self>), Error> {
        let (num_claim, den_claim) = claim;
        assert_eq!(
            num_claim.point, den_claim.point,
            "fractional claims must share the evaluation point"
        );

        let layer = self.layers.pop().expect("layers is non-empty");

        let remaining = if self.layers.is_empty() {
            None
        } else {
            Some(self)
        };

        let (num, den) = layer;
        let (num_0, num_1) = num.split_half_ref();
        let (den_0, den_1) = den.split_half_ref();
        let num_0 = FieldBuffer::new(num_0.log_len(), num_0.as_ref().into());
        let num_1 = FieldBuffer::new(num_1.log_len(), num_1.as_ref().into());
        let den_0 = FieldBuffer::new(den_0.log_len(), den_0.as_ref().into());
        let den_1 = FieldBuffer::new(den_1.log_len(), den_1.as_ref().into());
        let prover = frac_add_mle::new(
            [num_0, num_1, den_0, den_1],
            num_claim.point.clone(),
            [num_claim.eval, den_claim.eval],
        )?;

        Ok((prover, remaining))
    }

    /// Runs the fractional addition check protocol and returns the final evaluation claims.
    ///
    /// This consumes the prover and runs sumcheck reductions from the smallest layer back to
    /// the largest.
    ///
    /// # Arguments
    /// * `claim` - The initial multilinear evaluation claims (numerator, denominator)
    /// * `channel` - The channel for sending prover messages and sampling challenges
    ///
    /// # Preconditions
    /// * `claim.0.point.len() == witness.log_len() - k` (where k is the number of reduction layers)
    pub fn prove(
        self,
        claim: (MultilinearEvalClaim<F>, MultilinearEvalClaim<F>),
        channel: &mut impl IPProverChannel<F>,
    ) -> Result<(MultilinearEvalClaim<F>, MultilinearEvalClaim<F>), Error> {
        let mut prover_opt = Some(self);
        let mut claim = claim;

        while let Some(prover) = prover_opt {
            let (sumcheck_prover, remaining) = prover.layer_prover(claim)?;
            prover_opt = remaining;

            let output = batch_prove_mle_and_write_evals(vec![sumcheck_prover], channel)?;

            let mut multilinear_evals = output.multilinear_evals;
            let evals = multilinear_evals.pop().expect("batch contains one prover");

            let [num_0, num_1, den_0, den_1] = evals
                .try_into()
                .expect("prover evaluates four multilinears");

            let r = channel.sample();

            let next_num = extrapolate_line_packed(num_0, num_1, r);
            let next_den = extrapolate_line_packed(den_0, den_1, r);

            let mut next_point = output.challenges;
            next_point.push(r);

            let num_claim = MultilinearEvalClaim {
                eval: next_num,
                point: next_point.clone(),
            };
            let den_claim = MultilinearEvalClaim {
                eval: next_den,
                point: next_point,
            };

            claim = (num_claim, den_claim);
        }

        Ok(claim)
    }
}

#[cfg(test)]
mod tests {
    use binius_field::PackedField;
    use binius_ip::fracaddcheck;
    use binius_math::{
        multilinear::evaluate::evaluate,
        test_utils::{Packed128b, random_field_buffer, random_scalars},
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

    type StdChallenger = HasherChallenger<sha2::Sha256>;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    fn test_frac_add_check_prove_verify_helper<P: PackedField>(n: usize, k: usize) {
        let mut rng = StdRng::seed_from_u64(0);

        // 1. Create random witness with log_len = n + k
        let witness_num = random_field_buffer::<P>(&mut rng, n + k);
        let witness_den = random_field_buffer::<P>(&mut rng, n + k);

        // 2. Create prover (computes fractional-add layers)
        let (prover, sums) = FracAddCheckProver::new(k, (witness_num.clone(), witness_den.clone()));

        // 3. Generate random n-dimensional challenge point
        let eval_point = random_scalars::<P::Scalar>(&mut rng, n);

        // 4. Evaluate sums at challenge point to createzz claims
        let sum_num_eval = evaluate(&sums.0, &eval_point);
        let sum_den_eval = evaluate(&sums.1, &eval_point);
        let prover_claim = (
            MultilinearEvalClaim {
                eval: sum_num_eval,
                point: eval_point.clone(),
            },
            MultilinearEvalClaim {
                eval: sum_den_eval,
                point: eval_point.clone(),
            },
        );
        let verifier_claim = fracaddcheck::FracAddEvalClaim {
            num_eval: sum_num_eval,
            den_eval: sum_den_eval,
            point: eval_point,
        };

        // 5. Run prover
        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let prover_output = prover
            .prove(prover_claim.clone(), &mut prover_transcript)
            .unwrap();

        // 6. Run verifier
        let mut verifier_transcript = prover_transcript.into_verifier();
        let verifier_output =
            fracaddcheck::verify(k, verifier_claim, &mut verifier_transcript).unwrap();

        // 7. Check outputs match
        assert_eq!(prover_output.0.point, prover_output.1.point);
        assert_eq!(prover_output.0.point, verifier_output.point);
        assert_eq!(prover_output.0.eval, verifier_output.num_eval);
        assert_eq!(prover_output.1.eval, verifier_output.den_eval);

        // 8. Verify multilinear evaluation of original witness
        let expected_num = evaluate(&witness_num, &verifier_output.point);
        let expected_den = evaluate(&witness_den, &verifier_output.point);
        assert_eq!(verifier_output.num_eval, expected_num);
        assert_eq!(verifier_output.den_eval, expected_den);
    }

    #[test]
    fn test_frac_add_check_prove_verify() {
        test_frac_add_check_prove_verify_helper::<Packed128b>(4, 3);
    }

    #[test]
    fn test_frac_add_check_full_prove_verify() {
        test_frac_add_check_prove_verify_helper::<Packed128b>(0, 4);
    }

    fn test_frac_add_check_layer_computation_helper<P: PackedField>(n: usize, k: usize) {
        let mut rng = StdRng::seed_from_u64(0);

        // Create random witness with log_len = n + k
        let witness_num = random_field_buffer::<P>(&mut rng, n + k);
        let witness_den = random_field_buffer::<P>(&mut rng, n + k);

        // Create prover (computes fractional-add layers)
        let (_prover, sums) =
            FracAddCheckProver::new(k, (witness_num.clone(), witness_den.clone()));

        // For each index i in the sums layer, verify it equals the fractional sum of witness values
        // at indices i + z * 2^n for z in 0..2^k (strided access, not contiguous)
        let stride = 1 << n;
        let num_terms = 1 << k;
        for i in 0..(1 << n) {
            let mut expected_num = witness_num.get(i);
            let mut expected_den = witness_den.get(i);
            for z in 1..num_terms {
                let idx = i + z * stride;
                let num_z = witness_num.get(idx);
                let den_z = witness_den.get(idx);
                expected_num = expected_num * den_z + num_z * expected_den;
                expected_den *= den_z;
            }
            let actual_num = sums.0.get(i);
            let actual_den = sums.1.get(i);
            assert_eq!(actual_num, expected_num, "Numerator mismatch at index {i}");
            assert_eq!(
                actual_den, expected_den,
                "Denominator mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_frac_add_check_layer_computation() {
        test_frac_add_check_layer_computation_helper::<Packed128b>(4, 3);
    }
}
