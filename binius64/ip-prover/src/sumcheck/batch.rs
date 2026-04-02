// Copyright 2024-2025 Irreducible Inc.

use binius_field::Field;
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};

use crate::{
    channel::IPProverChannel,
    sumcheck::{
        common::{MleCheckProver, SumcheckProver},
        error::Error,
    },
};

/// Prover view of the execution result of a batched sumcheck.
#[derive(Debug, PartialEq, Eq)]
pub struct BatchSumcheckOutput<F: Field> {
    /// Verifier challenges for each round of the sumcheck protocol.
    ///
    /// One challenge is generated per variable in the multivariate polynomial,
    /// with challenges\[i\] corresponding to the i-th round of the protocol.
    ///
    /// Note: reverse when folding high-to-low to obtain evaluation claim.
    pub challenges: Vec<F>,
    /// Evaluation claims on non-transparent multilinears, per prover.
    ///
    /// Each inner vector contains the evaluation values for one prover's
    /// multilinear polynomials at the challenge point.
    pub multilinear_evals: Vec<Vec<F>>,
}

/// Prove a batched sumcheck protocol execution, where all provers have the same number of rounds.
///
/// The batched sumcheck reduces a set of claims about the sums of multivariate polynomials over
/// the boolean hypercube to their evaluation at a (shared) challenge point. This is achieved by
/// constructing an `n_vars + 1`-variate polynomial whose coefficients in the "new variable" are the
/// individual sum claims and evaluating it at a random point. Due to linearity of sums each claim
/// can be proven separately with an individual [`SumcheckProver`] followed by weighted summation of
/// the round polynomials.
///
/// This function performs the sumcheck protocol and returns the challenges and evaluation claims,
/// but does not write the evaluation claims to the channel. Use [`batch_prove_and_write_evals`]
/// if you need to write the evaluations to the channel.
pub fn batch_prove<F, Prover>(
    mut provers: Vec<Prover>,
    channel: &mut impl IPProverChannel<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
    F: Field,
    Prover: SumcheckProver<F>,
{
    let Some(first_prover) = provers.first() else {
        return Ok(BatchSumcheckOutput {
            challenges: Vec::new(),
            multilinear_evals: Vec::new(),
        });
    };

    let n_vars = first_prover.n_vars();

    if provers.iter().any(|prover| prover.n_vars() != n_vars) {
        return Err(Error::ProverRoundCountMismatch);
    }

    // Random linear-combination coefficient for batching multiple sum claims.
    let batch_coeff = channel.sample();

    let mut challenges = Vec::with_capacity(n_vars);
    for _ in 0..n_vars {
        let mut all_round_coeffs = Vec::new();

        for prover in &mut provers {
            // Each prover emits its round polynomial; we batch across provers.
            all_round_coeffs.extend(prover.execute()?);
        }

        // Horner-fold round polynomials into a single batched polynomial.
        let batched_round_coeffs = all_round_coeffs
            .into_iter()
            .rfold(RoundCoeffs::default(), |acc, coeffs| {
                acc * batch_coeff + &coeffs
            });

        // Truncate to the verifier-visible coefficients for this round.
        let round_proof = batched_round_coeffs.truncate();

        // Commit to the round polynomial, then sample the next challenge.
        channel.send_many(round_proof.coeffs());

        let challenge = channel.sample();
        challenges.push(challenge);

        // Fold all provers on the shared challenge to advance the state machine.
        for prover in &mut provers {
            prover.fold(challenge)?;
        }
    }

    // TODO: this differs from prove_single, which doesn't reverse.
    // Reverse to match high-to-low folding order for evaluation points.
    challenges.reverse();

    let multilinear_evals = provers
        .into_iter()
        .map(|prover| prover.finish())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(BatchSumcheckOutput {
        challenges,
        multilinear_evals,
    })
}

/// Prove a batched sumcheck protocol and write evaluation claims to the channel.
///
/// This function combines [`batch_prove`] with writing the evaluation claims to the channel.
/// It performs the batched sumcheck protocol execution and then writes all the multilinear
/// evaluation values to the channel in order.
///
/// # Arguments
///
/// * `provers` - Vector of sumcheck provers, each handling one claim in the batch
/// * `channel` - The channel for sending prover messages and sampling challenges
///
/// # Returns
///
/// Returns [`BatchSumcheckOutput`] containing the challenges and evaluation claims that were
/// written to the channel.
pub fn batch_prove_and_write_evals<F, Prover>(
    provers: Vec<Prover>,
    channel: &mut impl IPProverChannel<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
    F: Field,
    Prover: SumcheckProver<F>,
{
    let output = batch_prove(provers, channel)?;

    for evals in &output.multilinear_evals {
        // Preserve per-prover ordering when emitting evaluation claims.
        channel.send_many(evals);
    }
    Ok(output)
}

/// Prove a batched sumcheck for MLE-check provers sharing a common evaluation point.
///
/// This is the MLE-check analog of [`batch_prove`]: all provers are [`MleCheckProver`]s and must
/// agree on the same evaluation point, so the batched protocol can fold every prover with the
/// same per-round challenge and reduce all evaluation claims via a single batching coefficient.
pub fn batch_prove_mle<F, MleCheckProver_>(
    mut provers: Vec<MleCheckProver_>,
    channel: &mut impl IPProverChannel<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
    F: Field,
    MleCheckProver_: MleCheckProver<F>,
{
    let Some(first_prover) = provers.first() else {
        return Ok(BatchSumcheckOutput {
            challenges: Vec::new(),
            multilinear_evals: Vec::new(),
        });
    };

    let n_vars = first_prover.n_vars();
    let eval_point = first_prover.eval_point();

    if provers.iter().any(|prover| prover.n_vars() != n_vars) {
        return Err(Error::ProverRoundCountMismatch);
    }

    if provers
        .iter()
        .any(|prover| prover.eval_point() != eval_point)
    {
        // All MLE-check provers must share the same evaluation point to batch safely.
        return Err(Error::EvalPointLengthMismatch);
    }
    // Random linear-combination coefficient for batching multiple eval claims.
    let batch_coeff = channel.sample();

    let mut challenges = Vec::with_capacity(n_vars);
    for _ in 0..n_vars {
        let mut all_round_coeffs = Vec::new();

        for prover in &mut provers {
            // Each prover emits its round polynomial; we batch across provers.
            all_round_coeffs.extend(prover.execute()?);
        }

        // Horner-fold round polynomials into a single batched polynomial.
        let batched_round_coeffs = all_round_coeffs
            .into_iter()
            .rfold(RoundCoeffs::default(), |acc, coeffs| {
                acc * batch_coeff + &coeffs
            });

        // MLE-check uses a distinct round proof type.
        let round_proof = mlecheck::RoundProof::truncate(batched_round_coeffs);

        // Commit to the round polynomial, then sample the next challenge.
        channel.send_many(round_proof.coeffs());

        let challenge = channel.sample();
        challenges.push(challenge);

        // Fold all provers on the shared challenge to advance the state machine.
        for prover in &mut provers {
            prover.fold(challenge)?;
        }
    }

    // TODO: this differs from prove_single, which doesn't reverse.
    // Reverse to match high-to-low folding order for evaluation points.
    challenges.reverse();

    let multilinear_evals = provers
        .into_iter()
        .map(|prover| prover.finish())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(BatchSumcheckOutput {
        challenges,
        multilinear_evals,
    })
}

pub fn batch_prove_mle_and_write_evals<F, MleCheckProver_>(
    provers: Vec<MleCheckProver_>,
    channel: &mut impl IPProverChannel<F>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
    F: Field,
    MleCheckProver_: MleCheckProver<F>,
{
    let output = batch_prove_mle(provers, channel)?;

    for evals in &output.multilinear_evals {
        // Preserve per-prover ordering when emitting evaluation claims.
        channel.send_many(evals);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use binius_field::{
        Field, PackedField,
        arch::{OptimalB128, OptimalPackedB128},
    };
    use binius_ip::sumcheck::batch_verify_mle;
    use binius_math::{
        FieldBuffer,
        multilinear::evaluate::evaluate,
        test_utils::{random_field_buffer, random_scalars},
        univariate::evaluate_univariate,
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

    type StdChallenger = HasherChallenger<sha2::Sha256>;
    use rand::{SeedableRng, prelude::StdRng};

    use super::{Error, batch_prove_mle};
    use crate::sumcheck::bivariate_product_mle;

    fn product_eval_claim<F, P>(
        multilinear_a: &FieldBuffer<P>,
        multilinear_b: &FieldBuffer<P>,
        eval_point: &[F],
    ) -> F
    where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        let n_vars = eval_point.len();
        let product = multilinear_a
            .as_ref()
            .iter()
            .zip(multilinear_b.as_ref())
            .map(|(&l, &r)| l * r)
            .collect::<Vec<_>>();
        let product_buffer = FieldBuffer::new(n_vars, product.into_boxed_slice());
        evaluate(&product_buffer, eval_point)
    }

    #[test]
    fn test_batch_prove_verify_mlecheck() {
        type F = OptimalB128;
        type P = OptimalPackedB128;

        let n_vars = 6;
        let mut rng = StdRng::seed_from_u64(0);

        let eval_point = random_scalars::<F>(&mut rng, n_vars);

        let multilinear_a_0 = random_field_buffer::<P>(&mut rng, n_vars);
        let multilinear_b_0 = random_field_buffer::<P>(&mut rng, n_vars);
        let eval_claim_0 = product_eval_claim(&multilinear_a_0, &multilinear_b_0, &eval_point);

        let multilinear_a_1 = random_field_buffer::<P>(&mut rng, n_vars);
        let multilinear_b_1 = random_field_buffer::<P>(&mut rng, n_vars);
        let eval_claim_1 = product_eval_claim(&multilinear_a_1, &multilinear_b_1, &eval_point);

        let prover_0 = bivariate_product_mle::new(
            [multilinear_a_0.clone(), multilinear_b_0.clone()],
            eval_point.clone(),
            eval_claim_0,
        )
        .unwrap();
        let prover_1 = bivariate_product_mle::new(
            [multilinear_a_1.clone(), multilinear_b_1.clone()],
            eval_point.clone(),
            eval_claim_1,
        )
        .unwrap();

        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let output = batch_prove_mle(vec![prover_0, prover_1], &mut prover_transcript).unwrap();

        let mut writer = prover_transcript.message();
        for evals in &output.multilinear_evals {
            writer.write_scalar_slice(evals);
        }

        let mut verifier_transcript = prover_transcript.into_verifier();
        let sumcheck_output = batch_verify_mle(
            &eval_point,
            2, // degree 2 for bivariate product MLE-check
            &[eval_claim_0, eval_claim_1],
            &mut verifier_transcript,
        )
        .unwrap();

        let mut reduced_eval_point = sumcheck_output.challenges.clone();
        reduced_eval_point.reverse();

        let flattened_evals: Vec<F> = output
            .multilinear_evals
            .iter()
            .flat_map(|evals| evals.iter().copied())
            .collect();
        let evals_from_transcript: Vec<F> = verifier_transcript.message().read_vec(4).unwrap();
        assert_eq!(
            flattened_evals, evals_from_transcript,
            "Multilinear evaluations should round-trip through the transcript"
        );

        let eval_a_0 = evaluate(&multilinear_a_0, &reduced_eval_point);
        let eval_b_0 = evaluate(&multilinear_b_0, &reduced_eval_point);
        let eval_a_1 = evaluate(&multilinear_a_1, &reduced_eval_point);
        let eval_b_1 = evaluate(&multilinear_b_1, &reduced_eval_point);

        assert_eq!(eval_a_0, output.multilinear_evals[0][0]);
        assert_eq!(eval_b_0, output.multilinear_evals[0][1]);
        assert_eq!(eval_a_1, output.multilinear_evals[1][0]);
        assert_eq!(eval_b_1, output.multilinear_evals[1][1]);

        let composed_evals = vec![eval_a_0 * eval_b_0, eval_a_1 * eval_b_1];
        let expected_batched_eval =
            evaluate_univariate(&composed_evals, sumcheck_output.batch_coeff);

        assert_eq!(
            expected_batched_eval, sumcheck_output.eval,
            "Batched evaluation should match reduced evaluation"
        );

        let mut prover_challenges = output.challenges.clone();
        prover_challenges.reverse();
        assert_eq!(
            prover_challenges, sumcheck_output.challenges,
            "Prover and verifier challenges should match"
        );
    }

    #[test]
    fn test_batch_prove_mle_rejects_mismatched_eval_points() {
        type F = OptimalB128;
        type P = OptimalPackedB128;

        let n_vars = 4;
        let mut rng = StdRng::seed_from_u64(0);

        let eval_point = random_scalars::<F>(&mut rng, n_vars);
        let mut other_point = eval_point.clone();
        other_point[0] += F::ONE;

        let multilinear_a_0 = random_field_buffer::<P>(&mut rng, n_vars);
        let multilinear_b_0 = random_field_buffer::<P>(&mut rng, n_vars);
        let eval_claim_0 = product_eval_claim(&multilinear_a_0, &multilinear_b_0, &eval_point);

        let multilinear_a_1 = random_field_buffer::<P>(&mut rng, n_vars);
        let multilinear_b_1 = random_field_buffer::<P>(&mut rng, n_vars);
        let eval_claim_1 = product_eval_claim(&multilinear_a_1, &multilinear_b_1, &other_point);

        let prover_0 = bivariate_product_mle::new(
            [multilinear_a_0, multilinear_b_0],
            eval_point,
            eval_claim_0,
        )
        .unwrap();
        let prover_1 = bivariate_product_mle::new(
            [multilinear_a_1, multilinear_b_1],
            other_point,
            eval_claim_1,
        )
        .unwrap();

        let mut transcript = ProverTranscript::new(StdChallenger::default());
        let err = batch_prove_mle(vec![prover_0, prover_1], &mut transcript).unwrap_err();
        assert!(
            matches!(err, Error::EvalPointLengthMismatch),
            "Expected eval point mismatch error"
        );
    }

    #[test]
    fn test_batch_prove_mle_rejects_mismatched_round_counts() {
        type F = OptimalB128;
        type P = OptimalPackedB128;

        let mut rng = StdRng::seed_from_u64(0);

        let eval_point_a = random_scalars::<F>(&mut rng, 4);
        let eval_point_b = random_scalars::<F>(&mut rng, 5);

        let multilinear_a_0 = random_field_buffer::<P>(&mut rng, 4);
        let multilinear_b_0 = random_field_buffer::<P>(&mut rng, 4);
        let eval_claim_0 = product_eval_claim(&multilinear_a_0, &multilinear_b_0, &eval_point_a);

        let multilinear_a_1 = random_field_buffer::<P>(&mut rng, 5);
        let multilinear_b_1 = random_field_buffer::<P>(&mut rng, 5);
        let eval_claim_1 = product_eval_claim(&multilinear_a_1, &multilinear_b_1, &eval_point_b);

        let prover_0 = bivariate_product_mle::new(
            [multilinear_a_0, multilinear_b_0],
            eval_point_a,
            eval_claim_0,
        )
        .unwrap();
        let prover_1 = bivariate_product_mle::new(
            [multilinear_a_1, multilinear_b_1],
            eval_point_b,
            eval_claim_1,
        )
        .unwrap();

        let mut transcript = ProverTranscript::new(StdChallenger::default());
        let err = batch_prove_mle(vec![prover_0, prover_1], &mut transcript).unwrap_err();
        assert!(
            matches!(err, Error::ProverRoundCountMismatch),
            "Expected prover round count mismatch error"
        );
    }
}
