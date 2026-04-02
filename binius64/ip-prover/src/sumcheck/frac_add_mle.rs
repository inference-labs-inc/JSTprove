// Copyright 2025-2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;

use super::error::Error;
use crate::sumcheck::{batch_quadratic_mle::BatchQuadraticMleCheckProver, common::MleCheckProver};

pub type FractionalBuffer<P> = (FieldBuffer<P>, FieldBuffer<P>);
// Prover for the fractional additional claims required in LogUp*. We keep numerators and
// denominators to be added in a single buffer respectively, with the assumption that the 2
// collections to be added are in either half.
pub fn new<F, P>(
    fraction: [FieldBuffer<P>; 4],
    eval_point: Vec<F>,
    eval_claims: [F; 2],
) -> Result<impl MleCheckProver<F>, Error>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    BatchQuadraticMleCheckProver::new(
        fraction,
        |[num_a, num_b, den_a, den_b], out| {
            out[0] = num_a * den_b + num_b * den_a;
            out[1] = den_a * den_b;
        },
        |[num_a, num_b, den_a, den_b], out| {
            // The fractional addition formulas are purely quadratic, so the infinity composition
            // matches the regular composition.
            out[0] = num_a * den_b + num_b * den_a;
            out[1] = den_a * den_b;
        },
        eval_point,
        eval_claims,
    )
}

#[cfg(test)]
mod tests {
    use binius_field::arch::{OptimalB128, OptimalPackedB128};
    use binius_ip::sumcheck::batch_verify;
    use binius_math::{
        FieldBuffer,
        multilinear::{eq::eq_ind, evaluate::evaluate},
        test_utils::{random_field_buffer, random_scalars},
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

    type StdChallenger = HasherChallenger<sha2::Sha256>;
    use itertools::{Itertools, izip};
    use rand::{SeedableRng, prelude::StdRng};

    use super::*;
    use crate::sumcheck::{MleToSumCheckDecorator, batch::batch_prove, common::SumcheckProver};

    fn test_frac_add_sumcheck_prove_verify<F, P>(
        prover: impl SumcheckProver<F>,
        eval_claims: [F; 2],
        eval_point: &[F],
        num_a: FieldBuffer<P>,
        num_b: FieldBuffer<P>,
        den_a: FieldBuffer<P>,
        den_b: FieldBuffer<P>,
    ) where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        let n_vars = prover.n_vars();
        // Run the proving protocol
        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let output = batch_prove(vec![prover], &mut prover_transcript).unwrap();

        assert_eq!(output.multilinear_evals.len(), 1);
        let prover_evals = output.multilinear_evals[0].clone();

        // Write the multilinear evaluations to the transcript
        prover_transcript
            .message()
            .write_scalar_slice(&prover_evals);

        // Convert to verifier transcript and run verification
        let mut verifier_transcript = prover_transcript.into_verifier();
        let sumcheck_output =
		// Degree 3 because quadratic prime polynomials are multiplied by a linear eq term.
		batch_verify(n_vars, 3, &eval_claims, &mut verifier_transcript).unwrap();

        // The prover binds variables from high to low, but evaluate expects them from low to high
        let mut reduced_eval_point = sumcheck_output.challenges.clone();
        reduced_eval_point.reverse();

        // Read the multilinear evaluations from the transcript
        let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(4).unwrap();

        // Evaluate the equality indicator
        let eq_ind_eval = eq_ind(eval_point, &reduced_eval_point);

        // Check that the original multilinears evaluate to the claimed values at the challenge
        // point
        let eval_num_a = evaluate(&num_a, &reduced_eval_point);
        let eval_den_a = evaluate(&den_a, &reduced_eval_point);
        let eval_num_b = evaluate(&num_b, &reduced_eval_point);
        let eval_den_b = evaluate(&den_b, &reduced_eval_point);

        assert_eq!(
            eval_num_a, multilinear_evals[0],
            "Numerator A should evaluate to the first claimed evaluation"
        );

        assert_eq!(
            eval_num_b, multilinear_evals[1],
            "Numerator B should evaluate to the second claimed evaluation"
        );
        assert_eq!(
            eval_den_a, multilinear_evals[2],
            "Denominator A should evaluate to the third claimed evaluation"
        );

        assert_eq!(
            eval_den_b, multilinear_evals[3],
            "Denominator B should evaluate to the fourth claimed evaluation"
        );

        // Check that the batched evaluation matches the sumcheck output
        // Sumcheck wraps the prime polynomial with an eq factor, so include eq_ind_eval here.
        let numerator_eval = (eval_num_a * eval_den_b + eval_num_b * eval_den_a) * eq_ind_eval;
        let denominator_eval = (eval_den_a * eval_den_b) * eq_ind_eval;
        let batched_eval = numerator_eval + denominator_eval * sumcheck_output.batch_coeff;

        assert_eq!(
            batched_eval, sumcheck_output.eval,
            "Batched evaluation should equal the reduced evaluation"
        );

        // Also verify the challenges match what the prover saw
        let mut prover_challenges = output.challenges.clone();
        prover_challenges.reverse();
        assert_eq!(
            prover_challenges, sumcheck_output.challenges,
            "Prover and verifier challenges should match"
        );
    }

    #[test]
    fn test_frac_add_sumcheck() {
        type F = OptimalB128;
        type P = OptimalPackedB128;

        let n_vars = 8;
        let mut rng = StdRng::seed_from_u64(0);

        let num_a = random_field_buffer::<P>(&mut rng, n_vars);
        let num_b = random_field_buffer::<P>(&mut rng, n_vars);
        let den_a = random_field_buffer::<P>(&mut rng, n_vars);
        let den_b = random_field_buffer::<P>(&mut rng, n_vars);

        let numerator_values = izip!(
            num_a.as_ref(),
            den_a.as_ref(),
            num_b.as_ref(),
            den_b.as_ref()
        )
        .map(|(&num_a, &den_a, &num_b, &den_b)| num_a * den_b + num_b * den_a)
        .collect_vec();

        let denominator_values = izip!(den_a.as_ref(), den_b.as_ref())
            .map(|(&den_a, &den_b)| den_a * den_b)
            .collect_vec();

        let numerator_buffer = FieldBuffer::new(n_vars, numerator_values);
        let denominator_buffer = FieldBuffer::new(n_vars, denominator_values);

        let eval_point = random_scalars::<F>(&mut rng, n_vars);
        // Claims are at the original eval_point; verifier handles challenge ordering separately.
        let eval_claims = [
            evaluate(&numerator_buffer, &eval_point),
            evaluate(&denominator_buffer, &eval_point),
        ];

        let frac_prover = new(
            [num_a.clone(), num_b.clone(), den_a.clone(), den_b.clone()],
            eval_point.clone(),
            eval_claims,
        )
        .unwrap();

        // Wrap the MLE-check prover so it emits sumcheck-compatible round polynomials.
        let prover = MleToSumCheckDecorator::new(frac_prover);

        test_frac_add_sumcheck_prove_verify(
            prover,
            eval_claims,
            &eval_point,
            num_a,
            num_b,
            den_a,
            den_b,
        );
    }
}
