use std::io::{Cursor, Read};

use arith::{ExtensionField, SimdField};
use circuit::Circuit;
use gkr_engine::{ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, Transcript};
use sumcheck::SUMCHECK_GKR_SIMD_MPI_DEGREE;
use transcript::RandomTape;

// ================ Structured Claims ================
#[derive(Clone, Debug, Default)]
pub struct SumcheckClaim<F: FieldEngine> {
    pub challenge: ExpanderDualVarChallenge<F>,
    pub claim_x: F::ChallengeField,
    pub alpha: Option<F::ChallengeField>,
    pub claim_y: Option<F::ChallengeField>,
}

// ================ Verification Unit ================
#[derive(Clone, Debug, Default)]
pub struct SumcheckVerificationUnit<F: FieldEngine> {
    pub random_tape: RandomTape<F::ChallengeField>,
    pub claim: SumcheckClaim<F>,
    pub proof: Vec<u8>,
}

/// Read a challenge field from the proof reader and
///   1. Append the bytes to the proof_bytes vector.
///   2. Append the field element to the transcript.
#[inline(always)]
pub fn parse_challenge_field<ChallengeF: ExtensionField>(
    mut proof_reader: impl Read,
    transcript: &mut impl Transcript,
    proof_bytes: &mut Vec<u8>,
) -> ChallengeF {
    let mut buffer = vec![0; ChallengeF::SIZE];
    proof_reader.read_exact(&mut buffer).unwrap();
    proof_bytes.extend_from_slice(&buffer);
    let challenge = ChallengeF::deserialize_from(Cursor::new(buffer)).unwrap();
    transcript.append_field_element(&challenge);
    challenge
}

pub fn parse_sumcheck_rounds<F: FieldEngine>(
    mut proof_reader: impl Read,
    n_rounds: usize,
    degree: usize,
    transcript: &mut impl Transcript,
    challenge_vec: &mut Vec<F::ChallengeField>,
    proof_bytes: &mut Vec<u8>,
    random_tape: &mut RandomTape<F::ChallengeField>,
) {
    challenge_vec.clear();
    (0..n_rounds).for_each(|_| {
        (0..degree + 1).for_each(|_| {
            parse_challenge_field::<F::ChallengeField>(&mut proof_reader, transcript, proof_bytes);
        });

        challenge_vec.push(transcript.generate_field_element());
    });
    random_tape.tape.extend_from_slice(challenge_vec);
}

#[allow(clippy::type_complexity)]
/// Parse the proof into a vector of verification units.
pub fn parse_proof<F: FieldEngine>(
    mut proof_reader: impl Read,
    circuit: &Circuit<F>,
    proving_time_mpi_size: usize,
    xy_var_degree: usize,
    claimed_v: F::ChallengeField,
    transcript: &mut impl Transcript,
) -> (
    Vec<SumcheckVerificationUnit<F>>,
    ExpanderDualVarChallenge<F>,
    F::ChallengeField,
    Option<F::ChallengeField>,
) {
    let mut verification_units =
        vec![SumcheckVerificationUnit::<F>::default(); circuit.layers.len()];
    let n_output_vars = circuit.layers.last().unwrap().output_var_num;
    let n_simd_vars = <F::SimdCircuitField as SimdField>::PACK_SIZE.trailing_zeros() as usize;
    let n_mpi_vars = proving_time_mpi_size.trailing_zeros() as usize;

    let mut challenge: ExpanderDualVarChallenge<F> =
        ExpanderSingleVarChallenge::sample_from_transcript(
            transcript,
            n_output_vars,
            proving_time_mpi_size,
        )
        .into();
    let mut claim_x = claimed_v;
    let mut alpha = None;
    let mut claim_y = None;

    for i in (0..circuit.layers.len()).rev() {
        let verification_unit = &mut verification_units[i];
        verification_unit.claim = SumcheckClaim {
            challenge: challenge.clone(),
            claim_x,
            alpha,
            claim_y,
        };
        challenge = ExpanderDualVarChallenge::default(); // reset challenge for next layer

        let layer = &circuit.layers[i];
        let sumcheck_proof = &mut verification_unit.proof;
        let random_tape = &mut verification_unit.random_tape;
        let n_vars = layer.input_var_num;

        parse_sumcheck_rounds::<F>(
            &mut proof_reader,
            n_vars,
            xy_var_degree,
            transcript,
            &mut challenge.rz_0,
            sumcheck_proof,
            random_tape,
        );

        parse_sumcheck_rounds::<F>(
            &mut proof_reader,
            n_simd_vars,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut challenge.r_simd,
            sumcheck_proof,
            random_tape,
        );

        parse_sumcheck_rounds::<F>(
            &mut proof_reader,
            n_mpi_vars,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut challenge.r_mpi,
            sumcheck_proof,
            random_tape,
        );

        claim_x = parse_challenge_field::<F::ChallengeField>(
            &mut proof_reader,
            transcript,
            sumcheck_proof,
        );

        if !layer.structure_info.skip_sumcheck_phase_two {
            challenge.rz_1 = Some(vec![]);
            parse_sumcheck_rounds::<F>(
                &mut proof_reader,
                n_vars,
                xy_var_degree,
                transcript,
                challenge.rz_1.as_mut().unwrap(),
                sumcheck_proof,
                random_tape,
            );
            claim_y = Some(parse_challenge_field::<F::ChallengeField>(
                &mut proof_reader,
                transcript,
                sumcheck_proof,
            ));
        } else {
            claim_y = None;
        }

        alpha = if challenge.rz_1.is_some() {
            let alpha = transcript.generate_field_element();
            random_tape.tape.push(alpha);
            Some(alpha)
        } else {
            None
        };
    }

    (verification_units, challenge, claim_x, claim_y)
}
