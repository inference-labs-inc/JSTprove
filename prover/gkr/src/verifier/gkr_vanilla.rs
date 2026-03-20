use std::io::Read;

use circuit::{Circuit, RndCoefMap};
use gkr_engine::{ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, Transcript};
use sumcheck::VerifierScratchPad;
use utils::timer::Timer;

use super::common::{sumcheck_verify_gkr_layer, sumcheck_verify_gkr_layer_ref};

#[allow(clippy::type_complexity)]
pub fn gkr_verify<F: FieldEngine>(
    proving_time_mpi_size: usize,
    circuit: &Circuit<F>,
    public_input: &[F::SimdCircuitField],
    claimed_v: &F::ChallengeField,
    transcript: &mut impl Transcript,
    mut proof_reader: impl Read,
) -> (
    bool,
    ExpanderDualVarChallenge<F>,
    F::ChallengeField,
    Option<F::ChallengeField>,
) {
    let timer = Timer::new("gkr_verify", true);
    let mut sp = VerifierScratchPad::<F>::new(circuit, proving_time_mpi_size);

    let layer_num = circuit.layers.len();

    let mut challenge = ExpanderSingleVarChallenge::sample_from_transcript(
        transcript,
        circuit.layers.last().unwrap().output_var_num,
        proving_time_mpi_size,
    )
    .into();

    let mut alpha = None;
    let mut claimed_v0 = *claimed_v;
    let mut claimed_v1 = None;

    transcript.lock_proof();
    transcript.append_field_element(claimed_v);
    transcript.unlock_proof();

    let mut verified = true;
    for i in (0..layer_num).rev() {
        let cur_verified = sumcheck_verify_gkr_layer(
            proving_time_mpi_size,
            &circuit.layers[i],
            public_input,
            &mut challenge,
            &mut claimed_v0,
            &mut claimed_v1,
            alpha,
            &mut proof_reader,
            transcript,
            &mut sp,
            i == layer_num - 1,
            false,
        );

        verified &= cur_verified;
        alpha = if challenge.rz_1.is_some() {
            Some(transcript.generate_field_element::<F::ChallengeField>())
        } else {
            None
        };
    }
    timer.stop();
    let challenge = ExpanderDualVarChallenge::new(
        challenge.rz_0,
        challenge.rz_1,
        challenge.r_simd,
        challenge.r_mpi,
    );

    (verified, challenge, claimed_v0, claimed_v1)
}

#[allow(clippy::type_complexity)]
pub fn gkr_verify_ref<F: FieldEngine>(
    proving_time_mpi_size: usize,
    circuit: &Circuit<F>,
    public_input: &[F::SimdCircuitField],
    claimed_v: &F::ChallengeField,
    transcript: &mut impl Transcript,
    mut proof_reader: impl Read,
    rnd_map: &RndCoefMap<F::CircuitField>,
) -> (
    bool,
    ExpanderDualVarChallenge<F>,
    F::ChallengeField,
    Option<F::ChallengeField>,
) {
    let timer = Timer::new("gkr_verify_ref", true);
    let mut sp = VerifierScratchPad::<F>::new(circuit, proving_time_mpi_size);

    let layer_num = circuit.layers.len();

    let mut challenge = ExpanderSingleVarChallenge::sample_from_transcript(
        transcript,
        circuit.layers.last().unwrap().output_var_num,
        proving_time_mpi_size,
    )
    .into();

    let mut alpha = None;
    let mut claimed_v0 = *claimed_v;
    let mut claimed_v1 = None;

    transcript.lock_proof();
    transcript.append_field_element(claimed_v);
    transcript.unlock_proof();

    let mut verified = true;
    for i in (0..layer_num).rev() {
        let cur_verified = sumcheck_verify_gkr_layer_ref(
            proving_time_mpi_size,
            &circuit.layers[i],
            public_input,
            &mut challenge,
            &mut claimed_v0,
            &mut claimed_v1,
            alpha,
            &mut proof_reader,
            transcript,
            &mut sp,
            i == layer_num - 1,
            false,
            i,
            rnd_map,
        );

        verified &= cur_verified;
        alpha = if challenge.rz_1.is_some() {
            Some(transcript.generate_field_element::<F::ChallengeField>())
        } else {
            None
        };
    }
    timer.stop();
    let challenge = ExpanderDualVarChallenge::new(
        challenge.rz_0,
        challenge.rz_1,
        challenge.r_simd,
        challenge.r_mpi,
    );

    (verified, challenge, claimed_v0, claimed_v1)
}
