use super::verify_sumcheck_step;
use arith::Field;
use ark_std::{end_timer, start_timer};
use circuit::{Circuit, CircuitLayer};
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, FieldType, Transcript,
};
use serdes::ExpSerde;
use std::io::Read;
use sumcheck::{GKRVerifierHelper, VerifierScratchPad, SUMCHECK_GKR_SQUARE_DEGREE};

#[allow(clippy::type_complexity)]
pub fn gkr_square_verify<C: FieldEngine>(
    proving_time_mpi_size: usize,
    circuit: &Circuit<C>,
    public_input: &[C::SimdCircuitField],
    claimed_v: &C::ChallengeField,
    transcript: &mut impl Transcript,
    mut proof_reader: impl Read,
) -> (bool, ExpanderSingleVarChallenge<C>, C::ChallengeField) {
    assert_ne!(
        C::FIELD_TYPE,
        FieldType::GF2Ext128,
        "GF2 is not supported in GKR^2"
    );

    let timer = start_timer!(|| "gkr verify");
    let mut sp = VerifierScratchPad::<C>::new(circuit, proving_time_mpi_size);

    let layer_num = circuit.layers.len();

    let mut challenge = ExpanderSingleVarChallenge::sample_from_transcript(
        transcript,
        circuit.layers.last().unwrap().output_var_num,
        proving_time_mpi_size,
    );

    log::trace!("Initial rz0: {:?}", challenge.rz);
    log::trace!("Initial r_simd: {:?}", challenge.r_simd);
    log::trace!("Initial r_mpi: {:?}", challenge.r_mpi);

    let mut verified = true;
    let mut current_claim = *claimed_v;
    log::trace!("Starting claim: {current_claim:?}",);
    for i in (0..layer_num).rev() {
        let cur_verified = sumcheck_verify_gkr_square_layer(
            proving_time_mpi_size,
            &circuit.layers[i],
            public_input,
            &mut challenge,
            &mut current_claim,
            &mut proof_reader,
            transcript,
            &mut sp,
            i == layer_num - 1,
            false,
        );
        log::trace!("Layer {i} verified? {cur_verified}");
        verified &= cur_verified;
    }
    end_timer!(timer);
    (verified, challenge, current_claim)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::unnecessary_unwrap)]
pub fn sumcheck_verify_gkr_square_layer<C: FieldEngine>(
    proving_time_mpi_size: usize,
    layer: &CircuitLayer<C>,
    public_input: &[C::SimdCircuitField],
    challenge: &mut ExpanderSingleVarChallenge<C>,
    current_claim: &mut C::ChallengeField,
    mut proof_reader: impl Read,
    transcript: &mut impl Transcript,
    sp: &mut VerifierScratchPad<C>,
    is_output_layer: bool,
    parallel_verify: bool,
) -> bool {
    // GKR2 with Power5 gate has degree 6 polynomial
    let degree = SUMCHECK_GKR_SQUARE_DEGREE;

    let dual_challenge = ExpanderDualVarChallenge::from(challenge.clone());

    if parallel_verify {
        GKRVerifierHelper::prepare_layer_non_sequential(layer, &None, &dual_challenge, sp);
    } else {
        GKRVerifierHelper::prepare_layer(layer, &None, &dual_challenge, sp, is_output_layer);
    }

    let var_num = layer.input_var_num;
    let mut sum = *current_claim;
    sum -= GKRVerifierHelper::eval_cst(&layer.const_, public_input, sp);

    *challenge = ExpanderSingleVarChallenge::default();
    let mut verified = true;

    for i_var in 0..var_num {
        verified &= verify_sumcheck_step::<C>(
            &mut proof_reader,
            degree,
            transcript,
            &mut sum,
            &mut challenge.rz,
            sp,
        );
        log::trace!("x {i_var} var, verified? {verified}");
    }
    GKRVerifierHelper::set_rx(&challenge.rz, sp);

    for i_var in 0..C::get_field_pack_size().trailing_zeros() {
        verified &= verify_sumcheck_step::<C>(
            &mut proof_reader,
            degree,
            transcript,
            &mut sum,
            &mut challenge.r_simd,
            sp,
        );
        log::trace!("simd {i_var} var, verified? {verified}");
    }
    GKRVerifierHelper::set_r_simd_xy(&challenge.r_simd, sp);

    for _i_var in 0..proving_time_mpi_size.trailing_zeros() {
        verified &= verify_sumcheck_step::<C>(
            &mut proof_reader,
            degree,
            transcript,
            &mut sum,
            &mut challenge.r_mpi,
            sp,
        );
        log::trace!("{_i_var} mpi var, verified? {verified}");
    }
    GKRVerifierHelper::set_r_mpi_xy(&challenge.r_mpi, sp);

    let v_claim = C::ChallengeField::deserialize_from(&mut proof_reader).unwrap();
    log::trace!("v_claim: {v_claim:?}");

    sum -= v_claim * GKRVerifierHelper::eval_pow_1(&layer.uni, sp)
        + v_claim.exp(5) * GKRVerifierHelper::eval_pow_5(&layer.uni, sp);
    transcript.append_field_element(&v_claim);

    verified &= sum == C::ChallengeField::ZERO;

    *current_claim = v_claim;
    verified
}
