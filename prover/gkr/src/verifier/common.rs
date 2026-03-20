use std::{io::Read, vec};

use arith::Field;
use circuit::{CircuitLayer, RndCoefMap};
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, Transcript};
use sumcheck::{
    GKRVerifierHelper, VerifierScratchPad, SUMCHECK_GKR_DEGREE, SUMCHECK_GKR_SIMD_MPI_DEGREE,
    SUMCHECK_GKR_SQUARE_DEGREE,
};

#[inline(always)]
pub(crate) fn try_deserialize_field<F: Field>(mut proof_reader: impl Read) -> (F, bool) {
    match F::deserialize_from(&mut proof_reader) {
        Ok(v) => (v, true),
        Err(_) => (F::ZERO, false),
    }
}

#[inline(always)]
pub fn verify_sumcheck_step<F: FieldEngine>(
    mut proof_reader: impl Read,
    degree: usize,
    transcript: &mut impl Transcript,
    claimed_sum: &mut F::ChallengeField,
    randomness_vec: &mut Vec<F::ChallengeField>,
    sp: &VerifierScratchPad<F>,
) -> bool {
    let mut ps = vec![];
    let mut deser_ok = true;
    for i in 0..(degree + 1) {
        let (v, ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
        ps.push(v);
        deser_ok &= ok;
        transcript.append_field_element(&ps[i]);
    }

    let r = transcript.generate_field_element::<F::ChallengeField>();
    randomness_vec.push(r);

    let verified = deser_ok && (ps[0] + ps[1]) == *claimed_sum;

    if degree == SUMCHECK_GKR_DEGREE {
        *claimed_sum = GKRVerifierHelper::degree_2_eval(&ps, r, sp);
    } else if degree == SUMCHECK_GKR_SIMD_MPI_DEGREE {
        *claimed_sum = GKRVerifierHelper::degree_3_eval(&ps, r, sp);
    } else if degree == SUMCHECK_GKR_SQUARE_DEGREE {
        *claimed_sum = GKRVerifierHelper::degree_6_eval(&ps, r, sp);
    } else {
        return false;
    }

    verified
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::unnecessary_unwrap)]
pub fn sumcheck_verify_gkr_layer<F: FieldEngine>(
    proving_time_mpi_size: usize,
    layer: &CircuitLayer<F>,
    public_input: &[F::SimdCircuitField],
    challenge: &mut ExpanderDualVarChallenge<F>,
    claimed_v0: &mut F::ChallengeField,
    claimed_v1: &mut Option<F::ChallengeField>,
    alpha: Option<F::ChallengeField>,
    mut proof_reader: impl Read,
    transcript: &mut impl Transcript,
    sp: &mut VerifierScratchPad<F>,
    is_output_layer: bool,
    parallel_verify: bool,
) -> bool {
    assert_eq!(challenge.rz_1.is_none(), claimed_v1.is_none());
    assert_eq!(challenge.rz_1.is_none(), alpha.is_none());
    assert!(proving_time_mpi_size.is_power_of_two());

    if parallel_verify {
        GKRVerifierHelper::prepare_layer_non_sequential(layer, &alpha, challenge, sp);
    } else {
        GKRVerifierHelper::prepare_layer(layer, &alpha, challenge, sp, is_output_layer);
    }

    let var_num = layer.input_var_num;
    let simd_var_num = F::get_field_pack_size().trailing_zeros() as usize;
    let mut sum = *claimed_v0;
    if let Some(v1) = claimed_v1 {
        if let Some(a) = alpha {
            sum += *v1 * a;
        }
    }

    sum -= GKRVerifierHelper::eval_cst(&layer.const_, public_input, sp);

    let mut rx = vec![];
    let mut ry = None;
    let mut r_simd_xy = vec![];
    let mut r_mpi_xy = vec![];
    let mut verified = true;

    for _i_var in 0..var_num {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_DEGREE,
            transcript,
            &mut sum,
            &mut rx,
            sp,
        );
    }
    GKRVerifierHelper::set_rx(&rx, sp);

    for _i_var in 0..simd_var_num {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut sum,
            &mut r_simd_xy,
            sp,
        );
    }
    GKRVerifierHelper::set_r_simd_xy(&r_simd_xy, sp);

    for _i_var in 0..proving_time_mpi_size.trailing_zeros() {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut sum,
            &mut r_mpi_xy,
            sp,
        );
    }
    GKRVerifierHelper::set_r_mpi_xy(&r_mpi_xy, sp);

    let (vx_claim, vx_ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
    verified &= vx_ok;

    sum -= vx_claim * GKRVerifierHelper::eval_add(&layer.add, sp);
    transcript.append_field_element(&vx_claim);

    let vy_claim = if !layer.structure_info.skip_sumcheck_phase_two {
        ry = Some(vec![]);
        for _i_var in 0..var_num {
            verified &= verify_sumcheck_step::<F>(
                &mut proof_reader,
                SUMCHECK_GKR_DEGREE,
                transcript,
                &mut sum,
                ry.as_mut().unwrap(),
                sp,
            );
        }
        GKRVerifierHelper::set_ry(ry.as_ref().unwrap(), sp);

        let (vy_claim, vy_ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
        verified &= vy_ok;
        transcript.append_field_element(&vy_claim);
        verified &= sum == vx_claim * vy_claim * GKRVerifierHelper::eval_mul(&layer.mul, sp);
        Some(vy_claim)
    } else {
        verified &= sum == F::ChallengeField::ZERO;
        None
    };

    *challenge = ExpanderDualVarChallenge::new(rx, ry, r_simd_xy, r_mpi_xy);
    *claimed_v0 = vx_claim;
    *claimed_v1 = vy_claim;

    verified
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::unnecessary_unwrap)]
pub fn sumcheck_verify_gkr_layer_ref<F: FieldEngine>(
    proving_time_mpi_size: usize,
    layer: &CircuitLayer<F>,
    public_input: &[F::SimdCircuitField],
    challenge: &mut ExpanderDualVarChallenge<F>,
    claimed_v0: &mut F::ChallengeField,
    claimed_v1: &mut Option<F::ChallengeField>,
    alpha: Option<F::ChallengeField>,
    mut proof_reader: impl Read,
    transcript: &mut impl Transcript,
    sp: &mut VerifierScratchPad<F>,
    is_output_layer: bool,
    parallel_verify: bool,
    layer_idx: usize,
    rnd_map: &RndCoefMap<F::CircuitField>,
) -> bool {
    assert_eq!(challenge.rz_1.is_none(), claimed_v1.is_none());
    assert_eq!(challenge.rz_1.is_none(), alpha.is_none());
    assert!(proving_time_mpi_size.is_power_of_two());

    if parallel_verify {
        GKRVerifierHelper::prepare_layer_non_sequential(layer, &alpha, challenge, sp);
    } else {
        GKRVerifierHelper::prepare_layer(layer, &alpha, challenge, sp, is_output_layer);
    }

    let var_num = layer.input_var_num;
    let simd_var_num = F::get_field_pack_size().trailing_zeros() as usize;
    let mut sum = *claimed_v0;
    if let Some(v1) = claimed_v1 {
        if let Some(a) = alpha {
            sum += *v1 * a;
        }
    }

    sum -= GKRVerifierHelper::eval_cst_ref(&layer.const_, public_input, sp, layer_idx, rnd_map);

    let mut rx = vec![];
    let mut ry = None;
    let mut r_simd_xy = vec![];
    let mut r_mpi_xy = vec![];
    let mut verified = true;

    for _i_var in 0..var_num {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_DEGREE,
            transcript,
            &mut sum,
            &mut rx,
            sp,
        );
    }
    GKRVerifierHelper::set_rx(&rx, sp);

    for _i_var in 0..simd_var_num {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut sum,
            &mut r_simd_xy,
            sp,
        );
    }
    GKRVerifierHelper::set_r_simd_xy(&r_simd_xy, sp);

    for _i_var in 0..proving_time_mpi_size.trailing_zeros() {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut sum,
            &mut r_mpi_xy,
            sp,
        );
    }
    GKRVerifierHelper::set_r_mpi_xy(&r_mpi_xy, sp);

    let (vx_claim, vx_ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
    verified &= vx_ok;

    sum -= vx_claim * GKRVerifierHelper::eval_add_ref(&layer.add, sp, layer_idx, rnd_map);
    transcript.append_field_element(&vx_claim);

    let vy_claim = if !layer.structure_info.skip_sumcheck_phase_two {
        ry = Some(vec![]);
        for _i_var in 0..var_num {
            verified &= verify_sumcheck_step::<F>(
                &mut proof_reader,
                SUMCHECK_GKR_DEGREE,
                transcript,
                &mut sum,
                ry.as_mut().unwrap(),
                sp,
            );
        }
        GKRVerifierHelper::set_ry(ry.as_ref().unwrap(), sp);

        let (vy_claim, vy_ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
        verified &= vy_ok;
        transcript.append_field_element(&vy_claim);
        verified &= sum
            == vx_claim
                * vy_claim
                * GKRVerifierHelper::eval_mul_ref(&layer.mul, sp, layer_idx, rnd_map);
        Some(vy_claim)
    } else {
        verified &= sum == F::ChallengeField::ZERO;
        None
    };

    *challenge = ExpanderDualVarChallenge::new(rx, ry, r_simd_xy, r_mpi_xy);
    *claimed_v0 = vx_claim;
    *claimed_v1 = vy_claim;

    verified
}
