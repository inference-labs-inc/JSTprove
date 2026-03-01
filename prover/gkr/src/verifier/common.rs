use std::{io::Read, vec};

use arith::Field;
use circuit::CircuitLayer;
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, Transcript};
use serdes::ExpSerde;
use sumcheck::{
    GKRVerifierHelper, VerifierScratchPad, SUMCHECK_GKR_DEGREE, SUMCHECK_GKR_SIMD_MPI_DEGREE,
    SUMCHECK_GKR_SQUARE_DEGREE,
};

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
    for i in 0..(degree + 1) {
        ps.push(F::ChallengeField::deserialize_from(&mut proof_reader).unwrap());
        transcript.append_field_element(&ps[i]);
    }

    let r = transcript.generate_field_element::<F::ChallengeField>();
    randomness_vec.push(r);

    let verified = (ps[0] + ps[1]) == *claimed_sum;

    // This assumes SUMCHECK_GKR_DEGREE == 2, SUMCHECK_GKR_SIMD_MPI_DEGREE == 3,
    // SUMCHECK_GKR_SQUARE_DEGREE == 6
    if degree == SUMCHECK_GKR_DEGREE {
        *claimed_sum = GKRVerifierHelper::degree_2_eval(&ps, r, sp);
    } else if degree == SUMCHECK_GKR_SIMD_MPI_DEGREE {
        *claimed_sum = GKRVerifierHelper::degree_3_eval(&ps, r, sp);
    } else if degree == SUMCHECK_GKR_SQUARE_DEGREE {
        *claimed_sum = GKRVerifierHelper::degree_6_eval(&ps, r, sp);
    } else {
        panic!("unsupported degree");
    }

    verified
}

// todo: FIXME
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
        // println!("x {} var, verified? {}", _i_var, verified);
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
        // println!("{} simd var, verified? {}", _i_var, verified);
    }
    GKRVerifierHelper::set_r_simd_xy(&r_simd_xy, sp);

    for _i_var in 0..proving_time_mpi_size.ilog2() {
        verified &= verify_sumcheck_step::<F>(
            &mut proof_reader,
            SUMCHECK_GKR_SIMD_MPI_DEGREE,
            transcript,
            &mut sum,
            &mut r_mpi_xy,
            sp,
        );
        // println!("{} mpi var, verified? {}", _i_var, verified);
    }
    GKRVerifierHelper::set_r_mpi_xy(&r_mpi_xy, sp);

    let vx_claim = F::ChallengeField::deserialize_from(&mut proof_reader).unwrap();

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
            // println!("y {} var, verified? {}", _i_var, verified);
        }
        GKRVerifierHelper::set_ry(ry.as_ref().unwrap(), sp);

        let vy_claim = F::ChallengeField::deserialize_from(&mut proof_reader).unwrap();
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
