//! Circuit-free GKR per-layer sumcheck verification for
//! holographic proofs.
//!
//! Mirrors [`super::common::sumcheck_verify_gkr_layer`] but
//! replaces the three circuit-dependent evaluation calls
//! (`eval_cst`, `eval_add`, `eval_mul`) with values supplied by
//! the caller (which in the holographic flow come from verified
//! sparse-MLE WHIR openings against the verifying key). Everything
//! else — the sumcheck round parsing, challenge derivation, eq
//! table setup, and constraint checking — is identical to the
//! standard verifier.

use std::io::Read;

use circuit::StructureInfo;
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, Transcript};
use sumcheck::{VerifierScratchPad, SUMCHECK_GKR_DEGREE, SUMCHECK_GKR_SIMD_MPI_DEGREE};

use super::common::verify_sumcheck_step;

/// Per-layer structural parameters extracted from the VK. These
/// are the only fields `sumcheck_verify_gkr_layer` reads from
/// `CircuitLayer` — everything else is driven by the proof
/// transcript.
pub struct LayerShape {
    pub input_var_num: usize,
    pub output_var_num: usize,
    pub structure_info: StructureInfo,
}

/// Wiring evaluations for one GKR layer, supplied by the
/// holographic opening rather than computed from the circuit.
pub struct WiringEvals<F: FieldEngine> {
    /// `eval_cst`: the constant-gate contribution at the random
    /// point. For layers without const_ gates this is `ZERO`.
    pub eval_cst: F::ChallengeField,
    /// `eval_add`: the add-wiring polynomial evaluated at `(r_z, r_x)`.
    pub eval_add: F::ChallengeField,
    /// `eval_mul`: the mul-wiring polynomial evaluated at `(r_z, r_x, r_y)`.
    /// Only used when `!structure_info.skip_sumcheck_phase_two`.
    pub eval_mul: F::ChallengeField,
    /// `eval_uni`: the uni-12345 (x^5 S-box) wiring polynomial
    /// evaluated at `(r_z, r_x)`. The sumcheck constraint
    /// multiplies this by `vx_claim^4` to account for the x^5
    /// nonlinearity.
    pub eval_uni: F::ChallengeField,
}

/// Circuit-free variant of
/// [`super::common::sumcheck_verify_gkr_layer`].
///
/// The sumcheck-round parsing and constraint checks are identical
/// to the standard verifier. The three wiring evaluations that
/// would normally come from `GKRVerifierHelper::eval_cst/add/mul`
/// are instead read from `wiring` (which the holographic caller
/// has already verified against the VK commitment via
/// `sparse_verify_full`).
///
/// `prepare_fn` sets up the verifier scratch pad for this layer.
/// The standard verifier calls `GKRVerifierHelper::prepare_layer`
/// which needs only `output_var_num` from the circuit layer — the
/// holographic caller passes the VK-supplied dimension.
#[allow(clippy::too_many_arguments)]
pub fn sumcheck_verify_gkr_layer_holographic<F: FieldEngine>(
    proving_time_mpi_size: usize,
    shape: &LayerShape,
    wiring: &WiringEvals<F>,
    challenge: &mut ExpanderDualVarChallenge<F>,
    claimed_v0: &mut F::ChallengeField,
    claimed_v1: &mut Option<F::ChallengeField>,
    alpha: Option<F::ChallengeField>,
    mut proof_reader: impl Read,
    transcript: &mut impl Transcript,
    sp: &mut VerifierScratchPad<F>,
    is_output_layer: bool,
) -> bool {
    use super::common::try_deserialize_field;
    use arith::Field;
    use polynomials::EqPolynomial;

    if (challenge.rz_1.is_none() != claimed_v1.is_none())
        || (challenge.rz_1.is_none() != alpha.is_none())
        || !proving_time_mpi_size.is_power_of_two()
    {
        return false;
    }

    // --- prepare_layer (holographic version: no circuit ref) ---
    if is_output_layer {
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &challenge.rz_0,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_rz0,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );
    } else {
        let output_len = 1 << challenge.rz_0.len();
        sp.eq_evals_at_rz0[..output_len].copy_from_slice(&sp.eq_evals_at_rx[..output_len]);
        if alpha.is_some() && challenge.rz_1.is_some() {
            let a = alpha.unwrap();
            for i in 0..(1usize << shape.output_var_num) {
                sp.eq_evals_at_rz0[i] += a * sp.eq_evals_at_ry[i];
            }
        }
    }
    EqPolynomial::<F::ChallengeField>::eq_eval_at(
        &challenge.r_simd,
        &F::ChallengeField::ONE,
        &mut sp.eq_evals_at_r_simd,
        &mut sp.eq_evals_first_part,
        &mut sp.eq_evals_second_part,
    );
    EqPolynomial::<F::ChallengeField>::eq_eval_at(
        &challenge.r_mpi,
        &F::ChallengeField::ONE,
        &mut sp.eq_evals_at_r_mpi,
        &mut sp.eq_evals_first_part,
        &mut sp.eq_evals_second_part,
    );
    sp.r_simd = challenge.r_simd.clone();
    sp.r_mpi = challenge.r_mpi.clone();

    // --- sumcheck driving (identical to standard verifier) ---
    let var_num = shape.input_var_num;
    let simd_var_num = F::get_field_pack_size().trailing_zeros() as usize;
    let mut sum = *claimed_v0;
    if let Some(v1) = claimed_v1 {
        if let Some(a) = alpha {
            sum += *v1 * a;
        }
    }

    // Replace eval_cst(layer.const_, public_input, sp) with supplied value
    sum -= wiring.eval_cst;

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
    sumcheck::GKRVerifierHelper::<F>::set_rx(&rx, sp);

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
    sumcheck::GKRVerifierHelper::<F>::set_r_simd_xy(&r_simd_xy, sp);

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
    sumcheck::GKRVerifierHelper::<F>::set_r_mpi_xy(&r_mpi_xy, sp);

    let (vx_claim, vx_ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
    verified &= vx_ok;

    sum -= vx_claim * wiring.eval_add + vx_claim.exp(5) * wiring.eval_uni;
    transcript.append_field_element(&vx_claim);

    let vy_claim = if !shape.structure_info.skip_sumcheck_phase_two {
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
        sumcheck::GKRVerifierHelper::<F>::set_ry(ry.as_ref().unwrap(), sp);

        let (vy_claim, vy_ok) = try_deserialize_field::<F::ChallengeField>(&mut proof_reader);
        verified &= vy_ok;
        transcript.append_field_element(&vy_claim);
        // Replace eval_mul(layer.mul, sp) with supplied value
        verified &= sum == vx_claim * vy_claim * wiring.eval_mul;
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
