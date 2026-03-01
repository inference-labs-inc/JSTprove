use circuit::CircuitLayer;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, MPIConfig, MPIEngine,
    Transcript,
};

use crate::{
    prover_helper::{SumcheckGkrSquareHelper, SumcheckGkrVanillaHelper},
    utils::transcript_io,
    ProverScratchPad,
};

/// The degree of the polynomial for sumcheck, which is 2 for non-SIMD/MPI variables
/// and 3 for SIMD/MPI variables.
pub const SUMCHECK_GKR_DEGREE: usize = 2;
pub const SUMCHECK_GKR_SIMD_MPI_DEGREE: usize = 3;

/// The degree of the polynomial for sumcheck in the GKR square case.
/// It is 6 for both SIMD/MPI and non-SIMD/MPI variables.
pub const SUMCHECK_GKR_SQUARE_DEGREE: usize = 6;

// FIXME
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
// essentially the prev level of challenge passes here, once this level is done, new challenge gets
// written back into the prev space
pub fn sumcheck_prove_gkr_layer<F: FieldEngine, T: Transcript>(
    layer: &CircuitLayer<F>,
    challenge: &mut ExpanderDualVarChallenge<F>,
    alpha: Option<F::ChallengeField>,
    transcript: &mut T,
    sp: &mut ProverScratchPad<F>,
    mpi_config: &impl MPIEngine,
    is_output_layer: bool,
) -> (F::ChallengeField, Option<F::ChallengeField>) {
    let mut helper =
        SumcheckGkrVanillaHelper::new(layer, challenge, alpha, sp, mpi_config, is_output_layer);

    helper.prepare_simd();
    helper.prepare_mpi();

    // gkr phase 1 over variable x
    helper.prepare_x_vals();
    for i_var in 0..helper.input_var_num {
        let evals = helper.poly_evals_at_rx(i_var, SUMCHECK_GKR_DEGREE, mpi_config);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_rx(i_var, r);
        log::trace!("x i_var={i_var} evals: {evals:?} r: {r:?}");
    }

    helper.prepare_simd_var_vals();
    for i_var in 0..helper.simd_var_num {
        let evals =
            helper.poly_evals_at_r_simd_var(i_var, SUMCHECK_GKR_SIMD_MPI_DEGREE, mpi_config);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_r_simd_var(i_var, r);
        log::trace!("SIMD i_var={i_var} evals: {evals:?} r: {r:?}");
    }

    helper.prepare_mpi_var_vals(mpi_config);
    for i_var in 0..mpi_config.world_size().trailing_zeros() as usize {
        let evals = helper.poly_evals_at_r_mpi_var(i_var, SUMCHECK_GKR_SIMD_MPI_DEGREE);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_r_mpi_var(i_var, r);
    }

    let vx_claim = helper.vx_claim();
    transcript.append_field_element(&vx_claim);

    // gkr phase 2 over variable y
    let mut vy_claim = None;
    if !layer.structure_info.skip_sumcheck_phase_two {
        helper.prepare_y_vals(mpi_config);
        for i_var in 0..helper.input_var_num {
            let evals = helper.poly_evals_at_ry(i_var, SUMCHECK_GKR_DEGREE, mpi_config);
            let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
            helper.receive_ry(i_var, r);
        }
        vy_claim = Some(helper.vy_claim(mpi_config));
        transcript.append_field_element(&vy_claim.unwrap());
    }

    let rx = helper.rx;
    let ry = if !layer.structure_info.skip_sumcheck_phase_two {
        Some(helper.ry)
    } else {
        None
    };
    let r_simd = helper.r_simd_var;
    let r_mpi = helper.r_mpi_var;

    *challenge = ExpanderDualVarChallenge::new(rx, ry, r_simd, r_mpi);
    (vx_claim, vy_claim)
}

// FIXME
#[allow(clippy::needless_range_loop)] // todo: remove
#[allow(clippy::type_complexity)]
pub fn sumcheck_prove_gkr_square_layer<F: FieldEngine, T: Transcript>(
    layer: &CircuitLayer<F>,
    challenge: &mut ExpanderSingleVarChallenge<F>,
    transcript: &mut T,
    sp: &mut ProverScratchPad<F>,
    mpi_config: &MPIConfig,
) {
    const D: usize = SUMCHECK_GKR_SQUARE_DEGREE + 1;
    let mut helper = SumcheckGkrSquareHelper::<F, D>::new(layer, challenge, sp, mpi_config);

    helper.prepare_simd();
    helper.prepare_mpi();
    helper.prepare_g_x_vals();

    // x-variable sumcheck rounds
    for i_var in 0..layer.input_var_num {
        let evals = helper.poly_evals_at_x(i_var);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        log::trace!("x i_var={i_var} evals: {evals:?} r: {r:?}");
        helper.receive_x_challenge(i_var, r);
    }

    // Unpack SIMD witness polynomial evaluations
    helper.prepare_simd_var_vals();

    // SIMD-variable sumcheck rounds
    for i_var in 0..helper.simd_var_num {
        let evals = helper.poly_evals_at_simd(i_var);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        log::trace!("SIMD i_var={i_var} evals: {evals:?} r: {r:?}");
        helper.receive_simd_challenge(i_var, r);
    }

    helper.prepare_mpi_var_vals();
    for i_var in 0..mpi_config.world_size().trailing_zeros() as usize {
        let evals = helper.poly_evals_at_mpi(i_var);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_mpi_challenge(i_var, r);
    }

    log::trace!("vx claim: {:?}", helper.vx_claim());
    transcript.append_field_element(&helper.vx_claim());

    *challenge = ExpanderSingleVarChallenge::new(helper.rx, helper.r_simd_var, helper.r_mpi_var);
}
