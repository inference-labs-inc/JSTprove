// an implementation of the GKR^2 protocol
//! This module implements the core GKR^2 IOP.

use ark_std::{end_timer, start_timer};
use circuit::Circuit;
use gkr_engine::{
    ExpanderSingleVarChallenge, FieldEngine, FieldType, MPIConfig, MPIEngine, Transcript,
};
use sumcheck::{sumcheck_prove_gkr_square_layer, ProverScratchPad};

#[allow(clippy::type_complexity)]
pub fn gkr_square_prove<F: FieldEngine>(
    circuit: &Circuit<F>,
    sp: &mut ProverScratchPad<F>,
    transcript: &mut impl Transcript,
    mpi_config: &MPIConfig,
) -> (F::ChallengeField, ExpanderSingleVarChallenge<F>) {
    assert_ne!(
        F::FIELD_TYPE,
        FieldType::GF2Ext128,
        "GF2 is not supported in GKR^2"
    );
    let timer = start_timer!(|| "gkr^2 prove");
    let layer_num = circuit.layers.len();

    let mut challenge = ExpanderSingleVarChallenge::sample_from_transcript(
        transcript,
        circuit.layers.last().unwrap().output_var_num,
        mpi_config.world_size(),
    );

    let output_vals = &circuit.layers.last().unwrap().output_vals;
    let claimed_v = F::collectively_eval_circuit_vals_at_expander_challenge(
        output_vals,
        &challenge,
        &mut sp.hg_evals,
        &mut sp.eq_evals_first_half, // confusing name here..
        mpi_config,
    );

    log::trace!("Claimed v: {claimed_v:?}");

    for i in (0..layer_num).rev() {
        sumcheck_prove_gkr_square_layer(
            &circuit.layers[i],
            &mut challenge,
            transcript,
            sp,
            mpi_config,
        );

        log::trace!("Layer {i} proved");
        log::trace!("rz0.0: {:?}", challenge.rz[0]);
        log::trace!("rz0.1: {:?}", challenge.rz[1]);
        log::trace!("rz0.2: {:?}", challenge.rz[2]);
    }

    end_timer!(timer);
    (claimed_v, challenge)
}
