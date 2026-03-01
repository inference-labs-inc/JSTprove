use arith::ExtensionField;
use gkr_engine::{FieldEngine, Transcript};

use crate::{
    CrossLayerCircuitEvals, CrossLayerConnections, CrossLayerGatherHelper,
    CrossLayerProverScratchPad, CrossLayerScatterHelper, GenericLayer,
};

#[inline]
pub fn transcript_io<F: ExtensionField, T: Transcript>(ps: &[F], transcript: &mut T) -> F {
    assert!(ps.len() == 3 || ps.len() == 4);
    for p in ps {
        transcript.append_field_element(p);
    }
    transcript.generate_field_element::<F>()
}

// FIXME
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn sumcheck_prove_scatter_layer<F: FieldEngine, T: Transcript>(
    layer: &GenericLayer<F>,
    rz0: &[F::ChallengeField],
    r_simd: &[F::ChallengeField],
    connections: &CrossLayerConnections,
    circuit_vals: &CrossLayerCircuitEvals<F>,
    transcript: &mut T,
    sp: &mut CrossLayerProverScratchPad<F>,
) -> (
    Vec<F::ChallengeField>,
    Vec<F::ChallengeField>,
    Vec<(usize, Vec<F::ChallengeField>)>,
) {
    let mut helper =
        CrossLayerScatterHelper::new(layer, rz0, r_simd, connections, circuit_vals, sp);

    helper.prepare_simd();

    // gkr phase 1 over variable x
    helper.prepare_x_vals();
    for i_var in 0..helper.input_layer_var_num {
        let evals = helper.poly_evals_at_rx(i_var, 2);
        let r = transcript_io::<F::ChallengeField, T>(&evals, transcript);
        helper.receive_rx(i_var, r);
    }

    helper.prepare_simd_var_vals();
    for i_var in 0..F::get_field_pack_size().trailing_zeros() as usize {
        let evals = helper.poly_evals_at_r_simd_var(i_var, 3);
        let r = transcript_io::<F::ChallengeField, T>(&evals, transcript);
        helper.receive_r_simd_var(i_var, r);
    }

    let vx_claims: Vec<(usize, F::ChallengeField)> = helper.vx_claims();
    for (_i_layer, claim) in vx_claims {
        transcript.append_field_element(&claim);
    }

    // gkr phase 2 over variable y
    helper.prepare_y_vals();
    for i_var in 0..helper.input_layer_var_num {
        let evals = helper.poly_evals_at_ry(i_var, 2);
        let r = transcript_io::<F::ChallengeField, T>(&evals, transcript);
        helper.receive_ry(i_var, r);
    }
    let vy_claim = helper.vy_claim();
    transcript.append_field_element(&vy_claim);

    (helper.rx, helper.ry, helper.r_relays_next)
}

#[allow(clippy::too_many_arguments)]
pub fn sumcheck_prove_gather_layer<F: FieldEngine, T: Transcript>(
    layer: &GenericLayer<F>,
    rz0: &[F::ChallengeField],
    rz1: &[F::ChallengeField],
    r_relays: &[(usize, Vec<F::ChallengeField>)],
    connections: &CrossLayerConnections,
    circuit_vals: &CrossLayerCircuitEvals<F>,
    transcript: &mut T,
    sp: &mut CrossLayerProverScratchPad<F>,
) -> (Vec<F::ChallengeField>, F::ChallengeField) {
    let alpha = transcript.generate_field_element::<F::ChallengeField>();
    let betas = transcript.generate_field_elements::<F::ChallengeField>(r_relays.len());

    let mut helper = CrossLayerGatherHelper::new(
        layer,
        rz0,
        rz1,
        r_relays,
        &alpha,
        &betas,
        connections,
        circuit_vals,
        sp,
    );

    helper.prepare_x_vals();
    for i_var in 0..helper.cur_layer_var_num {
        let evals = helper.poly_evals_at_rx(i_var, 2);
        let r = transcript_io::<F::ChallengeField, T>(&evals, transcript);
        helper.receive_rx(i_var, r);
    }

    let vx_claim = helper.vx_claim();
    transcript.append_field_element(&vx_claim);

    // No variable Y in gathering layer

    (helper.rx, vx_claim)
}
