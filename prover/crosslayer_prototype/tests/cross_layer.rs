use arith::Field;
use crosslayer_prototype::{prove_gkr, CrossLayerCircuit, CrossLayerConnections};
use gkr_engine::{BN254Config, FieldEngine, GF2ExtConfig, M31x16Config, Transcript};
use gkr_hashers::SHA256hasher;
use transcript::BytesHashTranscript;

fn test_sumcheck_cross_layered_helper<F: FieldEngine>() {
    let mut transcript = BytesHashTranscript::<SHA256hasher>::new();

    let mut rng = rand::thread_rng();
    let n_layers = 5;
    let circuit = CrossLayerCircuit::<F>::random_for_testing(&mut rng, n_layers);

    let inputs = (0..circuit.layers[0].layer_size)
        .map(|_| F::SimdCircuitField::random_unsafe(&mut rng))
        .collect::<Vec<_>>();
    let evals = circuit.evaluate(&inputs);
    let connections = CrossLayerConnections::parse_circuit(&circuit);
    let mut sp = crosslayer_prototype::CrossLayerProverScratchPad::<F>::new(
        circuit.layers.len(),
        circuit.max_num_input_var(),
        circuit.max_num_output_var(),
        1,
    );

    let (_output_claim, _input_challenge, _input_claim) =
        prove_gkr(&circuit, &evals, &connections, &mut transcript, &mut sp);
}

#[test]
fn test_sumcheck_cross_layered() {
    test_sumcheck_cross_layered_helper::<M31x16Config>();
    test_sumcheck_cross_layered_helper::<GF2ExtConfig>();
    test_sumcheck_cross_layered_helper::<BN254Config>();
}
