#![allow(clippy::pedantic)]
mod circuit_def;
use circuit_def::gen_computation_graph_and_witness;
use expander_compiler::{
    frontend::{BN254Config, CircuitField},
    zkcuda::{
        context::ComputationGraph,
        proving_system::{
            expander::config::ZKCudaBN254KZG, ExpanderNoOverSubscribe, ProvingSystem,
        },
    },
};
use serdes::ExpSerde;

#[allow(clippy::needless_range_loop)]
fn main() {
    // Replace this with your actual input data.
    let mut input = vec![vec![]; 16];
    for i in 0..16 {
        for j in 0..2 {
            input[i].push(CircuitField::<BN254Config>::from((i * 2 + j + 1) as u32));
        }
    }

    let (_, extended_witness) = gen_computation_graph_and_witness::<BN254Config>(Some(input));

    // Note: we've saved the computation graph and setup in the server. In order to generate a proof, we only need to submit the witness.
    let dummy_prover_setup = <ExpanderNoOverSubscribe<ZKCudaBN254KZG> as ProvingSystem<
        BN254Config,
    >>::ProverSetup::default();
    let dummy_computation_graph = ComputationGraph::<BN254Config>::default();

    let proof = ExpanderNoOverSubscribe::<ZKCudaBN254KZG>::prove(
        &dummy_prover_setup,
        &dummy_computation_graph,
        extended_witness.unwrap(),
    );

    let mut bytes = vec![];
    proof.serialize_into(&mut bytes).unwrap();
    std::fs::write("/tmp/proof.bin", &bytes).unwrap();
}
