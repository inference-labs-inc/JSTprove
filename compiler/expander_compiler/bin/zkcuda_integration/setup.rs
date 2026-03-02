#![allow(clippy::pedantic, clippy::all)]
mod circuit_def;
use circuit_def::gen_computation_graph_and_witness;
use expander_compiler::{
    frontend::BN254Config,
    zkcuda::proving_system::{
        expander::config::ZKCudaBN254KZG, ExpanderNoOverSubscribe, ProvingSystem,
    },
};
use serdes::ExpSerde;

fn main() {
    let (computation_graph, _) = gen_computation_graph_and_witness::<BN254Config>(None);
    let (prover_setup, verifier_setup) =
        ExpanderNoOverSubscribe::<ZKCudaBN254KZG>::setup(&computation_graph);

    let mut bytes = vec![];
    prover_setup.serialize_into(&mut bytes).unwrap();
    std::fs::write("/tmp/prover_setup.bin", &bytes).unwrap();

    bytes.clear();
    verifier_setup.serialize_into(&mut bytes).unwrap();
    std::fs::write("/tmp/verifier_setup.bin", &bytes).unwrap();
}
