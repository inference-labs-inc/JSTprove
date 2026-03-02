#![allow(clippy::pedantic, clippy::all)]
mod circuit_def;
use std::io::Cursor;

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

    let verifier_setup_bytes = std::fs::read("/tmp/verifier_setup.bin").unwrap();
    let verifier_setup = <ExpanderNoOverSubscribe<ZKCudaBN254KZG> as ProvingSystem<
        BN254Config,
    >>::VerifierSetup::deserialize_from(Cursor::new(verifier_setup_bytes))
    .unwrap();

    let proof_bytes = std::fs::read("/tmp/proof.bin").unwrap();
    let proof = <ExpanderNoOverSubscribe<ZKCudaBN254KZG> as ProvingSystem<BN254Config>>::Proof::deserialize_from(Cursor::new(proof_bytes)).unwrap();

    let verified = <ExpanderNoOverSubscribe<ZKCudaBN254KZG> as ProvingSystem<BN254Config>>::verify(
        &verifier_setup,
        &computation_graph,
        &proof,
    );
    assert!(verified, "Proof verification failed");
}
