#![allow(clippy::pedantic, clippy::all)]
use expander_compiler::{
    frontend::BN254Config,
    zkcuda::proving_system::{
        expander::config::ZKCudaBN254KZG, ExpanderNoOverSubscribe, ProvingSystem,
    },
};

fn main() {
    <ExpanderNoOverSubscribe<ZKCudaBN254KZG> as ProvingSystem<BN254Config>>::post_process();
}
