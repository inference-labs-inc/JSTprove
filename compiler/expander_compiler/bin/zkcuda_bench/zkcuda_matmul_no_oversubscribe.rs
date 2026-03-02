#![allow(clippy::pedantic)]
#![allow(unused)]
mod zkcuda_matmul;
use expander_compiler::{
    frontend::BN254Config,
    zkcuda::proving_system::{
        expander::config::ZKCudaBN254Hyrax, expander_pcs_defered::BN254ConfigSha2UniKZG,
        ExpanderNoOverSubscribe,
    },
};
use zkcuda_matmul::zkcuda_matmul;

fn main() {
    zkcuda_matmul::<_, ExpanderNoOverSubscribe<ZKCudaBN254Hyrax>, 4>();
    zkcuda_matmul::<_, ExpanderNoOverSubscribe<ZKCudaBN254Hyrax>, 8>();
    zkcuda_matmul::<_, ExpanderNoOverSubscribe<ZKCudaBN254Hyrax>, 16>();

    zkcuda_matmul::<_, ExpanderNoOverSubscribe<ZKCudaBN254Hyrax>, 1024>();
}
