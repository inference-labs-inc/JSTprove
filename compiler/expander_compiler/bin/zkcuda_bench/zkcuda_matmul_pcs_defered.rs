#![allow(clippy::pedantic, clippy::all)]
#![allow(unused)]
mod zkcuda_matmul;
use expander_compiler::{
    frontend::BN254Config,
    zkcuda::proving_system::{expander_pcs_defered::BN254ConfigSha2UniKZG, ExpanderPCSDefered},
};
use gkr::BN254ConfigSha2Hyrax;
use zkcuda_matmul::zkcuda_matmul;

fn main() {
    // zkcuda_matmul::<BN254Config, ExpanderPCSDefered<BN254ConfigSha2Hyrax>, 4>();
    // zkcuda_matmul::<BN254Config, ExpanderPCSDefered<BN254ConfigSha2Hyrax>, 8>();
    // zkcuda_matmul::<BN254Config, ExpanderPCSDefered<BN254ConfigSha2Hyrax>, 16>();

    zkcuda_matmul::<BN254Config, ExpanderPCSDefered<BN254ConfigSha2UniKZG>, 4>();
    zkcuda_matmul::<BN254Config, ExpanderPCSDefered<BN254ConfigSha2UniKZG>, 8>();
    zkcuda_matmul::<BN254Config, ExpanderPCSDefered<BN254ConfigSha2UniKZG>, 16>();
}
