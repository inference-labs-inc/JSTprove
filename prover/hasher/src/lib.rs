#![allow(clippy::pedantic)]
// traits definitions

mod traits;
pub use traits::{FiatShamirHasher, PoseidonStateTrait};

// field hashers

mod mimc;
pub use mimc::MiMC5FiatShamirHasher;

pub mod poseidon;
pub use poseidon::PoseidonFiatShamirHasher;

// byte hashers

pub mod sha2_256;
pub use sha2_256::SHA256hasher;

pub mod keccak_256;
pub use keccak_256::Keccak256hasher;

#[cfg(test)]
mod mimc_test;
