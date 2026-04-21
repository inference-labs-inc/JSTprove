#![allow(clippy::pedantic, clippy::all)]

pub mod prover;
pub use prover::*;

pub mod verifier;
pub use verifier::*;

pub mod holographic;

pub mod utils;

pub mod gkr_configs;
pub use gkr_configs::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "grinding")]
const GRINDING_BITS: usize = 10;
