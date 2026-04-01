#![allow(
    clippy::all,
    clippy::pedantic,
    dead_code,
    unused_attributes,
    unused_must_use
)]

mod arithmetic;
pub mod ff_ext;
pub mod fft;
pub mod hash_to_curve;
pub mod msm;
pub mod serde;

pub mod bn256;
pub mod grumpkin;
pub mod pasta;
pub mod pluto_eris;
pub mod secp256k1;
pub mod secp256r1;
pub mod secq256k1;

#[macro_use]
mod derive;

// Re-export to simplify down stream dependencies
pub use ff;
pub use group;
pub use pairing;
pub use pasta_curves::arithmetic::{Coordinates, CurveAffine, CurveExt};

#[cfg(test)]
pub mod tests;
