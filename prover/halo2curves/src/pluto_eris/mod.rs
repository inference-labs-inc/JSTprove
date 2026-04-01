//! # `Pluto\Eris half-pairing cycle`
//!
//! Implementation of the Pluto / Eris half-pairing cycle of prime order elliptic curves.
//!
//! Supporting evidence: https://github.com/daira/pluto-eris
//! Field constant derivation: https://github.com/davidnevadoc/ec-constants/tree/main/pluto_eris
//! Pairing constants derivation: https://github.com/John-Gong-Math/pluto_eris/blob/main/pluto_pairing.ipynb
mod curve;
mod engine;
mod fields;

pub use curve::*;
pub use engine::*;
pub use fields::fp::*;
pub use fields::fq::*;
