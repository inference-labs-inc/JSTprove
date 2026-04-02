// Copyright 2025 Irreducible Inc.
//! Arithmetic for the GHASH field, GF(2)\[X\] / (X^128 + X^7 + X^2 + X + 1).

pub mod bit_sliced;
pub mod clmul;
pub mod soft64;

/// The multiplicative identity in GHASH
///
/// In GHASH, the standard representation of 1 is simply 0x01
pub const ONE: u128 = 0x01;

// Re-export mul_clmul for backward compatibility
#[allow(unused_imports)]
pub use clmul::{mul as mul_clmul, square as square_clmul};
