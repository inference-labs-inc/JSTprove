// Copyright 2025 Irreducible Inc.
//! SIMD-accelerated binary field arithmetic operations.
//!
//! This crate is currently only used for benchmarking arithmetic operations.

mod arch;
pub mod ghash;
pub mod monbijou;
pub mod polyval;
pub mod rijndael;
#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;
pub mod underlier;

pub use underlier::{PackedUnderlier, Underlier};
