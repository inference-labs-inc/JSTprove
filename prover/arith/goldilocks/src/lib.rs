#![allow(clippy::pedantic)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

/// Goldilocks field
mod goldilocks;
pub use goldilocks::{Goldilocks, EPSILON, GOLDILOCKS_MOD};

/// Goldilocks extension field
mod goldilocks_ext;
pub use goldilocks_ext::GoldilocksExt2;

/// Goldilocks x8
mod goldilocksx8;
pub use goldilocksx8::Goldilocksx8;

/// Goldilocks extension field x8
mod goldilocks_ext2x8;
pub use goldilocks_ext2x8::GoldilocksExt2x8;

#[cfg(test)]
mod tests;
