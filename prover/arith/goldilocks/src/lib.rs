#![allow(clippy::pedantic, clippy::all)]

/// Goldilocks field
mod goldilocks;
pub use goldilocks::{Goldilocks, EPSILON, GOLDILOCKS_MOD};

/// Goldilocks extension field (degree 2)
mod goldilocks_ext;
pub use goldilocks_ext::GoldilocksExt2;

/// Goldilocks extension field (degree 3, 192-bit)
mod goldilocks_ext3;
pub use goldilocks_ext3::GoldilocksExt3;

/// Goldilocks extension field (degree 4, 256-bit)
mod goldilocks_ext4;
pub use goldilocks_ext4::GoldilocksExt4;

/// GoldilocksExt2 as a standalone 128-bit scalar field (degree-1 self-extension)
mod goldilocks_ext2_scalar;
pub use goldilocks_ext2_scalar::GoldilocksExt2Scalar;

/// Goldilocks x8
mod goldilocksx8;
pub use goldilocksx8::Goldilocksx8;

/// Goldilocks extension field x8
mod goldilocks_ext2x8;
pub use goldilocks_ext2x8::GoldilocksExt2x8;

#[cfg(test)]
mod tests;
