#![allow(clippy::pedantic)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

/// Mersenne-31
mod m31;
pub use m31::M31;

/// SIMDx16 for Mersenne-31
mod m31x16;
pub use m31x16::M31x16;

/// Poseidon over Mersenne-31
mod poseidon;

/// Degree 3 extension field for Mersenne-31
mod m31_ext3;
pub use m31_ext3::M31Ext3;

/// Degree 6 extension field for Mersenne-31
/// instantiated by degree 2 extension of degree 3 extension for Mersenne-31
mod m31_ext6;
pub use m31_ext6::M31Ext6;

/// SIMDx16 for Degree 3 extension field for Mersenne-31
mod m31_ext3x16;
pub use m31_ext3x16::M31Ext3x16;

#[cfg(test)]
mod tests;
