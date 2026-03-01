#![allow(clippy::pedantic)]
// Most of the code is from https://github.com/Plonky3/Plonky3
//
// Ideally we want to simply wrap plonky3/baby-bear::AVXBabyBear
// But plonky3 has a feature flag on avx256/avx512 where expander
// decides whether to use avx256 or avx512 at compile time.
// This is not compatible with our compile time flag for avx256/avx512
// in the core crate.
// So we re-implement the BabyBearx16 field in our own crate.
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

mod babybear;
pub use babybear::{BabyBear, BabyBearParameters};

mod babybearx16;
pub use babybearx16::BabyBearx16;

mod babybear_ext;
pub use babybear_ext::BabyBearExt3;

mod babybear_ext3x16;
pub use babybear_ext3x16::BabyBearExt3x16;

#[cfg(test)]
mod tests;
