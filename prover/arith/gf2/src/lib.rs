#![allow(clippy::pedantic)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

mod gf2;
pub use gf2::GF2;

mod gf2x8;
pub use gf2x8::GF2x8;

mod gf2x64;
pub use gf2x64::GF2x64;

mod gf2x128;
pub use gf2x128::GF2x128;

#[cfg(test)]
mod tests;
