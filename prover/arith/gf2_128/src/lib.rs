#![allow(clippy::pedantic)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

mod gf2_ext128;
pub use gf2_ext128::GF2_128;

mod gf2_ext128x8;
pub use gf2_ext128x8::GF2_128x8;

#[cfg(test)]
mod tests;
