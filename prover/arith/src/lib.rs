#![allow(clippy::pedantic)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]
#![feature(slice_swap_unchecked)]

mod field;
pub use field::*;

mod fft_field;
pub use fft_field::*;

mod extension_field;
pub use extension_field::*;

mod bn254;
pub use bn254::*;

mod bn254xn;
pub use bn254xn::*;

mod simd_field;
pub use simd_field::*;

mod macros;

mod utils;
pub use utils::*;

mod benches;
pub use benches::*;

mod tests;
pub use tests::*;

mod monty;
pub use monty::*;
