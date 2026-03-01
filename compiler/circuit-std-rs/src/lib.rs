#![allow(clippy::pedantic)]
#![allow(clippy::manual_div_ceil)]
// allow range loop for better readability
#![allow(clippy::needless_range_loop)]

pub mod traits;
pub use traits::StdCircuit;

pub mod logup;
pub use logup::{LogUpCircuit, LogUpParams};

pub mod non_native;
pub use non_native::*;
pub mod matmul;

pub mod gnark;
pub mod poseidon;
pub mod sha256;
pub mod utils;

#[cfg(test)]
mod tests;
