// Copyright 2025 Irreducible Inc.
pub mod circuit;
pub mod constants;
pub mod reference;

#[cfg(test)]
mod tests;

pub use circuit::Blake2bCircuit;
pub use constants::*;
pub use reference::blake2b;

/// Shared test utilities for BLAKE2b testing
#[cfg(test)]
pub mod test_constants {
    use blake2::{Blake2b512, Digest};

    /// Generate ground truth hash using the standard blake2 crate
    pub fn ground_truth_blake2b(input: &[u8]) -> [u8; 64] {
        let mut hasher = Blake2b512::new();
        hasher.update(input);
        hasher.finalize().into()
    }

    /// Convert bytes to hex string
    pub fn hex_string(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    }
}
