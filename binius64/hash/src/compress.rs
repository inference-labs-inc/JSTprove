// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

//! These interfaces are taken from
//! [p3_symmetric](https://github.com/Plonky3/Plonky3/blob/main/symmetric/src/compression.rs) in
//! [Plonky3].
//!
//! Plonky3 is dual-licensed under MIT OR Apache 2.0. We use it under Apache 2.0.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

/// An `N`-to-1 compression function, collision-resistant in a hash tree setting.
///
/// Unlike `CompressionFunction`, it may not be collision-resistant in general.
/// Instead it is only collision-resistant in hash-tree like settings where
/// the preimage of a non-leaf node must consist of compression outputs.
pub trait PseudoCompressionFunction<T, const N: usize>: Clone {
    fn compress(&self, input: [T; N]) -> T;
}

/// An `N`-to-1 compression function.
pub trait CompressionFunction<T, const N: usize>: PseudoCompressionFunction<T, N> {}

pub mod sha256 {
    use bytemuck::{bytes_of_mut, must_cast};
    use digest::{Digest, core_api::Block};
    use sha2::{Sha256, compress256, digest::Output};

    use super::*;

    /// A two-to-one compression function for SHA-256 digests.
    #[derive(Debug, Clone)]
    pub struct Sha256Compression {
        initial_state: [u32; 8],
    }

    impl Default for Sha256Compression {
        fn default() -> Self {
            let initial_state_bytes = Sha256::digest(b"BINIUS SHA-256 COMPRESS");
            let mut initial_state = [0u32; 8];
            bytes_of_mut(&mut initial_state).copy_from_slice(&initial_state_bytes);
            Self { initial_state }
        }
    }

    impl PseudoCompressionFunction<Output<Sha256>, 2> for Sha256Compression {
        fn compress(&self, input: [Output<Sha256>; 2]) -> Output<Sha256> {
            let mut ret = self.initial_state;
            let mut block = <Block<Sha256>>::default();
            block.as_mut_slice()[..32].copy_from_slice(input[0].as_slice());
            block.as_mut_slice()[32..].copy_from_slice(input[1].as_slice());
            compress256(&mut ret, &[block]);
            must_cast::<[u32; 8], [u8; 32]>(ret).into()
        }
    }

    impl CompressionFunction<Output<Sha256>, 2> for Sha256Compression {}
}
