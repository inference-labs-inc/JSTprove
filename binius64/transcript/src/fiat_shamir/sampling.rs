// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 authors

//! Traits used to sample random values in a public-coin interactive protocol.
//!
//! These interfaces are taken from [p3_challenger](https://github.com/Plonky3/Plonky3/blob/main/challenger/src/lib.rs) in [Plonky3].
//!
//! Plonky3 is dual-licensed under MIT OR Apache 2.0. We use it under Apache 2.0.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

use std::array;

use bytes::Buf;

#[auto_impl::auto_impl(&mut)]
pub trait CanSample<T> {
    fn sample(&mut self) -> T;

    fn sample_array<const N: usize>(&mut self) -> [T; N] {
        array::from_fn(|_| self.sample())
    }

    fn sample_vec(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.sample()).collect()
    }
}

#[auto_impl::auto_impl(&mut)]
pub trait CanSampleBits<T> {
    fn sample_bits(&mut self, bits: usize) -> T;
}

pub fn sample_bits_reader<Reader: Buf>(mut reader: Reader, bits: usize) -> u32 {
    let bits = bits.min(u32::BITS as usize);

    let bytes_to_sample = size_of::<u32>();

    let mut bytes = [0u8; size_of::<u32>()];

    reader.copy_to_slice(&mut bytes[..bytes_to_sample]);

    let unmasked = u32::from_le_bytes(bytes);
    let mask = 1u32.checked_shl(bits as u32);
    let mask = match mask {
        Some(x) => x - 1,
        None => u32::MAX,
    };
    mask & unmasked
}
