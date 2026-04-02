// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, mem::MaybeUninit};

use binius_field::{BinaryField128bGhash as Ghash, Field};
use binius_utils::{DeserializeBytes, SerializeBytes};
use digest::Output;

use super::{
    constants::M,
    digest::{PADDING_BLOCK, RATE_AS_U8, RATE_AS_U128, VisionHasherDigest, fill_padding},
    parallel_permutation::batch_permutation,
};
use crate::parallel_digest::MultiDigest;

/// A Vision hasher with state size M=6 suited for parallelization.
///
/// Without using packed fields, there is only one advantage of an explicit parallelized
/// Vision hasher over invoking the Vision hasher multiple times in parallel:
/// we can amortize the cost of inversion in the sbox using
/// [Montogery's trick](https://medium.com/eryxcoop/montgomerys-trick-for-batch-galois-field-inversion-9b6d0f399da2).
/// We slightly modify Montogery's trick to use a binary tree structure,
/// maximizing independence of multiplications for better instruction pipelining.
#[derive(Clone)]
pub struct VisionHasherMultiDigest<const N: usize, const MN: usize> {
    states: [Ghash; MN],
    buffers: [[u8; RATE_AS_U8]; N],
    filled_bytes: usize,
}

impl<const N: usize, const MN: usize> Default for VisionHasherMultiDigest<N, MN> {
    fn default() -> Self {
        assert!(
            N.is_power_of_two() && N >= 2,
            "N must be a power of 2 and >= 2"
        );
        assert_eq!(MN, M * N);
        Self {
            states: array::from_fn(|_| Ghash::ZERO),
            buffers: array::from_fn(|_| [0; RATE_AS_U8]),
            filled_bytes: 0,
        }
    }
}

impl<const N: usize, const MN: usize> VisionHasherMultiDigest<N, MN> {
    #[inline]
    fn advance_data(data: &mut [&[u8]; N], bytes: usize) {
        for i in 0..N {
            data[i] = &data[i][bytes..];
        }
    }

    fn permute(states: &mut [Ghash; MN], data: [&[u8]; N]) {
        for (i, data) in data.iter().enumerate() {
            debug_assert_eq!(data.len(), RATE_AS_U8);

            // Overwrite first RATE_AS_U128 elements of state i with data
            let state_start = i * M;
            for j in 0..RATE_AS_U128 {
                let element_bytes = &data[j * (128 / 8)..];
                states[state_start + j] =
                    Ghash::deserialize(element_bytes).expect("data len checked");
            }
        }

        batch_permutation::<N, MN>(states);
    }
    fn finalize(&mut self, out: &mut [MaybeUninit<digest::Output<VisionHasherDigest>>; N]) {
        if self.filled_bytes != 0 {
            for i in 0..N {
                fill_padding(&mut self.buffers[i][self.filled_bytes..]);
            }
            Self::permute(&mut self.states, array::from_fn(|i| &self.buffers[i][..]));
        } else {
            Self::permute(&mut self.states, array::from_fn(|_| &PADDING_BLOCK[..]));
        }

        // Serialize first two state elements for each digest (32 bytes total per digest)
        for i in 0..N {
            let output_slice = out[i].as_mut_ptr() as *mut u8;
            let output_bytes = unsafe { std::slice::from_raw_parts_mut(output_slice, 32) };
            let (state0, state1) = output_bytes.split_at_mut(16);
            self.states[i * M]
                .serialize(state0)
                .expect("fits in 16 bytes");
            self.states[i * M + 1]
                .serialize(state1)
                .expect("fits in 16 bytes");
        }
    }
}

impl<const N: usize, const MN: usize> MultiDigest<N> for VisionHasherMultiDigest<N, MN> {
    type Digest = VisionHasherDigest;

    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, mut data: [&[u8]; N]) {
        data[1..].iter().for_each(|row| {
            assert_eq!(row.len(), data[0].len());
        });

        if self.filled_bytes != 0 {
            let to_copy = std::cmp::min(data[0].len(), RATE_AS_U8 - self.filled_bytes);
            data.iter().enumerate().for_each(|(row_i, row)| {
                self.buffers[row_i][self.filled_bytes..self.filled_bytes + to_copy]
                    .copy_from_slice(&row[..to_copy]);
            });
            Self::advance_data(&mut data, to_copy);
            self.filled_bytes += to_copy;

            if self.filled_bytes == RATE_AS_U8 {
                Self::permute(&mut self.states, array::from_fn(|i| &self.buffers[i][..]));
                self.filled_bytes = 0;
            }
        }

        while data[0].len() >= RATE_AS_U8 {
            let chunks = array::from_fn(|i| &data[i][..RATE_AS_U8]);
            Self::permute(&mut self.states, chunks);
            Self::advance_data(&mut data, RATE_AS_U8);
        }

        if !data[0].is_empty() {
            data.iter().enumerate().for_each(|(row_i, row)| {
                self.buffers[row_i][..row.len()].copy_from_slice(row);
            });
            self.filled_bytes = data[0].len();
        }
    }

    fn finalize_into(mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
        self.finalize(out);
    }

    fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
        self.finalize(out);
        self.reset();
    }

    fn reset(&mut self) {
        self.states = array::from_fn(|_| Ghash::ZERO);
        self.buffers = array::from_fn(|_| [0; RATE_AS_U8]);
        self.filled_bytes = 0;
    }

    fn digest(data: [&[u8]; N], out: &mut [MaybeUninit<Output<Self::Digest>>; N]) {
        let mut digest = Self::default();
        digest.update(data);
        digest.finalize_into(out);
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use digest::Digest;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    // Helper function to generate random data vectors
    fn generate_random_data<const N: usize>(length: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data_vecs = Vec::new();
        for _ in 0..N {
            let mut vec = Vec::with_capacity(length);
            for _ in 0..length {
                vec.push(rng.random());
            }
            data_vecs.push(vec);
        }
        data_vecs
    }

    // Generic test function that compares parallel vs sequential execution
    fn test_parallel_vs_sequential<const N: usize, const MN: usize>(
        data: [&[u8]; N],
        description: &str,
    ) {
        // Parallel computation
        let mut parallel_outputs = [MaybeUninit::uninit(); N];
        VisionHasherMultiDigest::<N, MN>::digest(data, &mut parallel_outputs);
        let parallel_results: [Output<VisionHasherDigest>; N] =
            array::from_fn(|i| unsafe { parallel_outputs[i].assume_init() });

        // Sequential computation
        let sequential_results: [_; N] = array::from_fn(|i| {
            let mut hasher = VisionHasherDigest::new();
            hasher.update(data[i]);
            hasher.finalize()
        });

        // Compare results
        for i in 0..N {
            assert_eq!(
                parallel_results[i], sequential_results[i],
                "Mismatch at index {i} for {description}"
            );
        }
    }

    #[test]
    fn test_empty_inputs() {
        const N: usize = 4;
        let data: [&[u8]; N] = [&[], &[], &[], &[]];
        test_parallel_vs_sequential::<N, { N * M }>(data, "empty inputs");
    }

    #[test]
    fn test_small_inputs() {
        const N: usize = 2;
        let data: [&[u8]; N] = [b"Hello... World!", b"Rust is awesome"];
        test_parallel_vs_sequential::<N, { N * M }>(data, "small inputs");
    }

    #[test]
    fn test_multi_block() {
        const N: usize = 4;
        // Multiple blocks with different random patterns
        let target_len = RATE_AS_U8 * 2 + 10;
        let data_vecs = generate_random_data::<N>(target_len, 42);
        let data: [&[u8]; N] = array::from_fn(|i| data_vecs[i].as_slice());

        test_parallel_vs_sequential::<N, { N * M }>(data, "multi-block inputs");
    }

    #[test]
    fn test_various_sizes() {
        // Test different sizes separately since parallel requires same length per batch
        let sizes = [
            1,
            RATE_AS_U8 - 7,
            RATE_AS_U8,
            RATE_AS_U8 + 5,
            RATE_AS_U8 * 2 - 3,
        ];

        for &size in &sizes {
            const N: usize = 2;
            let data_vecs = generate_random_data::<N>(size, 123);
            let data: [&[u8]; N] = array::from_fn(|i| data_vecs[i].as_slice());
            test_parallel_vs_sequential::<N, { N * M }>(data, &format!("size {size}"));
        }
    }
}
