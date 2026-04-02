// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, fmt::Debug, mem::MaybeUninit};

use binius_field::{BinaryField128bGhash as Ghash, Field};
use binius_utils::{
    DeserializeBytes, SerializeBytes,
    rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::{ParallelSlice, ParallelSliceMut},
    },
};
use digest::Output;

use super::{
    compression::VisionCompression, constants::M, digest::VisionHasherDigest,
    parallel_permutation::batch_permutation,
};
use crate::parallel_compression::ParallelPseudoCompression;

// The number of parallel compressions N must be a power of 2.
// The amortization of batch inversion grows with the batch size
// and thus with N. Heuristically 128 is the largest N before
// performance degrades.
const N: usize = 128;
const MN: usize = N * M;

/// Parallel Vision compression with N parallel compressions using rayon.
///
/// Processes N compression pairs simultaneously using parallel Vision permutation
/// and multithreading for optimal performance.
#[derive(Clone, Debug, Default)]
pub struct VisionParallelCompression {
    compression: VisionCompression,
}

impl VisionParallelCompression {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ParallelPseudoCompression<Output<VisionHasherDigest>, 2> for VisionParallelCompression {
    type Compression = VisionCompression;

    fn compression(&self) -> &Self::Compression {
        &self.compression
    }

    // If we add another implementation of `ParallelPseudoCompression`, it makes sense to add the
    // compression-equivalent of `MultiDigest` and `ParallelMultidigestImpl` to avoid
    // duplicating the logic below of breaking into chunks of N and handling remainders.
    #[tracing::instrument(
        "VisionParallelCompression::parallel_compress",
        skip_all,
        level = "debug"
    )]
    fn parallel_compress(
        &self,
        inputs: &[Output<VisionHasherDigest>],
        out: &mut [MaybeUninit<Output<VisionHasherDigest>>],
    ) {
        assert_eq!(
            inputs.len(),
            2 * out.len(),
            "Input length must be 2 * output length"
        );

        inputs
            .par_chunks_exact(N * 2)
            .zip(out.par_chunks_exact_mut(N))
            .for_each(|(input_chunk, output_chunk)| {
                self.compress_batch_parallel(input_chunk, output_chunk);
            });

        // Handle remaining pairs using batched processing
        let remainder_inputs = inputs.chunks_exact(N * 2).remainder();
        let remainder_outputs = out.chunks_exact_mut(N).into_remainder();

        if !remainder_outputs.is_empty() {
            // Use stack-allocated arrays for remainder handling
            let mut padded_inputs = [Output::<VisionHasherDigest>::default(); N * 2];
            let mut padded_outputs = [MaybeUninit::uninit(); N];

            // Copy remainder inputs
            padded_inputs[..remainder_inputs.len()].copy_from_slice(remainder_inputs);

            // Process full batch (including padding)
            self.compress_batch_parallel(&padded_inputs, &mut padded_outputs);

            // Copy only the actual results back
            for (output, padded) in remainder_outputs.iter_mut().zip(padded_outputs) {
                // Safety: `compress_batch_parallel` guarantees to initialize `padded_outputs`
                output.write(unsafe { padded.assume_init() });
            }
        }
    }
}

impl VisionParallelCompression {
    /// Compress exactly N pairs using parallel permutation.
    #[tracing::instrument(
        "VisionParallelCompression::compress_batch_parallel",
        skip_all,
        level = "debug"
    )]
    #[inline]
    fn compress_batch_parallel(
        &self,
        inputs: &[Output<VisionHasherDigest>],
        out: &mut [MaybeUninit<Output<VisionHasherDigest>>],
    ) {
        assert_eq!(out.len(), N, "Must process exactly {N} pairs");
        assert_eq!(inputs.len(), 2 * N, "Must have 2*N inputs");

        // Step 1: Deserialize inputs into flattened state array
        let mut states = [Ghash::ZERO; MN];
        for i in 0..N {
            let input0 = &inputs[i * 2];
            let input1 = &inputs[i * 2 + 1];

            // Deserialize each 32-byte input into 2 Ghash elements
            states[i * M] = Ghash::deserialize(&input0[0..16]).expect("16 bytes fits in Ghash");
            states[i * M + 1] =
                Ghash::deserialize(&input0[16..32]).expect("16 bytes fits in Ghash");
            states[i * M + 2] = Ghash::deserialize(&input1[0..16]).expect("16 bytes fits in Ghash");
            states[i * M + 3] =
                Ghash::deserialize(&input1[16..32]).expect("16 bytes fits in Ghash");
        }

        // Step 2: Copy original first 2 elements for each state
        let originals: [_; N] = array::from_fn(|i| (states[i * M], states[i * M + 1]));

        // Step 3: Apply parallel permutation to all states
        batch_permutation::<N, MN>(&mut states);

        // Step 4: Add original elements back and serialize outputs
        for i in 0..N {
            states[i * M] += originals[i].0;
            states[i * M + 1] += originals[i].1;

            let mut output = Output::<VisionHasherDigest>::default();
            let (left, right) = output.as_mut_slice().split_at_mut(16);
            states[i * M].serialize(left).expect("fits in 16 bytes");
            states[i * M + 1]
                .serialize(right)
                .expect("fits in 16 bytes");
            out[i].write(output);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use digest::Digest;

    use super::*;
    use crate::PseudoCompressionFunction;

    #[test]
    fn test_parallel_vs_sequential_simple() {
        let parallel = VisionParallelCompression::default();
        let sequential = &parallel.compression;

        // Create test inputs (4 inputs = 2 pairs)
        let inputs = [
            VisionHasherDigest::new().finalize(), // input 0 (pair 0, element 0)
            {
                let mut hasher = VisionHasherDigest::new();
                hasher.update(b"first");
                hasher.finalize()
            }, // input 1 (pair 0, element 1)
            {
                let mut hasher = VisionHasherDigest::new();
                hasher.update(b"second");
                hasher.finalize()
            }, // input 2 (pair 1, element 0)
            {
                let mut hasher = VisionHasherDigest::new();
                hasher.update(b"third");
                hasher.finalize()
            }, // input 3 (pair 1, element 1)
        ];

        // Compute expected results sequentially
        let sequential_results = [
            sequential.compress([inputs[0], inputs[1]]),
            sequential.compress([inputs[2], inputs[3]]),
        ];

        // Compute parallel results
        let mut parallel_outputs = [MaybeUninit::uninit(); 2];
        parallel.parallel_compress(&inputs, &mut parallel_outputs);
        let parallel_results: [_; 2] =
            array::from_fn(|i| unsafe { parallel_outputs[i].assume_init() });

        // Compare
        assert_eq!(sequential_results, parallel_results);
    }

    #[test]
    fn test_parallel_compress_large_batch() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let parallel = VisionParallelCompression::default();
        let mut rng = StdRng::seed_from_u64(0);

        // Test 300 pairs (600 inputs) to exercise batch processing (N=128) + remainder handling
        const NUM_PAIRS: usize = 300;

        // Generate test inputs
        let inputs: Vec<_> = (0..NUM_PAIRS * 2)
            .map(|i| {
                let mut hasher = VisionHasherDigest::new();
                hasher.update(i.to_le_bytes());
                hasher.update(rng.random::<[u8; 32]>());
                hasher.finalize()
            })
            .collect();

        // Compute expected results sequentially
        let sequential_results: Vec<_> = (0..NUM_PAIRS)
            .map(|i| {
                parallel
                    .compression
                    .compress([inputs[i * 2], inputs[i * 2 + 1]])
            })
            .collect();

        // Compute parallel results
        let mut parallel_outputs = vec![MaybeUninit::uninit(); NUM_PAIRS];
        parallel.parallel_compress(&inputs, &mut parallel_outputs);
        let parallel_results: Vec<_> = parallel_outputs
            .into_iter()
            .map(|out| unsafe { out.assume_init() })
            .collect();

        assert_eq!(sequential_results, parallel_results);
    }
}
