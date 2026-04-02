// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, fmt::Debug, mem::MaybeUninit};

use binius_utils::rayon::prelude::*;

use crate::PseudoCompressionFunction;

/// A trait for parallel application of N-to-1 compression functions.
///
/// This trait enables efficient batch compression operations where multiple N-element
/// chunks are compressed in parallel. It's particularly useful for constructing hash trees
/// and Merkle trees where many compression operations need to be performed simultaneously.
///
/// The trait is parameterized by:
/// - `T`: The type of values being compressed (typically hash digests)
/// - `N`: The arity of the compression function (number of inputs per compression)
pub trait ParallelPseudoCompression<T, const N: usize> {
    /// The underlying compression function that performs N-to-1 compression.
    type Compression: PseudoCompressionFunction<T, N>;

    /// Returns a reference to the underlying compression function.
    fn compression(&self) -> &Self::Compression;

    /// Compresses multiple N-element chunks in parallel.
    ///
    /// # Arguments
    /// * `inputs` - A slice containing the values to compress. Must have length `N * out.len()`.
    /// * `out` - Output buffer where compressed values will be written.
    ///
    /// # Behavior
    /// For each index `i` in `0..out.len()`, this method computes:
    /// ```text
    /// out[i] = Compression::compress([inputs[i*N], inputs[i*N+1], ..., inputs[i*N+N-1]])
    /// ```
    ///
    /// All compressions are performed in parallel for efficiency.
    ///
    /// # Post-conditions
    /// After this method returns, all elements in `out` will be initialized with the
    /// compressed values from their corresponding N-element chunks in `inputs`.
    ///
    /// # Panics
    /// Panics if `inputs.len() != N * out.len()`.
    fn parallel_compress(&self, inputs: &[T], out: &mut [MaybeUninit<T>]);
}

/// A simple adapter that wraps any `PseudoCompressionFunction` to implement `ParallelCompression`.
///
/// This adapter provides a straightforward way to use existing compression functions
/// in parallel contexts by applying them sequentially to each N-element chunk.
#[derive(Debug, Clone)]
pub struct ParallelCompressionAdaptor<C> {
    compression: C,
}

impl<C> ParallelCompressionAdaptor<C> {
    /// Creates a new adapter wrapping the given compression function.
    pub fn new(compression: C) -> Self {
        Self { compression }
    }

    /// Returns a reference to the underlying compression function.
    pub fn compression(&self) -> &C {
        &self.compression
    }
}

impl<T, C, const ARITY: usize> ParallelPseudoCompression<T, ARITY> for ParallelCompressionAdaptor<C>
where
    T: Clone + Send + Sync,
    C: PseudoCompressionFunction<T, ARITY> + Sync,
{
    type Compression = C;

    fn compression(&self) -> &Self::Compression {
        &self.compression
    }

    fn parallel_compress(&self, inputs: &[T], out: &mut [MaybeUninit<T>]) {
        assert_eq!(
            inputs.len(),
            ARITY * out.len(),
            "Input length must be N * output length"
        );

        inputs
            .par_chunks_exact(ARITY)
            .zip(out.par_iter_mut())
            .for_each(|(chunk, output)| {
                // Convert slice to array for compression function
                let chunk_array: [T; ARITY] = array::from_fn(|j| chunk[j].clone());
                let compressed = self.compression.compress(chunk_array);
                output.write(compressed);
            });
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    // Simple test compression function that XORs all inputs
    #[derive(Clone, Debug)]
    struct XorCompression;

    impl PseudoCompressionFunction<u64, 3> for XorCompression {
        fn compress(&self, input: [u64; 3]) -> u64 {
            input[0] ^ input[1] ^ input[2]
        }
    }

    #[test]
    fn test_parallel_compression_adaptor() {
        let mut rng = StdRng::seed_from_u64(0);
        let compression = XorCompression;
        let adaptor = ParallelCompressionAdaptor::new(compression.clone());

        // Test with 4 chunks of 3 elements each
        const N: usize = 3;
        const NUM_CHUNKS: usize = 4;
        let inputs: Vec<u64> = (0..N * NUM_CHUNKS).map(|_| rng.random()).collect();

        // Use the adaptor
        let mut adaptor_output = [MaybeUninit::<u64>::uninit(); NUM_CHUNKS];
        adaptor.parallel_compress(&inputs, &mut adaptor_output);
        let adaptor_results: Vec<u64> = adaptor_output
            .into_iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();

        // Manually compress each chunk
        let mut manual_results = Vec::new();
        for chunk_idx in 0..NUM_CHUNKS {
            let start = chunk_idx * N;
            let chunk = [inputs[start], inputs[start + 1], inputs[start + 2]];
            manual_results.push(compression.compress(chunk));
        }

        // Results should be identical
        assert_eq!(adaptor_results, manual_results);
    }

    #[test]
    #[should_panic(expected = "Input length must be N * output length")]
    fn test_mismatched_input_length() {
        let compression = XorCompression;
        let adaptor = ParallelCompressionAdaptor::new(compression);

        let inputs = vec![1u64, 2, 3, 4]; // 4 elements
        let mut output = [MaybeUninit::<u64>::uninit(); 2]; // Expecting 6 elements (2 * 3)

        adaptor.parallel_compress(&inputs, &mut output);
    }
}
