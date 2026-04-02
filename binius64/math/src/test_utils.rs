// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_field::{BinaryField128bGhash, Field, PackedBinaryGhash4x128b, PackedField};
use rand::RngCore;

use crate::FieldBuffer;

/// Type alias for 128b field element with fast arithmetic.
pub type B128 = BinaryField128bGhash;

/// Type alias for a packed 128b field element with non-trivial packing width.
pub type Packed128b = PackedBinaryGhash4x128b;

/// Generates a vector of random field elements.
///
/// # Arguments
///
/// * `rng` - Random number generator implementing RngCore
/// * `n` - Number of random field elements to generate
///
/// # Returns
///
/// Vector containing n random field elements
pub fn random_scalars<F: Field>(mut rng: impl RngCore, n: usize) -> Vec<F> {
    repeat_with(|| F::random(&mut rng)).take(n).collect()
}

/// Generates a [`FieldBuffer`] of random elements.
///
/// # Arguments
///
/// * `rng` - Random number generator implementing RngCore
/// * `log_n` - log2 the number of random field elements to generate
///
/// # Returns
///
/// Vector containing `2^log_n` random field elements
pub fn random_field_buffer<P: PackedField>(mut rng: impl RngCore, log_n: usize) -> FieldBuffer<P> {
    FieldBuffer::<P>::new(
        log_n,
        repeat_with(|| P::random(&mut rng))
            .take(1 << log_n.saturating_sub(P::LOG_WIDTH))
            .collect(),
    )
}

/// Converts an index to a hypercube point representation.
///
/// Given an index and number of variables, decomposes the index into a vector
/// of field elements where each element is either F::ZERO or F::ONE based on
/// the corresponding bit in the index.
///
/// # Arguments
///
/// * `n_vars` - Number of variables (bits) in the hypercube point
/// * `index` - The index to convert (must be less than 2^n_vars)
///
/// # Returns
///
/// Vector of n_vars field elements, where element i is F::ONE if bit i of index is 1,
/// and F::ZERO otherwise.
///
/// # Example
///
/// ```
/// # use binius_field::BinaryField128bGhash as B128;
/// # use binius_math::test_utils::index_to_hypercube_point;
/// let point = index_to_hypercube_point::<B128>(3, 5);
/// // 5 = 0b101, so point = [F::ONE, F::ZERO, F::ONE]
/// ```
pub fn index_to_hypercube_point<F: Field>(n_vars: usize, index: usize) -> Vec<F> {
    debug_assert!(
        index < (1 << n_vars),
        "Index {index} out of bounds for {n_vars}-variable hypercube"
    );
    (0..n_vars)
        .map(|i| {
            if (index >> i) & 1 == 1 {
                F::ONE
            } else {
                F::ZERO
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use binius_field::BinaryField128bGhash as B128;
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    proptest! {
        #[test]
        fn same_seed_produces_identical_results(
            seed: u64,
            n in 0..100usize
        ) {
            let mut rng1 = StdRng::seed_from_u64(seed);
            let mut rng2 = StdRng::seed_from_u64(seed);

            let scalars1 = random_scalars::<B128>(&mut rng1, n);
            let scalars2 = random_scalars::<B128>(&mut rng2, n);

            prop_assert_eq!(scalars1, scalars2);
        }

        #[test]
        fn different_seeds_produce_different_results(seed1: u64, seed2: u64) {
            prop_assume!(seed1 != seed2);

            // Test with 10 elements - collision probability is 1/2^320 ≈ 10^-96
            let n = 10;

            let mut rng1 = StdRng::seed_from_u64(seed1);
            let mut rng2 = StdRng::seed_from_u64(seed2);

            let scalars1 = random_scalars::<B128>(&mut rng1, n);
            let scalars2 = random_scalars::<B128>(&mut rng2, n);

            prop_assert_ne!(scalars1, scalars2);
        }
    }
}
