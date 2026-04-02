// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Parallel Vision-6 hash permutation using flattened state arrays.
//!
//! Processes N Vision-6 states simultaneously by flattening them into a single N×6 array.
//! The key optimization is **batch inversion** - replacing N expensive field inversions
//! with a single inversion across all states using Montgomery's algorithm.
//!
//! # Layout
//! States: `[s0[0], s0[1], ..., s0[5], s1[0], s1[1], ...]` where `N` = number of states, `M = 6`.
//!
//! # Round Structure
//! Each round: inversion → transform → MDS → constants → inversion → transform → MDS → constants

use binius_field::{BinaryField128bGhash as Ghash, arithmetic_traits::Square};
use binius_math::batch_invert::BatchInversion;

use super::{
    constants::{B_FWD_COEFFS, M, NUM_ROUNDS, ROUND_CONSTANTS},
    permutation::{linearized_b_inv_transform_scalar, mds_mul},
};

/// Applies forward B-polynomial transformation: B(x) = c₀ + c₁x + c₂x² + c₃x⁴.
#[inline]
fn batch_forward_transform<const N: usize, const MN: usize>(states: &mut [Ghash; MN]) {
    for i in 0..MN {
        let scalar = states[i];
        let square = scalar.square();
        let quartic = square.square();

        states[i] = B_FWD_COEFFS[0]
            + B_FWD_COEFFS[1] * scalar
            + B_FWD_COEFFS[2] * square
            + B_FWD_COEFFS[3] * quartic;
    }
}

/// Applies inverse B-polynomial transformation using lookups.
#[inline]
fn batch_inverse_transform<const N: usize, const MN: usize>(states: &mut [Ghash; MN]) {
    for i in 0..MN {
        linearized_b_inv_transform_scalar(&mut states[i]);
    }
}

/// Applies MDS matrix multiplication to each of the N batch states.
#[inline]
fn batch_mds_mul<const N: usize, const MN: usize>(states: &mut [Ghash; MN]) {
    for i in 0..N {
        let state = &mut states[i * M..];
        mds_mul(state);
    }
}

/// Adds round constants to each of the N batch states.
#[inline]
fn batch_constants_add<const N: usize, const MN: usize>(
    states: &mut [Ghash; MN],
    constants: &[Ghash; M],
) {
    for i in 0..N {
        let state_start = i * M;
        for j in 0..M {
            states[state_start + j] += constants[j];
        }
    }
}

/// Executes a complete Vision-6 round on all batch states.
#[inline]
fn batch_round<const N: usize, const MN: usize>(
    states: &mut [Ghash; MN],
    inverter: &mut BatchInversion<Ghash>,
    round_constants_idx: usize,
) {
    // First half-round: inversion → inverse transform → MDS → constants
    inverter.invert_or_zero(states);
    batch_inverse_transform::<N, MN>(states);
    batch_mds_mul::<N, MN>(states);
    batch_constants_add::<N, MN>(states, &ROUND_CONSTANTS[round_constants_idx]);

    // Second half-round: inversion → forward transform → MDS → constants
    inverter.invert_or_zero(states);
    batch_forward_transform::<N, MN>(states);
    batch_mds_mul::<N, MN>(states);
    batch_constants_add::<N, MN>(states, &ROUND_CONSTANTS[round_constants_idx + 1]);
}

/// Executes the complete Vision-6 permutation on N batch states.
///
/// Main entry point for batch Vision-6 hashing.
#[inline]
pub fn batch_permutation<const N: usize, const MN: usize>(states: &mut [Ghash; MN]) {
    // Initial round constant addition
    batch_constants_add::<N, MN>(states, &ROUND_CONSTANTS[0]);

    let mut inverter = BatchInversion::<Ghash>::new(MN);

    // Execute all rounds of the permutation
    for round_num in 0..NUM_ROUNDS {
        batch_round::<N, MN>(states, &mut inverter, 1 + 2 * round_num);
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use binius_field::Random;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::vision_6::permutation::permutation;

    macro_rules! test_batch_permutation {
        ($name:ident, $n:expr) => {
            #[test]
            fn $name() {
                const N: usize = $n;
                const MN: usize = M * N;
                let mut rng = StdRng::seed_from_u64(0);

                for _ in 0..4 {
                    let mut batch_states: [Ghash; MN] = array::from_fn(|_| Ghash::random(&mut rng));

                    let mut single_states: [[Ghash; M]; N] =
                        array::from_fn(|i| array::from_fn(|j| batch_states[i * M + j]));

                    batch_permutation::<N, MN>(&mut batch_states);

                    for state in single_states.iter_mut() {
                        permutation(state);
                    }

                    let expected_batch: [Ghash; MN] =
                        array::from_fn(|i| single_states[i / M][i % M]);

                    assert_eq!(batch_states, expected_batch);
                }
            }
        };
    }

    test_batch_permutation!(test_batch_permutation_1, 1);
    test_batch_permutation!(test_batch_permutation_2, 2);
    test_batch_permutation!(test_batch_permutation_4, 4);
    test_batch_permutation!(test_batch_permutation_8, 8);
    test_batch_permutation!(test_batch_permutation_16, 16);
    test_batch_permutation!(test_batch_permutation_32, 32);
    test_batch_permutation!(test_batch_permutation_64, 64);
}
