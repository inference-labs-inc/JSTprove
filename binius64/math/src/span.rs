// Copyright 2024-2025 Irreducible Inc.

//! Utilities for computing subset sums and vector span operations.

use std::iter;

use binius_field::Field;

/// Expands an array of field elements into all possible subset sums.
///
/// For an input array `[a, b, c]`, this computes all possible sums of subsets:
/// `[0, a, b, a+b, c, a+c, b+c, a+b+c]`
///
/// This is used to create lookup tables for the Method of Four Russians optimization,
/// where we precompute all possible combinations of a small set of values to avoid
/// doing the additions at runtime.
///
/// ## Type Parameters
///
/// * `F` - The field element type
/// * `N` - Size of the input array
/// * `N_EXP2` - Size of the output array, must be 2^N
///
/// ## Arguments
///
/// * `elems` - Input array of N field elements
///
/// ## Returns
///
/// An array of size N_EXP2 containing all possible subset sums of the input elements
///
/// ## Preconditions
///
/// * N_EXP2 must equal 2^N
///
/// ## Example
///
/// ```ignore
/// let input = [F::ONE, F::from(2)];
/// let sums = expand_subset_sums_array(input);
/// // sums = [F::ZERO, F::ONE, F::from(2), F::from(3)]
/// ```
pub fn expand_subset_sums_array<F: Field, const N: usize, const N_EXP2: usize>(
    elems: [F; N],
) -> [F; N_EXP2] {
    assert_eq!(N_EXP2, 1 << N);

    let mut expanded = [F::ZERO; N_EXP2];
    for (i, elem_i) in elems.into_iter().enumerate() {
        let span = &mut expanded[..1 << (i + 1)];
        let (lo_half, hi_half) = span.split_at_mut(1 << i);
        for (lo_half_i, hi_half_i) in iter::zip(lo_half, hi_half) {
            *hi_half_i = *lo_half_i + elem_i;
        }
    }
    expanded
}

/// Expands a slice of field elements into all possible subset sums.
///
/// For an input slice `[a, b, c]`, this computes all possible sums of subsets:
/// `[0, a, b, a+b, c, a+c, b+c, a+b+c]`
///
/// This is a dynamic version of [`expand_subset_sums_array`] that works with slices
/// and returns a Vec with length 2^n where n is the input length.
///
/// ## Arguments
///
/// * `elems` - Input slice of field elements
///
/// ## Returns
///
/// A Vec containing all possible subset sums of the input elements, with length 2^n
/// where n is the length of the input slice.
///
/// ## Example
///
/// ```ignore
/// let input = vec![F::ONE, F::from(2)];
/// let sums = expand_subset_sums(&input);
/// // sums = vec![F::ZERO, F::ONE, F::from(2), F::from(3)]
/// ```
pub fn expand_subset_sums<F: Field>(elems: &[F]) -> Vec<F> {
    let n = elems.len();
    let n_exp2 = 1 << n;

    let mut expanded = vec![F::ZERO; n_exp2];
    for (i, &elem_i) in elems.iter().enumerate() {
        let span = &mut expanded[..1 << (i + 1)];
        let (lo_half, hi_half) = span.split_at_mut(1 << i);
        for (lo_half_i, hi_half_i) in iter::zip(lo_half, hi_half) {
            *hi_half_i = *lo_half_i + elem_i;
        }
    }
    expanded
}

#[cfg(test)]
mod tests {
    use binius_field::{BinaryField128bGhash, Field, Random};
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    type F = BinaryField128bGhash;

    proptest! {
        #[test]
        fn test_expand_subset_sums_correctness(
            n in 0usize..=8,  // Input length (small to avoid exponential blowup)
            index in 0usize..256,  // Index to check
        ) {
            // Truncate index to the correct range
            let index = index % (1 << n);

            let mut rng = StdRng::seed_from_u64(n as u64);

            // Generate random input elements
            let elems: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();

            // Compute the expansion
            let result = expand_subset_sums(&elems);

            // Verify the result length
            prop_assert_eq!(result.len(), 1 << n);

            // Compute expected sum based on binary representation of index
            let mut expected = F::ZERO;
            for (bit_pos, &elem) in elems.iter().enumerate() {
                if (index >> bit_pos) & 1 == 1 {
                    expected += elem;
                }
            }

            prop_assert_eq!(
                result[index],
                expected,
                "Index {} should have subset sum corresponding to its binary representation",
                index
            );
        }
    }

    #[test]
    fn test_expand_subset_sums_array_slice_consistency() {
        let mut rng = StdRng::seed_from_u64(0);

        // Helper function to test consistency for a specific size
        fn check_consistency<const N: usize, const N_EXP2: usize>(elems_vec: &[F]) {
            assert_eq!(elems_vec.len(), N);
            assert_eq!(N_EXP2, 1 << N);

            let mut elems_array = [F::ZERO; N];
            elems_array.copy_from_slice(elems_vec);

            let result_array = expand_subset_sums_array::<_, N, N_EXP2>(elems_array);
            let result_slice = expand_subset_sums(elems_vec);
            assert_eq!(result_array.as_ref(), result_slice.as_slice());
        }

        // Test with different sizes to verify array/slice consistency
        for n in 0..=4 {
            // Generate random input elements
            let elems_vec: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();

            // Test with the specific size n
            match n {
                0 => check_consistency::<0, 1>(&elems_vec),
                1 => check_consistency::<1, 2>(&elems_vec),
                2 => check_consistency::<2, 4>(&elems_vec),
                3 => check_consistency::<3, 8>(&elems_vec),
                4 => check_consistency::<4, 16>(&elems_vec),
                _ => unreachable!("n is constrained to 0..=4"),
            }
        }
    }
}
