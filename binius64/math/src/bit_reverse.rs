// Copyright 2025 Irreducible Inc.

use binius_field::{PackedField, square_transpose};
use binius_utils::{checked_arithmetics::log2_strict_usize, rayon::prelude::*};
use bytemuck::zeroed_vec;

use crate::field_buffer::FieldSliceMut;

/// Reverses the low `bits` bits of an unsigned integer.
///
/// # Arguments
///
/// * `x` - The value whose bits to reverse
/// * `bits` - The number of low-order bits to reverse
///
/// # Returns
///
/// The value with its low `bits` bits reversed
pub fn reverse_bits(x: usize, bits: u32) -> usize {
    x.reverse_bits().unbounded_shr(usize::BITS - bits)
}

/// Applies a bit-reversal permutation to packed field elements in a buffer using parallelization.
///
/// This function permutes the field elements such that element at index `i` is moved to
/// index `reverse_bits(i, log_len)`. The permutation is performed in-place and correctly
/// handles packed field representations.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of packed field elements to permute
pub fn bit_reverse_packed<P: PackedField>(mut buffer: FieldSliceMut<P>) {
    // The algorithm has two parallelized phases:
    // 1. Process P::WIDTH x P::WIDTH submatrices in parallel
    // 2. Apply bit-reversal to independent chunks in parallel

    let log_len = buffer.log_len();
    if log_len < 2 * P::LOG_WIDTH {
        return bit_reverse_packed_naive(buffer);
    }

    let bits = (log_len - P::LOG_WIDTH) as u32;
    let data = buffer.as_mut();

    // Phase 1: Process submatrices in parallel
    // Each iteration accesses disjoint memory locations, so parallelization is safe
    let data_ptr = data.as_mut_ptr() as usize;
    (0..1 << (log_len - 2 * P::LOG_WIDTH))
        .into_par_iter()
        .for_each_init(
            || zeroed_vec::<P>(P::WIDTH),
            |tmp, i| {
                // SAFETY: Different values of i access non-overlapping submatrices.
                // The indexing pattern reverse_bits(j, bits) | i ensures that:
                // - reverse_bits(j, bits) places j in the high bits
                // - | i places i in the low bits
                // Therefore, different i values access completely disjoint index sets.
                unsafe {
                    let data = data_ptr as *mut P;
                    for j in 0..P::WIDTH {
                        tmp[j] = *data.add(reverse_bits(j, bits) | i);
                    }
                }
                square_transpose(P::LOG_WIDTH, tmp);
                unsafe {
                    let data = data_ptr as *mut P;
                    for j in 0..P::WIDTH {
                        *data.add(reverse_bits(j, bits) | i) = tmp[j];
                    }
                }
            },
        );

    // Phase 2: Apply bit_reverse_indices to chunks in parallel
    // Chunks are non-overlapping, so this is safe
    data.par_chunks_mut(1 << (log_len - 2 * P::LOG_WIDTH))
        .for_each(|chunk| {
            bit_reverse_indices(chunk);
        });
}

/// Applies a bit-reversal permutation to packed field elements using a simple algorithm.
///
/// This is a straightforward reference implementation that directly swaps field elements
/// according to the bit-reversal permutation. It serves as a baseline for correctness
/// testing of optimized implementations.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of packed field elements to permute
fn bit_reverse_packed_naive<P: PackedField>(mut buffer: FieldSliceMut<P>) {
    let bits = buffer.log_len() as u32;
    for i in 0..buffer.len() {
        let i_rev = reverse_bits(i, bits);
        if i < i_rev {
            let tmp = buffer.get(i);
            buffer.set(i, buffer.get(i_rev));
            buffer.set(i_rev, tmp);
        }
    }
}

/// Applies a bit-reversal permutation to elements in a slice using parallel iteration.
///
/// This function permutes the elements such that element at index `i` is moved to
/// index `reverse_bits(i, log2(length))`. The permutation is performed in-place
/// by swapping elements in parallel.
///
/// # Arguments
///
/// * `buffer` - Mutable slice of elements to permute
///
/// # Panics
///
/// Panics if the buffer length is not a power of two.
pub fn bit_reverse_indices<T>(buffer: &mut [T]) {
    let bits = log2_strict_usize(buffer.len()) as u32;

    // We need to use UnsafeCell-like semantics here to get proper Sync behavior.
    // Creating a raw pointer from the slice inside the closure avoids Sync issues.
    let buffer_ptr = buffer.as_mut_ptr() as usize;

    (0..buffer.len()).into_par_iter().for_each(|i| {
        let i_rev = reverse_bits(i, bits);
        if i < i_rev {
            // SAFETY: The i < i_rev condition guarantees that:
            // 1. Each (i, i_rev) pair is processed by exactly one thread (the one with i < i_rev)
            // 2. Since bit-reversal is bijective, no two threads access the same pair
            // 3. Therefore, ptr.add(i) and ptr.add(i_rev) point to disjoint memory locations
            // 4. No data races can occur
            // 5. buffer_ptr is valid for the lifetime of this closure
            unsafe {
                let ptr = buffer_ptr as *mut T;
                let ptr_i = ptr.add(i);
                let ptr_i_rev = ptr.add(i_rev);
                std::ptr::swap_nonoverlapping(ptr_i, ptr_i_rev, 1);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::test_utils::{Packed128b, random_field_buffer};

    // For Packed128b (PackedBinaryGhash4x128b), LOG_WIDTH = 2, so 2 * LOG_WIDTH = 4
    // Test three cases around the threshold where bit_reverse_packed switches between
    // naive and optimized implementations
    #[rstest::rstest]
    #[case::below_threshold(3)] // log_d < 2 * P::LOG_WIDTH
    #[case::at_threshold(4)] // log_d == 2 * P::LOG_WIDTH
    #[case::above_threshold(8)] // log_d > 2 * P::LOG_WIDTH
    fn test_bit_reverse_packed_equivalence(#[case] log_d: usize) {
        let mut rng = StdRng::seed_from_u64(0);

        let data_orig = random_field_buffer::<Packed128b>(&mut rng, log_d);

        let mut data_optimized = data_orig.clone();
        let mut data_naive = data_orig.clone();

        bit_reverse_packed(data_optimized.to_mut());
        bit_reverse_packed_naive(data_naive.to_mut());

        assert_eq!(data_optimized, data_naive, "Mismatch at log_d={}", log_d);
    }
}
