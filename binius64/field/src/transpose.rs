// Copyright 2023-2025 Irreducible Inc.

use binius_utils::checked_arithmetics::log2_strict_usize;

use super::packed::PackedField;
use crate::{ExtensionField, Field, PackedExtension};

/// Transpose square blocks of elements within packed field elements in place.
///
/// The input elements are interpreted as a rectangular matrix with height `n = 2^n` in row-major
/// order. This matrix is interpreted as a vector of square matrices of field elements, and each
/// square matrix is transposed in-place.
///
/// # Arguments
///
/// * `log_n`: The base-2 logarithm of the dimension of the n x n square matrix. Must be less than
///   or equal to the base-2 logarithm of the packing width.
/// * `elems`: The packed field elements, length is a power-of-two multiple of `1 << log_n`.
///
/// # Preconditions
///
/// * `log_n` must be at most `P::LOG_WIDTH`.
/// * `elems.len()` must be a power of two and at least `2^log_n`.
pub fn square_transpose<P: PackedField>(log_n: usize, elems: &mut [P]) {
    assert!(
        P::LOG_WIDTH >= log_n,
        "dimension n of square blocks must divide packing width"
    );

    let size = elems.len();
    assert!(
        size.is_power_of_two(),
        "elems length must be a power of two, got {size}"
    );
    let log_size = log2_strict_usize(size);
    assert!(
        log_size >= log_n,
        "elems must have length at least 2^log_n = {}, got {size}",
        1 << log_n
    );

    let log_w = log_size - log_n;

    // See Hacker's Delight, Section 7-3.
    // https://dl.acm.org/doi/10.5555/2462741
    for i in 0..log_n {
        for j in 0..1 << (log_n - i - 1) {
            for k in 0..1 << (log_w + i) {
                let idx0 = (j << (log_w + i + 1)) | k;
                let idx1 = idx0 | (1 << (log_w + i));

                let v0 = elems[idx0];
                let v1 = elems[idx1];
                let (v0, v1) = v0.interleave(v1, i);
                elems[idx0] = v0;
                elems[idx1] = v1;
            }
        }
    }
}

pub fn square_transforms_extension_field<F: Field, FE: ExtensionField<F> + PackedExtension<F>>(
    values: &mut [FE],
) {
    square_transpose(FE::LOG_DEGREE, FE::cast_bases_mut(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PackedBinaryField128x1b;

    #[test]
    fn test_square_transpose_128x1b() {
        let mut elems = [
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
        ];
        square_transpose(3, &mut elems);

        let expected = [
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
            PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
        ];
        assert_eq!(elems, expected);
    }

    #[test]
    fn test_square_transpose_128x1b_multi_row() {
        let mut elems = [
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
            PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
        ];
        square_transpose(1, &mut elems);

        let expected = [
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
            PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
        ];
        assert_eq!(elems, expected);
    }
}
