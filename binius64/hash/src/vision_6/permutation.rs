// Copyright 2025 Irreducible Inc.

//! Vision-6 hash permutation implementation.
//!
//! Core permutation functions for the Vision-6 cryptographic hash, operating on 6-element
//! states over GF(2^128). Each round applies: S-box → MDS → constants → S-box → MDS → constants.

use binius_field::{BinaryField128bGhash as Ghash, Divisible, WithUnderlier};
use binius_math::batch_invert::BatchInversion;

use super::{
    constants::{B_FWD_COEFFS, B_INV_COEFFS, BYTES_PER_GHASH, M, NUM_ROUNDS, ROUND_CONSTANTS},
    linear_tables::{LINEAR_B_FWD_TABLE, LINEAR_B_INV_TABLE},
};

/// Applies linearized B⁻¹ transformation to a single field element using lookup tables.
pub fn linearized_b_inv_transform_scalar(x: &mut Ghash) {
    linearized_transform_scalar(x, &LINEAR_B_INV_TABLE);
    *x += B_INV_COEFFS[0];
}

/// Applies linearized transformation using precomputed lookup tables for efficiency.
pub fn linearized_transform_scalar(x: &mut Ghash, table: &'static [[Ghash; 256]; BYTES_PER_GHASH]) {
    *x = <u128 as Divisible<u8>>::ref_iter(x.to_underlier_ref())
        .zip(table)
        .map(|(byte, lookup)| lookup[byte as usize])
        .sum();
}

/// Applies forward B-polynomial transformation to all elements in state.
pub fn b_fwd_transform<const N: usize>(state: &mut [Ghash; N]) {
    (0..N).for_each(|i| {
        linearized_transform_scalar(&mut state[i], &LINEAR_B_FWD_TABLE);
        state[i] += B_FWD_COEFFS[0];
    });
}

/// Applies inverse B-polynomial transformation to all elements in state.
pub fn b_inv_transform<const N: usize>(state: &mut [Ghash; N]) {
    (0..N).for_each(|i| {
        linearized_transform_scalar(&mut state[i], &LINEAR_B_INV_TABLE);
        state[i] += B_INV_COEFFS[0];
    });
}

/// S-box operation: batch inversion followed by polynomial transformation.
pub fn sbox(
    state: &mut [Ghash; M],
    transform: impl Fn(&mut [Ghash; M]),
    inverter: &mut BatchInversion<Ghash>,
) {
    inverter.invert_or_zero(state);
    transform(state);
}

/// Applies MDS matrix multiplication for optimal diffusion using circulant structure.
pub fn mds_mul(a: &mut [Ghash]) {
    // a = [a0, a1, a2, a3, a4, a5]
    // Matrix first row: [x^(-1), x^(-1), x, 1, x, x + 1] = [1/2, 1/2, 2, 1, 2, 3]
    // This is circulant, so we can use efficient structure

    // Save original values
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let a5 = a[5];

    // Row 0: [x^(-1), x^(-1), x, 1, x, x + 1] * [a0, a1, a2, a3, a4, a5]
    //      = (1/2)*a0 + (1/2)*a1 + 2*a2 + 1*a3 + 2*a4 + 3*a5
    a[0] = (a0 + a1).mul_inv_x() + (a2 + a4).mul_x() + a3 + (a5.mul_x() + a5);

    // Row 1: [x + 1, x^(-1), x^(-1), x, 1, x] * [a0, a1, a2, a3, a4, a5]
    //      = 3*a0 + (1/2)*a1 + (1/2)*a2 + 2*a3 + 1*a4 + 2*a5
    a[1] = (a0.mul_x() + a0) + (a1 + a2).mul_inv_x() + a3.mul_x() + a4 + a5.mul_x();

    // Row 2: [x, x + 1, x^(-1), x^(-1), x, 1] * [a0, a1, a2, a3, a4, a5]
    //      = 2*a0 + 3*a1 + (1/2)*a2 + (1/2)*a3 + 2*a4 + 1*a5
    a[2] = a0.mul_x() + (a1.mul_x() + a1) + (a2 + a3).mul_inv_x() + a4.mul_x() + a5;

    // Row 3: [1, x, x + 1, x^(-1), x^(-1), x] * [a0, a1, a2, a3, a4, a5]
    //      = 1*a0 + 2*a1 + 3*a2 + (1/2)*a3 + (1/2)*a4 + 2*a5
    a[3] = a0 + a1.mul_x() + (a2.mul_x() + a2) + (a3 + a4).mul_inv_x() + a5.mul_x();

    // Row 4: [x, 1, x, x + 1, x^(-1), x^(-1)] * [a0, a1, a2, a3, a4, a5]
    //      = 2*a0 + 1*a1 + 2*a2 + 3*a3 + (1/2)*a4 + (1/2)*a5
    a[4] = a0.mul_x() + a1 + a2.mul_x() + (a3.mul_x() + a3) + (a4 + a5).mul_inv_x();

    // Row 5: [x^(-1), x, 1, x, x + 1, x^(-1)] * [a0, a1, a2, a3, a4, a5]
    //      = (1/2)*a0 + 2*a1 + 1*a2 + 2*a3 + 3*a4 + (1/2)*a5
    a[5] = (a0 + a5).mul_inv_x() + a1.mul_x() + a2 + a3.mul_x() + (a4.mul_x() + a4);
}

/// Adds round constants to prevent slide attacks.
pub fn constants_add(state: &mut [Ghash], constants: &[Ghash]) {
    for i in 0..M {
        state[i] += constants[i];
    }
}

/// Executes a single Vision-6 round with two S-box applications.
fn round(state: &mut [Ghash; M], round_constants_idx: usize, inverter: &mut BatchInversion<Ghash>) {
    // First half
    sbox(state, b_inv_transform, inverter);
    mds_mul(state);
    constants_add(state, &ROUND_CONSTANTS[round_constants_idx]);
    // Second half
    sbox(state, b_fwd_transform, inverter);
    mds_mul(state);
    constants_add(state, &ROUND_CONSTANTS[round_constants_idx + 1]);
}

/// Main Vision-6 permutation function operating on 6-element states.
pub fn permutation(state: &mut [Ghash; M]) {
    constants_add(state, &ROUND_CONSTANTS[0]);
    let mut inverter = BatchInversion::<Ghash>::new(M);
    for round_num in 0..NUM_ROUNDS {
        round(state, 1 + 2 * round_num, &mut inverter);
    }
}

#[cfg(test)]
mod tests {

    use std::array;

    use binius_field::{Field, Random};
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    fn matrix_mul(matrix: &[Ghash; M * M], input: &[Ghash; M]) -> [Ghash; M] {
        let mut result = [Ghash::ZERO; M];
        for i in 0..M {
            for j in 0..M {
                result[i] += matrix[i * M + j] * input[j];
            }
        }
        result
    }

    #[test]
    fn test_mds() {
        use rand::{SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(0);
        let input: [Ghash; M] = std::array::from_fn(|_| Ghash::random(&mut rng));

        // 6x6 circulant matrix: x = 2, x^(-1) = 1/2, x + 1 = 3
        let x_inv = Ghash::new(2).invert().expect("2 is invertible");
        let x = Ghash::new(2);
        let x_plus_1 = Ghash::new(3);
        let one = Ghash::ONE;

        let matrix: [Ghash; M * M] = [
            x_inv, x_inv, x, one, x, x_plus_1, // Row 0
            x_plus_1, x_inv, x_inv, x, one, x, // Row 1
            x, x_plus_1, x_inv, x_inv, x, one, // Row 2
            one, x, x_plus_1, x_inv, x_inv, x, // Row 3
            x, one, x, x_plus_1, x_inv, x_inv, // Row 4
            x_inv, x, one, x, x_plus_1, x_inv, // Row 5
        ];
        let expected = matrix_mul(&matrix, &input);

        let mut actual = input;
        mds_mul(&mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_permutation() {
        let mut rng = StdRng::seed_from_u64(0);
        // Outputs computed from a Sage script
        let cases = [
            (
                array::from_fn(|_| Ghash::new(0x0)),
                [
                    Ghash::new(0xd41c58ea75c2e3a8e5004834f122d650),
                    Ghash::new(0xb1a6fb890a3a7520384c2e21f6dcc18d),
                    Ghash::new(0xe4ef3f5c84fe8bef518f57ee1f38dc05),
                    Ghash::new(0x9cb1081fc97c17719c9527727f991bc1),
                    Ghash::new(0x42ae4487ccb1a4af24ad33acf8f9a8cd),
                    Ghash::new(0x53faafdae4007e9983ec18971b8ce524),
                ],
            ),
            (
                [
                    Ghash::new(0xdeadbeef),
                    Ghash::new(0x0),
                    Ghash::new(0xdeadbeef),
                    Ghash::new(0x0),
                    Ghash::new(0xdeadbeef),
                    Ghash::new(0x0),
                ],
                [
                    Ghash::new(0x6cef13e30578bbc055e541b8daae5525),
                    Ghash::new(0xe19d52ae54a01f3aacdb3dd2b8968a5f),
                    Ghash::new(0xca3289530c76d0c696e313ed5c1b7727),
                    Ghash::new(0x14dc021e84aa3ce6e7bb3a9452f61adc),
                    Ghash::new(0xde1d940f6c8b7d4869f02157f2f939df),
                    Ghash::new(0xd19a101d6d736dacaedad738dd35596a),
                ],
            ),
            (
                array::from_fn(|_| Ghash::random(&mut rng)),
                [
                    Ghash::new(0x54cf96e0da8f01e8b8a4688cd0f8b881),
                    Ghash::new(0xdbbc6cb3d5a96cee7d5ad99fe00f7874),
                    Ghash::new(0xaa44b45bed826f5baa02979e91593a7b),
                    Ghash::new(0x679fd76c310f55de3d216c5b4572597d),
                    Ghash::new(0x535121f744503928e021d9c4b8a56c46),
                    Ghash::new(0xc7de0f11beaf12aed3cdd8d3c4d1b4b8),
                ],
            ),
        ];

        for (input, expected) in cases {
            let mut state = input;
            permutation(&mut state);
            assert_eq!(state, expected);
        }
    }
}
