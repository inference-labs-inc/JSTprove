// Copyright 2025 Irreducible Inc.

//! Vision-4 hash permutation implementation.
//!
//! Core permutation functions for the Vision-4 cryptographic hash, operating on 4-element
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

/// Applies MDS matrix multiplication for optimal diffusion.
pub fn mds_mul(a: &mut [Ghash]) {
    // a = [a0, a1, a2, a3]
    let sum = a[0] + a[1] + a[2] + a[3];
    let a0 = a[0];

    // 2*a0 + 3*a1 + a2 + a3
    a[0] += sum + (a[0] + a[1]).mul_x();

    // a0 + 2*a1 + 3*a2 + a3
    a[1] += sum + (a[1] + a[2]).mul_x();

    // a0 + a1 + 2*a2 + 3*a3
    a[2] += sum + (a[2] + a[3]).mul_x();

    // 3*a0 + a1 + a2 + 2*a3
    a[3] += sum + (a[3] + a0).mul_x();
}

/// Adds round constants to prevent slide attacks.
pub fn constants_add(state: &mut [Ghash], constants: &[Ghash]) {
    for i in 0..M {
        state[i] += constants[i];
    }
}

/// Executes a single Vision-4 round with two S-box applications.
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

/// Main Vision-4 permutation function operating on 4-element states.
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

    use binius_field::Random;
    use rand::{SeedableRng, rngs::StdRng};

    use super::{super::constants::tests::matrix_mul, *};

    #[test]
    fn test_mds() {
        let mut rng = StdRng::seed_from_u64(0);
        let input: [Ghash; M] = std::array::from_fn(|_| Ghash::random(&mut rng));

        let matrix: [Ghash; M * M] = [
            2, 3, 1, 1, //
            1, 2, 3, 1, //
            1, 1, 2, 3, //
            3, 1, 1, 2,
        ]
        .map(Ghash::new);
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
                    Ghash::new(0x5e9a7b63d8d1a93953d56ceb6dcf6a35),
                    Ghash::new(0xa3262c57f6cdd8c368639c1a4f01ab5a),
                    Ghash::new(0x1dc99e37723063c4f178826d2a6802e3),
                    Ghash::new(0xfdf935c9d9fae3d560a75026a049bf7c),
                ],
            ),
            (
                [
                    Ghash::new(0xdeadbeef),
                    Ghash::new(0x0),
                    Ghash::new(0xdeadbeef),
                    Ghash::new(0x0),
                ],
                [
                    Ghash::new(0x1d02eaf6cf48c108a2ae1d9e27812364),
                    Ghash::new(0xc9bae4f4c782d46ed28245525f04fb3c),
                    Ghash::new(0xf4fea518a1e62f97748266e86acac536),
                    Ghash::new(0x22b25c68a52fef4b855f8862bdd418c4),
                ],
            ),
            (
                array::from_fn(|_| Ghash::random(&mut rng)),
                [
                    Ghash::new(0xdd1c99b8f9f2ec20abf21f082a56c9f3),
                    Ghash::new(0x3f5ec0a548673b571ba93d7751c98624),
                    Ghash::new(0xe1c5c8fc8f4c80cfa8841cfd0ae0fbbb),
                    Ghash::new(0xa054cc0d7379b474df8726cb448ca22b),
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
