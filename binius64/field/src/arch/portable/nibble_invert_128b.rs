// Copyright 2023-2025 Irreducible Inc.

//! Generic nibble-based inversion algorithm for 128-bit binary fields.
//!
//! This module provides a generic implementation of the nibble table-based inversion
//! algorithm that can be used for any 128-bit binary field, including GHASH.

use std::ops::BitXor;

use crate::{Field, arithmetic_traits::Square, underlier::WithUnderlier};

/// Generic nibble-based inversion algorithm for 128-bit binary fields.
///
/// This implements the algorithm from <https://github.com/reyzin/GF2t/blob/master/src/main/gf2t/GF2_128.java#L400>
/// adapted to work with any field F that implements Field and WithUnderlier<Underlier = u128>.
///
/// # Parameters
/// - `value`: The field element to invert
/// - `to_bases`: Function that converts field element to its basis representation (u128)
/// - `nibble_table`: Precomputed table of nibble powers of 2^(2^n)
///
/// # Returns
/// The multiplicative inverse of `value`, or zero if `value` is zero.
pub(super) fn nibble_invert_128b<F>(
    value: F,
    to_bases: impl Fn(F) -> u128,
    nibble_table: &[[[u128; 16]; 32]; 6],
) -> F
where
    F: Field + WithUnderlier<Underlier = u128> + Square,
{
    // Computes value^{2^128-2}
    // value * value^(2^128 - 2) = value^(2^128-1) = 1 if value != 0

    // Contains self raised to the power whose binary representation is 2^k ones
    let mut self_pow_2_pow_k1s = value;

    // Square result to get its exponent to be 10 in binary
    let mut res = pow_2_2_n(self_pow_2_pow_k1s, 0, &to_bases, nibble_table);

    // Contains self raised to the power whose binary representation is 2^k ones followed by 2^k
    // zeros
    let mut self_pow_2_pow_k1s_to_k0s = res;

    // Loop invariant:
    // res contains z raised to the power whose binary representation is 2^{k+1}-1 ones followed
    // by a single zero self_pow_2_pow_k1s contains z raised to the power whose binary
    // representation is 2^k ones self_pow_2_pow_k1s_to_k0s contains z raised to the power
    // whose binary representation is 2^k ones followed by 2^k zeros
    for k in 1..7 {
        // Fill in the zeros in the exponent of self_pow_2_pow_k1s_to_k0s with ones
        self_pow_2_pow_k1s *= self_pow_2_pow_k1s_to_k0s;

        // self_pow_2_pow_k1s_to_k0s = pow_2_2_n with 2^k zeros appended to the exponent
        self_pow_2_pow_k1s_to_k0s = pow_2_2_n(self_pow_2_pow_k1s, k, &to_bases, nibble_table);

        // prepend 2^k ones to res
        res *= self_pow_2_pow_k1s_to_k0s;
    }

    res
}

/// Generic implementation of pow_2_2_n that calculates `value^(2^(2^n))`.
///
/// This uses nibble tables for efficient computation when n >= 1.
pub(super) fn pow_2_2_n<F>(
    value: F,
    n: usize,
    to_bases: &impl Fn(F) -> u128,
    nibble_table: &[[[u128; 16]; 32]; 6],
) -> F
where
    F: Field + WithUnderlier<Underlier = u128> + Square,
{
    match n {
        // value^(2^(2^0)) = value^2
        0 => value.square(),
        1..=6 => {
            // Use the fact that for finite fields with characteristics 2
            // (a + b)^2 = a^2 + b^2, so we can compute the result nibble by nibble.
            let bases_form = to_bases(value);
            let result = (0..32)
                .map(|nibble_index| {
                    let nibble_value = (bases_form >> (nibble_index * 4)) & 0x0F;
                    nibble_table[n - 1][nibble_index][nibble_value as usize]
                })
                .fold(0, BitXor::bitxor);

            F::from_underlier(result)
        }
        _ => value,
    }
}

/// Generate nibble power table for a specific 128-bit binary field.
///
/// This function generates the precomputed table needed for efficient inversion.
/// Table format: `table[n][nibble_index][nibble_value] = F(nibble_value << 4 *
/// nibble_index)^(2^(2^(n+1)))`
///
/// # Parameters
/// - `new_fn`: Function to create field element from u128
/// - `to_u128`: Function to convert field element to u128
///
/// # Returns
/// The precomputed nibble power table that can be used with `nibble_invert_128b`.
#[allow(unused)]
pub fn generate_nibble_pow_2_n_table<F>(
    new_fn: impl Fn(u128) -> F,
    to_u128: impl Fn(F) -> u128,
) -> [[[u128; 16]; 32]; 6]
where
    F: Field + WithUnderlier<Underlier = u128> + Square,
{
    let mut results = [[[0u128; 16]; 32]; 6];

    for nibble_index in 0..32 {
        for nibble_value in 0..16 {
            let nibble_val = new_fn((nibble_value as u128) << (4 * nibble_index));
            for n in 0..6 {
                let mut result = nibble_val;
                // Compute result^(2^(2^(n+1)))
                for _ in 0..(1 << (n + 1)) {
                    result = result.square();
                }
                results[n][nibble_index][nibble_value] = to_u128(result);
            }
        }
    }

    results
}

/// Print the nibble power table in a format that can be copied into source code.
/// This is a utility function for generating static tables.
#[allow(unused)]
pub fn print_nibble_table(table: &[[[u128; 16]; 32]; 6], table_name: &str) {
    println!("static {table_name}: [[[u128; 16]; 32]; 6] = [");
    for n in 0..6 {
        println!("    [");
        for nibble_index in 0..32 {
            print!("        [");
            for value in table[n][nibble_index] {
                print!("{value}, ");
            }
            println!("],");
        }
        println!("    ],");
    }
    println!("];");
}
