// Copyright 2025 Irreducible Inc.
//! Byte manipulation circuits for Binius64.
//!
//! This module provides utility functions for byte-level operations on 64-bit words,
//! including byte swapping (endianness conversion).

use binius_frontend::{CircuitBuilder, Wire};

/// Swaps bytes within each 32-bit half of a 64-bit word independently.
///
/// This function performs byte swapping on the high and low 32-bit halves
/// of a 64-bit word in parallel. Each half has its byte order reversed
/// independently, without affecting the other half.
///
/// # Algorithm
///
/// The implementation uses the Hacker's Delight algorithm applied to both
/// 32-bit halves simultaneously:
///
/// 1. **Pass 1**: Swap adjacent bytes within each 32-bit half
///    - Mask: `0x00FF00FF00FF00FF`
///    - Operation: `((x & mask) << 8) | ((x >> 8) & mask)`
///
/// 2. **Pass 2**: Swap adjacent 16-bit units within each 32-bit half
///    - Mask: `0x0000FFFF0000FFFF`
///    - Operation: `((x & mask) << 16) | ((x >> 16) & mask)`
///
/// # Arguments
/// * `builder` - The circuit builder to add constraints to
/// * `input` - Wire containing the 64-bit value to process
///
/// # Returns
/// * Wire containing the result with bytes swapped within each 32-bit half
///
/// # Example
/// Input:  `0x0123456789ABCDEF` (bytes: 01 23 45 67 | 89 AB CD EF)
/// Output: `0x6745230189ABCDEF` (bytes: 67 45 23 01 | EF CD AB 89)
pub fn swap_bytes_32(builder: &CircuitBuilder, input: Wire) -> Wire {
    // Create constant masks for each pass
    let mask_00ff = builder.add_constant_64(0x00FF00FF00FF00FF);
    let mask_0000ffff = builder.add_constant_64(0x0000FFFF0000FFFF);

    // Pass 1: Swap adjacent bytes within each 32-bit half
    // x = ((x & 0x00FF00FF00FF00FF) << 8) | ((x >> 8) & 0x00FF00FF00FF00FF)
    let masked_input_bytes = builder.band(input, mask_00ff);
    let shl_8 = builder.shl(masked_input_bytes, 8);
    let shr_8 = builder.shr(input, 8);
    let masked_shr_8 = builder.band(shr_8, mask_00ff);
    let step1 = builder.bxor(shl_8, masked_shr_8);

    // Pass 2: Swap adjacent 16-bit units within each 32-bit half
    // x = ((x & 0x0000FFFF0000FFFF) << 16) | ((x >> 16) & 0x0000FFFF0000FFFF)
    let masked_step1_words = builder.band(step1, mask_0000ffff);
    let shl_16 = builder.shl(masked_step1_words, 16);
    let shr_16 = builder.shr(step1, 16);
    let masked_shr_16 = builder.band(shr_16, mask_0000ffff);
    builder.bxor(shl_16, masked_shr_16)
}

/// Reverses the byte order of a 64-bit word.
///
/// This function swaps the bytes of the input word, converting between
/// little-endian and big-endian representations. It implements the same
/// operation as the Rust standard library's `u64::swap_bytes()`.
///
/// # Algorithm
///
/// The implementation is decomposed into two steps:
/// 1. Swap bytes within each 32-bit half independently using `swap_bytes_32`
/// 2. Rotate the entire word by 32 bits to swap the halves
///
/// This is equivalent to the full Hacker's Delight byte reversal algorithm
/// but expressed more modularly.
///
/// # Arguments
/// * `builder` - The circuit builder to add constraints to
/// * `input` - Wire containing the 64-bit value to swap bytes of
///
/// # Returns
/// * Wire containing the byte-swapped result
///
/// # Cost Analysis
/// * Uses `swap_bytes_32`: 4 shifts, 4 ANDs, 2 XORs
/// * Plus 1 rotation (implemented as 2 shifts + 1 XOR internally)
/// * Total: 6 shift operations, 4 AND operations, 3 XOR operations
///
/// All shifts are free in Binius64 when part of constraints, making this
/// approach very efficient.
///
/// # Example
///
/// ```rust,ignore
/// use binius_core::word::Word;
/// use binius_frontend::crate::bytes::swap_bytes;
/// use binius_frontend::compiler::CircuitBuilder;
///
/// // Build circuit
/// let mut builder = CircuitBuilder::new();
/// let input = builder.add_witness();
/// let output = builder.add_witness();
/// let swapped = swap_bytes(&builder, input);
/// builder.assert_eq("swap_bytes_result", swapped, output);
/// let circuit = builder.build();
///
/// // Fill witness
/// let mut w = circuit.new_witness_filler();
/// w[input] = Word(0x0123456789ABCDEF);
/// w[output] = Word(0xEFCDAB8967452301);  // Bytes reversed
///
/// // Verify
/// circuit.populate_wire_witness(&mut w).unwrap();
/// ```
///
/// # Reference
/// Based on the byte swapping algorithm from "Hacker's Delight" by Henry S. Warren Jr.
pub fn swap_bytes(builder: &CircuitBuilder, input: Wire) -> Wire {
    // Step 1: Swap bytes within each 32-bit half independently
    let swapped_halves = swap_bytes_32(builder, input);

    // Step 2: Rotate by 32 bits to swap the two halves
    // This completes the full byte reversal
    builder.rotl(swapped_halves, 32)
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use proptest::prelude::*;

    use super::*;

    /// Helper function to test swap_bytes circuit with given input and expected output
    fn test_swap_bytes_helper(input_val: u64, expected: u64) {
        let builder = CircuitBuilder::new();
        let input = builder.add_witness();
        let output = builder.add_witness();
        let swapped = swap_bytes(&builder, input);
        builder.assert_eq("swap_bytes_result", swapped, output);
        let circuit = builder.build();

        let mut w = circuit.new_witness_filler();
        w[input] = Word(input_val);
        w[output] = Word(expected);

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
    }

    /// Helper function to test swap_bytes_32 circuit with given input and expected output
    fn test_swap_bytes_32_helper(input_val: u64, expected: u64) {
        let builder = CircuitBuilder::new();
        let input = builder.add_witness();
        let output = builder.add_witness();
        let swapped = swap_bytes_32(&builder, input);
        builder.assert_eq("swap_bytes_32_result", swapped, output);
        let circuit = builder.build();

        let mut w = circuit.new_witness_filler();
        w[input] = Word(input_val);
        w[output] = Word(expected);

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
    }

    proptest! {
        #[test]
        fn test_swap_bytes_random(input_val: u64) {
            let expected = input_val.swap_bytes();
            test_swap_bytes_helper(input_val, expected);
        }

        #[test]
        fn test_swap_bytes_32_random(input_val: u64) {
            // For each 32-bit half, swap its bytes
            let lo = (input_val & 0xFFFFFFFF) as u32;
            let hi = ((input_val >> 32) & 0xFFFFFFFF) as u32;
            let swapped_lo = lo.swap_bytes() as u64;
            let swapped_hi = (hi.swap_bytes() as u64) << 32;
            let expected = swapped_hi | swapped_lo;
            test_swap_bytes_32_helper(input_val, expected);
        }
    }
}
