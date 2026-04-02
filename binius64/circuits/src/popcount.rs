// Copyright 2025 Irreducible Inc.
//! Popcount circuit implementation for counting 1-bits in a 64-bit word.
//!
//! This module implements the population count (popcount) operation using
//! the SWAR (SIMD Within A Register) algorithm, optimized for Binius64's
//! constraint system.
//!
//! # Example
//!
//! ```rust,ignore
//! use binius_core::word::Word;
//! use binius_frontend::crate::popcount::popcount;
//! use binius_frontend::compiler::CircuitBuilder;
//!
//! // Build circuit
//! let mut builder = CircuitBuilder::new();
//! let input = builder.add_witness();
//! let output = builder.add_witness();
//! let computed = popcount(&mut builder, input);
//! builder.assert_eq("popcount_result", computed, output);
//! let circuit = builder.build();
//!
//! // Fill witness
//! let mut w = circuit.new_witness_filler();
//! w[input] = Word(0xFF);  // 8 bits set
//! w[output] = Word(8);
//!
//! // Verify
//! circuit.populate_wire_witness(&mut w).unwrap();
//! ```

use binius_frontend::{CircuitBuilder, Wire};

/// Computes the population count (number of 1-bits) of a 64-bit word.
///
/// This function implements the SWAR algorithm to efficiently count bits
/// using parallel operations within a single 64-bit register.
///
/// # Arguments
/// * `builder` - The circuit builder to add constraints to
/// * `input` - Wire containing the 64-bit value to count bits in
///
/// # Returns
/// * Wire containing the popcount result (value between 0 and 64)
pub fn popcount(builder: &mut CircuitBuilder, input: Wire) -> Wire {
    // SWAR Algorithm Implementation
    // Reference: https://nimrod.blog/posts/algorithms-behind-popcount/#swar-algorithm

    // Create constant masks used in SWAR
    let mask_5555 = builder.add_constant_64(0x5555555555555555); // 0101...
    let mask_3333 = builder.add_constant_64(0x3333333333333333); // 0011...
    let mask_0f0f = builder.add_constant_64(0x0F0F0F0F0F0F0F0F); // 00001111...
    let mask_00ff = builder.add_constant_64(0x00FF00FF00FF00FF); // 8 ones, 8 zeros...
    let mask_0000ffff = builder.add_constant_64(0x0000FFFF0000FFFF); // 16 ones, 16 zeros...
    let mask_00000000ffffffff = builder.add_constant_64(0x00000000FFFFFFFF); // 32 ones

    // Step 1: Count bits in 2-bit groups using subtraction trick
    // n = n - ((n >> 1) & 0x5555555555555555)
    let n_shr_1 = builder.shr(input, 1);
    let masked_shr_1 = builder.band(n_shr_1, mask_5555);
    let zero = builder.add_constant_64(0);
    let (n_step1, _borrow) = builder.isub_bin_bout(input, masked_shr_1, zero);

    // Step 2: Sum adjacent 2-bit groups into 4-bit groups
    // n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
    // IMPORTANT: We MUST mask before adding here!
    // After Step 1, 4-bit groups can contain values up to 1010 (binary) = 10 (decimal)
    // Without masking first, adding 1010 + 1010 = 10100 would overflow 4 bits
    // and create carries that corrupt adjacent groups.
    // Masking first ensures we only add the 2-bit counts (max value 2+2=4).
    let n_masked_3333 = builder.band(n_step1, mask_3333);
    let n_shr_2 = builder.shr(n_step1, 2);
    let n_shr_2_masked = builder.band(n_shr_2, mask_3333);
    let (n_step2, _carry) = builder.iadd_cin_cout(n_masked_3333, n_shr_2_masked, zero);

    // Step 3: Sum adjacent 4-bit groups into 8-bit groups
    // n = (n + (n >> 4)) & 0x0F0F0F0F0F0F0F0F
    // NOTE: Here we can safely add THEN mask (unlike Step 2)
    // After Step 2, max value per 4-bit group is 4 (0100 binary)
    // Adding 0100 + 0100 = 1000 (8) still fits in 4 bits, no overflow!
    let n_shr_4 = builder.shr(n_step2, 4);
    let (n_sum3, _carry) = builder.iadd_cin_cout(n_step2, n_shr_4, zero);
    let n_step3 = builder.band(n_sum3, mask_0f0f);

    // Step 4: Sum adjacent 8-bit groups into 16-bit groups
    // n = (n + (n >> 8)) & 0x00FF00FF00FF00FF
    let n_shr_8 = builder.shr(n_step3, 8);
    let (n_sum4, _carry) = builder.iadd_cin_cout(n_step3, n_shr_8, zero);
    let n_step4 = builder.band(n_sum4, mask_00ff);

    // Step 5: Sum adjacent 16-bit groups into 32-bit groups
    // n = (n + (n >> 16)) & 0x0000FFFF0000FFFF
    let n_shr_16 = builder.shr(n_step4, 16);
    let (n_sum5, _carry) = builder.iadd_cin_cout(n_step4, n_shr_16, zero);
    let n_step5 = builder.band(n_sum5, mask_0000ffff);

    // Step 6: Sum adjacent 32-bit groups to get final 64-bit result
    // n = (n + (n >> 32)) & 0x00000000FFFFFFFF
    let n_shr_32 = builder.shr(n_step5, 32);
    let (n_sum6, _carry) = builder.iadd_cin_cout(n_step5, n_shr_32, zero);

    // The final result is in the lower bits and represents the popcount (0-64)
    builder.band(n_sum6, mask_00000000ffffffff)
}

#[cfg(test)]
mod tests {
    // Note: Proptest uses deterministic seeding by default for reproducible tests.
    // The default seed is always 0 unless explicitly configured otherwise.
    // See: https://docs.rs/proptest/latest/proptest/test_runner/struct.Config.html#structfield.rng_algorithm
    use binius_core::word::Word;
    use proptest::prelude::*;

    use super::*;

    /// Helper function to build a test circuit with popcount
    fn build_popcount_circuit() -> (binius_frontend::Circuit, Wire, Wire) {
        let mut builder = CircuitBuilder::new();
        let input = builder.add_witness();
        let output = builder.add_witness();
        let computed = popcount(&mut builder, input);
        builder.assert_eq("popcount_result", computed, output);
        let circuit = builder.build();
        (circuit, input, output)
    }

    /// Helper to test a specific popcount value
    fn test_popcount_value(value: u64) {
        let (circuit, input, output) = build_popcount_circuit();
        let mut w = circuit.new_witness_filler();

        let expected = value.count_ones() as u64;
        w[input] = Word(value);
        w[output] = Word(expected);

        circuit
            .populate_wire_witness(&mut w)
            .unwrap_or_else(|_| panic!("Popcount of 0x{:016x} should be {}", value, expected));
    }

    #[test]
    fn test_popcount_basic() {
        // Build the circuit
        let (circuit, input, output) = build_popcount_circuit();

        // Create witness filler
        let mut w = circuit.new_witness_filler();

        // Test with a simple value
        let test_value = 0b10110101u64; // Binary: has 5 bits set
        let expected_count = test_value.count_ones() as u64;
        assert_eq!(expected_count, 5, "Expected count should be 5");

        w[input] = Word(test_value);
        w[output] = Word(expected_count);

        // Verify the circuit constraints are satisfied
        circuit
            .populate_wire_witness(&mut w)
            .expect("Circuit should be satisfied with correct popcount");
    }

    proptest! {
        #[test]
        fn test_popcount_edge_cases(
            value in prop::sample::select(vec![
                0x0000000000000000u64, // all zeros
                0xFFFFFFFFFFFFFFFFu64, // all ones
                0x5555555555555555u64, // alternating 01
                0xAAAAAAAAAAAAAAAAu64, // alternating 10
                0x00000000000000FFu64, // one byte set
                0x000000000000FFFFu64, // two bytes set
                0x00000000FFFFFFFFu64, // four bytes set
                0x0F0F0F0F0F0F0F0Fu64, // nibble pattern
                0xCCCCCCCCCCCCCCCCu64, // 11001100 pattern
            ])
        ) {
            test_popcount_value(value);
        }

        #[test]
        fn test_single_bit_positions(bit_pos in 0u32..64) {
            // Test single bit set at each position
            test_popcount_value(1u64 << bit_pos);
        }

        #[test]
        fn test_all_bits_except_one(byte_idx in 0usize..8) {
            // Test all bits set except one (sampling every 8th bit)
            test_popcount_value(0xFFFFFFFFFFFFFFFF ^ (1u64 << (byte_idx * 8)));
        }
    }

    proptest! {
        #[test]
        fn test_popcount_known_bit_counts(
            value in prop::sample::select(vec![
                0x0000000000000001u64, // 1 bit set
                0x0000000000000003u64, // 2 bits set
                0x000000000000000Fu64, // 4 bits set
                0x00000000000000FFu64, // 8 bits set
                0x000000000000FFFFu64, // 16 bits set
                0x00000000FFFFFFFFu64, // 32 bits set
                0x7FFFFFFFFFFFFFFFu64, // 63 bits set
            ])
        ) {
            test_popcount_value(value);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn test_popcount_random_values(value in any::<u64>()) {
            // Tests 100 random values
            test_popcount_value(value);
        }
    }

    proptest! {
        #[test]
        fn test_popcount_incorrect_output_fails(
            (input, wrong_offset) in prop::sample::select(vec![
                (0xFFu64, -1i8),        // 8 bits set, test with count 7
                (0x0Fu64, 1i8),         // 4 bits set, test with count 5
                (0xFFFFu64, -3i8),      // 16 bits set, test with count 13
                (0x7u64, 2i8),          // 3 bits set, test with count 5
            ])
        ) {
            let (circuit, input_wire, output_wire) = build_popcount_circuit();
            let mut w = circuit.new_witness_filler();

            let correct_count = input.count_ones() as u64;
            // Note: Wrapping arithmetic is intentional here. If wrong_offset is negative
            // and larger than correct_count, we get a very large u64 due to wrapping.
            // This is fine - we're testing that the circuit rejects ANY incorrect value,
            // including wrapped values. All our test cases avoid actual underflow.
            let wrong_count = (correct_count as i64 + wrong_offset as i64) as u64;

            w[input_wire] = Word(input);
            w[output_wire] = Word(wrong_count);

            let result = circuit.populate_wire_witness(&mut w);
            assert!(result.is_err(),
                "Circuit should fail with incorrect count {} instead of {}",
                wrong_count, correct_count);
        }

        #[test]
        fn test_popcount_overflow_fails(
            overflow_amount in 1u64..100
        ) {
            // Test that counts > 64 are rejected
            let (circuit, input_wire, output_wire) = build_popcount_circuit();
            let mut w = circuit.new_witness_filler();

            w[input_wire] = Word(0xFFFFFFFFFFFFFFFF);
            w[output_wire] = Word(64 + overflow_amount);

            let result = circuit.populate_wire_witness(&mut w);
            assert!(result.is_err(),
                "Circuit should fail with popcount = {}", 64 + overflow_amount);
        }
    }

    proptest! {
        #[test]
        fn test_step2_masking_edge_cases(
            value in prop::sample::select(vec![
                0xFFFFFFFFFFFFFFFFu64,  // All bits set - produces 0xAAAA... after Step 1
                0xCCCCCCCCCCCCCCCCu64,  // Pattern 11001100 - interesting Step 1 result
                0xF0F0F0F0F0F0F0F0u64,  // Pattern 11110000 - another edge case
                0xE0E0E0E0E0E0E0E0u64,  // Pattern 11100000 - tests masking
            ])
        ) {
            // These patterns specifically stress Step 2's mask-before-add requirement
            // After Step 1, they produce values that would overflow 4-bit groups
            // if we incorrectly added before masking
            test_popcount_value(value);
        }
    }

    proptest! {
        #[test]
        fn test_step3_onwards_no_overflow(
            pattern in prop::sample::select(vec![
                0x0F0F0F0F0F0F0F0Fu64, // nibble boundaries (tests Step 3)
                0x00FF00FF00FF00FFu64, // byte boundaries (tests Step 4)
                0x0000FFFF0000FFFFu64, // 16-bit boundaries (tests Step 5)
                0x00000000FFFFFFFFu64, // 32-bit boundaries (tests Step 6)
                0xFFFF00000000FFFFu64, // split pattern (tests accumulation)
            ])
        ) {
            // This verifies Steps 3-6 handle add-then-mask safely
            test_popcount_value(pattern);
        }
    }

    #[test]
    fn test_incorrect_step2_would_fail() {
        // This test documents why Step 2 MUST mask before adding.
        // We demonstrate this by showing what the incorrect result would be.

        // Simulate what would happen with incorrect Step 2 implementation
        let input: u64 = 0xFFFFFFFFFFFFFFFF;

        // Step 1: Correct subtraction
        let step1 = input.wrapping_sub((input >> 1) & 0x5555555555555555);
        assert_eq!(step1, 0xAAAAAAAAAAAAAAAA, "Step 1 should produce 0xAAAA...");

        // Incorrect Step 2: Add first, then mask (WRONG!)
        let incorrect_sum = step1.wrapping_add(step1 >> 2);
        let incorrect_step2 = incorrect_sum & 0x3333333333333333;

        // Correct Step 2: Mask first, then add
        let correct_step2 =
            (step1 & 0x3333333333333333).wrapping_add((step1 >> 2) & 0x3333333333333333);

        // Show they produce different results
        assert_ne!(
            incorrect_step2, correct_step2,
            "Incorrect Step 2 produces different result!"
        );

        // The incorrect approach gives wrong final count
        // Continue with incorrect value through remaining steps
        let mut incorrect = incorrect_step2;
        incorrect = (incorrect.wrapping_add(incorrect >> 4)) & 0x0F0F0F0F0F0F0F0F;
        incorrect = (incorrect.wrapping_add(incorrect >> 8)) & 0x00FF00FF00FF00FF;
        incorrect = (incorrect.wrapping_add(incorrect >> 16)) & 0x0000FFFF0000FFFF;
        incorrect = (incorrect.wrapping_add(incorrect >> 32)) & 0x00000000FFFFFFFF;

        // The incorrect approach gives 15 instead of 64!
        assert_eq!(incorrect, 15, "Incorrect Step 2 leads to wrong count of 15");
        assert_eq!(input.count_ones() as u64, 64, "Correct count should be 64");

        // This dramatic difference (15 vs 64) shows how badly the algorithm breaks
        // when Step 2 doesn't mask before adding!
        println!(
            "Demonstrated: Incorrect Step 2 gives {} instead of {}",
            incorrect,
            input.count_ones()
        );
    }
}
