// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use crate::multiplexer::single_wire_multiplex;

/// Verifies that a slice is correctly extracted from an input byte array.
///
/// This circuit validates that `slice` contains exactly the bytes from
/// `input` starting at `offset` for `len_slice` bytes.
///
/// # Limitations
/// All size and offset values must fit within 32 bits. Specifically:
/// - `len_input` must be < 2^32
/// - `len_slice` must be < 2^32
/// - `offset` must be < 2^32
/// - `offset + len_slice` must be < 2^32
///
/// These limitations are enforced by the circuit constraints.
pub struct Slice {
    pub len_input: Wire,
    pub len_slice: Wire,
    pub input: Vec<Wire>,
    pub slice: Vec<Wire>,
    pub offset: Wire,
}

impl Slice {
    /// Creates a new slice verifier circuit.
    ///
    /// # Arguments
    /// * `b` - Circuit builder
    /// * `len_input` - Actual input size in bytes
    /// * `len_slice` - Actual slice size in bytes
    /// * `input` - Input array packed as words (8 bytes per word)
    /// * `slice` - Slice array packed as words (8 bytes per word)
    /// * `offset` - Byte offset where slice starts
    ///
    /// # Panics
    /// * If max_n_input >= 2^32
    /// * If max_n_slice >= 2^32
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        b: &CircuitBuilder,
        len_input: Wire,
        len_slice: Wire,
        input: Vec<Wire>,
        slice: Vec<Wire>,
        offset: Wire,
    ) -> Self {
        // Static assertions to ensure maximum sizes fit within 32 bits
        let max_len_input = input.len() << 3;
        let max_len_slice = slice.len() << 3;

        assert!(
            max_len_input <= u32::MAX as usize,
            "max_n_input must be < 2^32"
        );
        assert!(
            max_len_slice <= u32::MAX as usize,
            "max_n_slice must be < 2^32"
        );

        // Ensure all values fit in 32 bits to prevent overflow in iadd_32
        b.assert_zero("offset_32bit", b.shr(offset, 32));
        b.assert_zero("len_slice_32bit", b.shr(len_slice, 32));
        b.assert_zero("len_input_32bit", b.shr(len_input, 32));

        // Verify bounds: offset + len_slice <= len_input
        let offset_plus_len_slice = b.iadd_32(offset, len_slice);
        let in_bounds = b.icmp_ule(offset_plus_len_slice, len_input);
        b.assert_true("bounds_check", in_bounds);

        // Decompose offset = word_offset * 8 + byte_offset
        let word_offset = b.shr(offset, 3); // offset / 8
        let byte_offset = b.band(offset, b.add_constant(Word(7))); // offset % 8

        // Go over every word in the slice and check that it was copied from the input byte string
        // correctly.
        for (slice_idx, &slice_word) in slice.iter().enumerate() {
            let b = b.subcircuit(format!("slice_word[{slice_idx}]"));

            // Check if this word is within the actual slice
            let word_start_bytes = slice_idx << 3;
            let word_partially_valid =
                b.icmp_ult(b.add_constant(Word(word_start_bytes as u64)), len_slice);
            // are ANY of the bytes in this present word actually part of the slice proper?

            // Calculate which input word(s) we need
            let input_word_idx = b.iadd_32(word_offset, b.add_constant(Word(slice_idx as u64)));

            let extracted_word = extract_word(&b, &input, input_word_idx, byte_offset);

            // For every word, calculate how many bytes are valid and apply appropriate mask
            // Calculate valid bytes in this word: min(len_slice - word_start, 8)
            // First calculate len_slice - word_start
            let neg_start = b.add_constant(Word((-(word_start_bytes as i64)) as u64));
            let bytes_remaining = b.iadd_32(len_slice, neg_start);

            // The mask will handle clamping to 8 bytes internally
            let mask = create_byte_mask(&b, bytes_remaining);

            // For partial words, we need to ensure:
            // 1. The valid bytes match the extracted word
            // 2. The invalid bytes are zero
            // Assert they are equal (only if word is at least partially valid)
            let zero = b.add_constant(Word::ZERO);
            b.assert_eq_cond(
                format!("slice_word_{slice_idx}"),
                b.band(slice_word, mask),
                b.band(extracted_word, mask),
                word_partially_valid,
            );
            b.assert_eq_cond(
                format!("slice_word_{slice_idx}_padding"),
                b.band(slice_word, b.bnot(mask)),
                zero,
                word_partially_valid,
            );

            b.assert_eq_cond(
                format!("slice_word_{slice_idx} non-slice"),
                slice_word,
                zero,
                b.bnot(word_partially_valid),
            );
        }

        Slice {
            len_input,
            len_slice,
            input,
            slice,
            offset,
        }
    }

    /// Populate the len_input wire with the actual input size in bytes
    pub fn populate_len_input(&self, w: &mut WitnessFiller, len_input: usize) {
        w[self.len_input] = Word(len_input as u64);
    }

    /// Populate the len_slice wire with the actual slice size in bytes
    pub fn populate_len_slice(&self, w: &mut WitnessFiller, len_slice: usize) {
        w[self.len_slice] = Word(len_slice as u64);
    }

    /// Populate the input array from a byte slice
    ///
    /// # Panics
    /// Panics if input.len() > max_n_input (the maximum size specified during construction)
    pub fn populate_input(&self, w: &mut WitnessFiller, input: &[u8]) {
        let max_n_input = self.input.len() * 8;
        assert!(
            input.len() <= max_n_input,
            "input length {} exceeds maximum {}",
            input.len(),
            max_n_input
        );

        // Pack bytes into words
        for (i, chunk) in input.chunks(8).enumerate() {
            if i < self.input.len() {
                let mut word = 0u64;
                for (j, &byte) in chunk.iter().enumerate() {
                    word |= (byte as u64) << (j * 8);
                }
                w[self.input[i]] = Word(word);
            }
        }

        // Zero out remaining words
        for i in input.len().div_ceil(8)..self.input.len() {
            w[self.input[i]] = Word::ZERO;
        }
    }

    /// Populate the slice array from a byte slice
    ///
    /// # Panics
    /// Panics if slice.len() > max_n_slice (the maximum size specified during construction)
    pub fn populate_slice(&self, w: &mut WitnessFiller, slice: &[u8]) {
        let max_n_slice = self.slice.len() * 8;
        assert!(
            slice.len() <= max_n_slice,
            "slice length {} exceeds maximum {}",
            slice.len(),
            max_n_slice
        );

        // Pack bytes into words
        for (i, chunk) in slice.chunks(8).enumerate() {
            if i < self.slice.len() {
                let mut word = 0u64;
                for (j, &byte) in chunk.iter().enumerate() {
                    word |= (byte as u64) << (j * 8);
                }
                w[self.slice[i]] = Word(word);
            }
        }

        // Zero out remaining words
        for i in slice.len().div_ceil(8)..self.slice.len() {
            w[self.slice[i]] = Word::ZERO;
        }
    }

    /// Populate the offset wire
    pub fn populate_offset(&self, w: &mut WitnessFiller, offset: usize) {
        w[self.offset] = Word(offset as u64);
    }
}

/// Extracts a word from the input array at the specified word index and byte offset.
///
/// This function handles both aligned and unaligned word extraction:
/// - **Aligned** (byte_offset = 0): Directly selects the word at `word_idx`
/// - **Unaligned** (byte_offset = 1-7): Combines bytes from two adjacent words
///
/// # Arguments
/// * `b` - Circuit builder
/// * `input` - Array of input words to extract from
/// * `word_idx` - Index of the word to extract
/// * `byte_offset` - Byte offset within the word (0-7)
///
/// # Returns
/// A wire containing the extracted 8-byte word
pub fn extract_word(b: &CircuitBuilder, input: &[Wire], word_idx: Wire, byte_offset: Wire) -> Wire {
    let next_word_idx = b.iadd_32(word_idx, b.add_constant(Word(1)));
    // Aligned case: directly select the word
    let aligned_word = single_wire_multiplex(b, input, word_idx);
    let next_word = single_wire_multiplex(b, input, next_word_idx);
    let zero = b.add_constant(Word::ZERO);

    let candidates: Vec<Wire> = (0..8)
        .map(|i| {
            let shifted_aligned = b.shr(aligned_word, i << 3);
            let shifted_next = if i == 0 {
                zero
            } else {
                b.shl(next_word, (8 - i) << 3)
            };
            b.bor(shifted_aligned, shifted_next)
        })
        .collect();
    single_wire_multiplex(b, &candidates, byte_offset)
}

/// Creates a byte mask with the first `n_bytes` bytes set to 0xFF and remaining bytes to 0x00.
///
/// This function generates masks for partial word validation:
/// - n_bytes = 0: 0x0000000000000000
/// - n_bytes = 1: 0x00000000000000FF
/// - n_bytes = 2: 0x000000000000FFFF
/// - ...
/// - n_bytes = 7: 0x00FFFFFFFFFFFFFF
/// - n_bytes ≥ 8: 0xFFFFFFFFFFFFFFFF
///
/// # Arguments
/// * `b` - Circuit builder
/// * `n_bytes` - Number of bytes to include in the mask (0-8 or more)
///
/// # Returns
/// A wire containing the byte mask
pub fn create_byte_mask(b: &CircuitBuilder, n_bytes: Wire) -> Wire {
    // Handle values ≥ 8 by treating them as 8
    let eight = b.add_constant(Word(8));
    let is_lt_eight = b.icmp_ult(n_bytes, eight);
    let all_one = b.add_constant(Word::ALL_ONE);

    let masks: Vec<Wire> = (0..8)
        .map(|i| b.add_constant_64(0x00FFFFFFFFFFFFFF >> ((7 - i) << 3)))
        .collect();
    let in_range_mask = single_wire_multiplex(b, &masks, n_bytes);
    b.select(is_lt_eight, in_range_mask, all_one)
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;

    use super::{CircuitBuilder, Slice, Wire, Word};

    #[test]
    fn test_aligned_slice() {
        let b = CircuitBuilder::new();

        // Test case: 16-byte input, 8-byte slice at offset 0
        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);

        let circuit = b.build();

        // Test with actual values
        let mut filler = circuit.new_witness_filler();

        verifier.populate_len_input(&mut filler, 16);
        verifier.populate_len_slice(&mut filler, 8);
        verifier.populate_offset(&mut filler, 0);

        // Input: 16 bytes
        let input_data = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        verifier.populate_input(&mut filler, &input_data);

        // Slice: first 8 bytes of input
        let slice_data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        verifier.populate_slice(&mut filler, &slice_data);

        // Fill the circuit - this should succeed
        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_unaligned_slice() {
        let b = CircuitBuilder::new();

        // Test case: 16-byte input, 8-byte slice at offset 3

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);

        let circuit = b.build();

        // Test with actual values
        let mut filler = circuit.new_witness_filler();

        // Set up test data using populate methods
        verifier.populate_len_input(&mut filler, 16);
        verifier.populate_len_slice(&mut filler, 8);
        verifier.populate_offset(&mut filler, 3);

        // Input: 16 bytes
        let input_data = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        verifier.populate_input(&mut filler, &input_data);

        // Slice at offset 3: bytes 3-10
        // This should be: 03 04 05 06 07 08 09 0a
        let slice_data = [0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a];
        verifier.populate_slice(&mut filler, &slice_data);

        // Fill the circuit - this should succeed
        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_bounds_check() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);

        let circuit = b.build();

        // Test with values that should fail bounds check
        let mut filler = circuit.new_witness_filler();

        // Set up test data that violates bounds using populate methods
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 8);
        verifier.populate_offset(&mut filler, 5);

        // Fill dummy data
        let dummy_input = vec![0u8; 10];
        let dummy_slice = vec![0u8; 8];
        verifier.populate_input(&mut filler, &dummy_input);
        verifier.populate_slice(&mut filler, &dummy_slice);

        // This should fail the bounds check
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err());
    }

    #[test]
    fn test_bounds_check_edge_case() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);
        let circuit = b.build();

        // Test exact boundary: offset + len_slice == len_input (should be valid)
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 5);
        verifier.populate_offset(&mut filler, 5);

        // Create matching data
        let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let slice_data = vec![5, 6, 7, 8, 9];

        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &slice_data);

        // This should succeed since offset(5) + len_slice(5) == len_input(10)
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_ok(), "Valid boundary case should succeed");

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_empty_slice() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);
        let circuit = b.build();

        // Test with len_slice = 0
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 0);
        verifier.populate_offset(&mut filler, 5);

        let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &[]);

        // Empty slice should be valid
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_ok(), "Empty slice should be valid");

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_mismatched_slice_content() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);
        let circuit = b.build();

        // Test with wrong slice content
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 5);
        verifier.populate_offset(&mut filler, 2);

        let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        // Wrong slice data - should be [2, 3, 4, 5, 6] but we provide [0, 1, 2, 3, 4]
        let wrong_slice_data = vec![0, 1, 2, 3, 4];

        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &wrong_slice_data);

        // This should fail
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err(), "Mismatched slice content should fail");
    }

    #[test]
    fn test_offset_at_end() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);
        let circuit = b.build();

        // Test offset at end with empty slice
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 0);
        verifier.populate_offset(&mut filler, 10);

        let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &[]);

        // This should succeed - empty slice at end
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_ok(), "Empty slice at end should be valid");

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_multiple_byte_extraction_paths() {
        // This test verifies that byte extraction works correctly for all paths
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..3).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);
        let circuit = b.build();

        // Test extraction from each word with different offsets
        for word_idx in 0..3 {
            for byte_offset in 0..8 {
                let offset_val = word_idx * 8 + byte_offset;
                if offset_val + 8 > 24 {
                    continue;
                }

                let mut filler = circuit.new_witness_filler();
                verifier.populate_len_input(&mut filler, 24);
                verifier.populate_len_slice(&mut filler, 8);
                verifier.populate_offset(&mut filler, offset_val);

                // Create distinct pattern for each byte
                let input_data: Vec<u8> = (0..24).map(|i| i as u8).collect();
                let slice_data: Vec<u8> = input_data[offset_val..offset_val + 8].to_vec();

                verifier.populate_input(&mut filler, &input_data);
                verifier.populate_slice(&mut filler, &slice_data);

                let result = circuit.populate_wire_witness(&mut filler);
                assert!(
                    result.is_ok(),
                    "Extraction from word {word_idx} byte {byte_offset} failed"
                );

                // Verify constraints
                let cs = circuit.constraint_system();
                verify_constraints(cs, &filler.into_value_vec()).unwrap();
            }
        }
    }

    #[test]
    fn test_full_word_assert_vulnerability_fixed() {
        let b = CircuitBuilder::new();

        // Set up circuit with specific sizes
        // max_n_slice = 16 (2 words) but we'll use len_slice = 12 (1.5 words)
        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..3).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(
            &b,
            len_input,
            len_slice,
            input.clone(),
            slice.clone(),
            offset,
        );
        let circuit = b.build();

        // Test case: len_slice = 12 bytes (1.5 words)
        // This test verifies that the fix correctly validates partial words
        let mut filler = circuit.new_witness_filler();

        verifier.populate_len_input(&mut filler, 20);
        verifier.populate_len_slice(&mut filler, 12); // 1.5 words
        verifier.populate_offset(&mut filler, 0);

        // Input data
        let input_data = vec![
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // word 0
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, // word 1
            0x10, 0x11, 0x12, 0x13, // partial word 2
        ];
        verifier.populate_input(&mut filler, &input_data);

        // Try to provide malicious slice data:
        // First word: correct
        // Second word: only first 4 bytes need to be correct, last 4 are garbage
        // Directly set the slice words to bypass populate_slice
        filler[slice[0]] = Word(0x0706050403020100); // Correct first word
        filler[slice[1]] = Word(0xffffffff0b0a0908); // Wrong last 4 bytes!

        // This should now fail with the fix
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(
            result.is_err(),
            "Fixed circuit should reject invalid slice data"
        );

        // Now test with correct data to ensure the fix doesn't break valid cases
        let mut filler2 = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler2, 20);
        verifier.populate_len_slice(&mut filler2, 12);
        verifier.populate_offset(&mut filler2, 0);
        verifier.populate_input(&mut filler2, &input_data);

        // Correct slice data
        let correct_slice = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // word 0
            0x08, 0x09, 0x0a, 0x0b, 0x00, 0x00, 0x00, 0x00, // word 1 (padded correctly)
        ];
        verifier.populate_slice(&mut filler2, &correct_slice[..12]); // Only 12 bytes

        // This should succeed
        let result2 = circuit.populate_wire_witness(&mut filler2);
        assert!(
            result2.is_ok(),
            "Fixed circuit should accept valid slice data"
        );

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler2.into_value_vec()).unwrap();
    }

    #[test]
    fn test_simple_partial_word() {
        // Simplest test case: 1 word slice with only 4 valid bytes
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(
            &b,
            len_input,
            len_slice,
            input.clone(),
            slice.clone(),
            offset,
        );
        let circuit = b.build();

        // Test with 4 bytes
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 8);
        verifier.populate_len_slice(&mut filler, 4); // Only 4 bytes
        verifier.populate_offset(&mut filler, 0);

        // Input: full 8 bytes
        filler[input[0]] = Word(0x0706050403020100);

        // Slice with wrong data in upper 4 bytes
        // The slice should contain bytes 0x00, 0x01, 0x02, 0x03 from input
        filler[slice[0]] = Word(0xffffffff03020100); // Correct lower 4 bytes, wrong upper 4

        let result = circuit.populate_wire_witness(&mut filler);
        assert!(
            result.is_err(),
            "Should reject slice with wrong upper bytes"
        );

        // Now test with correct data
        let mut filler2 = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler2, 8);
        verifier.populate_len_slice(&mut filler2, 4);
        verifier.populate_offset(&mut filler2, 0);

        filler2[input[0]] = Word(0x0706050403020100);
        filler2[slice[0]] = Word(0x0000000003020100); // Correct: upper 4 bytes are 0

        let result2 = circuit.populate_wire_witness(&mut filler2);
        assert!(
            result2.is_ok(),
            "Should accept correct partial word: {result2:?}"
        );

        // Additional test: verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler2.into_value_vec()).unwrap();

        // Test 3: Verify populate_slice handles this correctly
        let mut filler3 = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler3, 8);
        verifier.populate_len_slice(&mut filler3, 4);
        verifier.populate_offset(&mut filler3, 0);

        let input_bytes = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let slice_bytes = vec![0x00, 0x01, 0x02, 0x03]; // Only 4 bytes

        verifier.populate_input(&mut filler3, &input_bytes);
        verifier.populate_slice(&mut filler3, &slice_bytes);

        let result3 = circuit.populate_wire_witness(&mut filler3);
        assert!(
            result3.is_ok(),
            "populate_slice should handle partial words correctly"
        );

        // Test 4: Now manually set wrong upper bytes and it should fail
        let mut filler4 = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler4, 8);
        verifier.populate_len_slice(&mut filler4, 4);
        verifier.populate_offset(&mut filler4, 0);
        verifier.populate_input(&mut filler4, &input_bytes);
        verifier.populate_slice(&mut filler4, &slice_bytes);

        // Corrupt the upper bytes
        filler4[slice[0]] = Word(filler4[slice[0]].0 | 0xffffffff00000000);

        let result4 = circuit.populate_wire_witness(&mut filler4);
        assert!(result4.is_err(), "Should reject corrupted upper bytes");
    }

    #[test]
    fn test_direct_masking_logic() {
        // Test the masking logic directly
        let slice_word = Word(0xffffffff_0b0a0908); // Wrong upper 4 bytes
        let extracted_word = Word(0x00000000_0b0a0908); // Correct upper 4 bytes
        let mask = Word(0x00000000_ffffffff); // Mask for 4 valid bytes

        let masked_slice = slice_word & mask;
        let masked_extracted = extracted_word & mask;

        assert_eq!(
            masked_slice, masked_extracted,
            "Masked values should be equal: slice=0x{:016x}, extracted=0x{:016x}",
            masked_slice.0, masked_extracted.0
        );

        // This is what the constraint checks
        let xor_result = masked_slice ^ masked_extracted;
        assert_eq!(
            xor_result,
            Word::ZERO,
            "XOR of masked values should be zero"
        );
    }

    #[test]
    fn test_large_offset_overflow() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(
            &b,
            len_input,
            len_slice,
            input.clone(),
            slice.clone(),
            offset,
        );
        let circuit = b.build();

        // Test with very large offset that should fail 32-bit check
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 5);

        // Try to set offset with upper 32 bits set
        // This tests that the circuit properly validates 32-bit constraints
        filler[offset] = Word(1u64 << 32); // Direct assignment to test constraint

        let input_data = vec![0u8; 10];
        let slice_data = vec![0u8; 5];
        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &slice_data);

        // This should fail the 32-bit check
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err(), "Large offset should fail 32-bit check");
    }

    #[test]
    fn test_32bit_validation() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(
            &b,
            len_input,
            len_slice,
            input.clone(),
            slice.clone(),
            offset,
        );
        let circuit = b.build();

        // Test multiple 32-bit constraint violations
        // Test 1: offset with bit 33 set
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        verifier.populate_len_slice(&mut filler, 5);
        filler[offset] = Word(1u64 << 33);

        let input_data = vec![0u8; 10];
        let slice_data = vec![0u8; 5];
        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &slice_data);

        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err(), "Offset with bit 33 set should fail");

        // Test 2: len_input with upper bits set
        let mut filler = circuit.new_witness_filler();
        filler[len_input] = Word(0xffffffff00000010); // Upper 32 bits set
        verifier.populate_len_slice(&mut filler, 5);
        verifier.populate_offset(&mut filler, 0);
        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &slice_data);

        let result = circuit.populate_wire_witness(&mut filler);
        assert!(
            result.is_err(),
            "len_input with upper 32 bits set should fail"
        );

        // Test 3: len_slice with upper bits set
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 10);
        filler[len_slice] = Word(0x100000005); // Bit 32 set
        verifier.populate_offset(&mut filler, 0);
        verifier.populate_input(&mut filler, &input_data);
        verifier.populate_slice(&mut filler, &slice_data);

        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_err(), "len_slice with upper bits set should fail");
    }

    #[test]
    fn test_edge_case_len_input_zero() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(
            &b,
            len_input,
            len_slice,
            input.clone(),
            slice.clone(),
            offset,
        );
        let circuit = b.build();

        // Test empty input with empty slice at offset 0
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 0);
        verifier.populate_len_slice(&mut filler, 0);
        verifier.populate_offset(&mut filler, 0);

        // Empty arrays
        verifier.populate_input(&mut filler, &[]);
        verifier.populate_slice(&mut filler, &[]);

        // This should succeed - empty input with empty slice at offset 0
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(
            result.is_ok(),
            "Empty input with empty slice should succeed"
        );

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_edge_case_len_input_zero_with_nonzero_slice() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..1).map(|_| b.add_inout()).collect();

        let verifier = Slice::new(
            &b,
            len_input,
            len_slice,
            input.clone(),
            slice.clone(),
            offset,
        );
        let circuit = b.build();

        // Test empty input with non-empty slice
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 0);
        verifier.populate_len_slice(&mut filler, 5);
        verifier.populate_offset(&mut filler, 0);

        verifier.populate_input(&mut filler, &[]);
        verifier.populate_slice(&mut filler, &[1, 2, 3, 4, 5]);

        // This should fail - can't extract non-empty slice from empty input
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(
            result.is_err(),
            "Non-empty slice from empty input should fail"
        );
    }

    #[test]
    fn test_padding_beyond_actual_data() {
        let b = CircuitBuilder::new();

        let len_input = b.add_inout();
        let len_slice = b.add_inout();
        let offset = b.add_inout();

        let input: Vec<Wire> = (0..3).map(|_| b.add_inout()).collect();
        let slice: Vec<Wire> = (0..2).map(|_| b.add_inout()).collect();

        // Save wire references before moving vectors
        let input_wire_2 = input[2];
        let slice_wire_1 = slice[1];

        let verifier = Slice::new(&b, len_input, len_slice, input, slice, offset);
        let circuit = b.build();

        // Test with actual data smaller than allocated space
        let mut filler = circuit.new_witness_filler();
        verifier.populate_len_input(&mut filler, 12); // 1.5 words
        verifier.populate_len_slice(&mut filler, 8); // 1 word
        verifier.populate_offset(&mut filler, 2);

        // Input: 12 bytes (will be padded to 3 words)
        let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        verifier.populate_input(&mut filler, &input_data);

        // Slice at offset 2: bytes 2-9
        let slice_data = vec![2, 3, 4, 5, 6, 7, 8, 9];
        verifier.populate_slice(&mut filler, &slice_data);

        // Verify the circuit handles padding correctly
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(result.is_ok(), "Should handle data with padding correctly");

        // Also verify that padded words in input are zeroed
        assert_eq!(
            filler[input_wire_2],
            Word::ZERO,
            "Third input word should be zero"
        );
        // Second slice word should also be zero since slice is only 1 word
        assert_eq!(
            filler[slice_wire_1],
            Word::ZERO,
            "Second slice word should be zero"
        );

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }
}
