// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

use std::ops::Range;

use binius_core::{consts::WORD_SIZE_BYTES, word::Word};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller, util::pack_bytes_into_wires_le};

/// A variable-length byte vector with fixed capacity determined at circuit construction time.
///
/// This struct represents a byte vector whose actual length can vary at runtime (stored in
/// `len_bytes`), but whose maximum capacity is fixed and determined by the number of `data` wires
/// allocated.
///
/// ## Capacity Model
/// - Each wire in the `data` vector holds up to 8 bytes packed in little-endian format
/// - The capacity in bytes = `data.len() * 8`
/// - The actual length is stored in the `len_bytes` wire and can be any value from 0 to the
///   capacity
///
/// ## Example
/// ```ignore
/// // Create a ByteVec with capacity for 32 bytes (4 wires)
/// let byte_vec = ByteVec::new_witness(builder, 32);
/// // byte_vec.data.len() == 4 (since 32 / 8 = 4)
/// // Can hold any actual length from 0 to 32 bytes at runtime
/// ```
#[derive(Clone)]
pub struct ByteVec {
    /// The actual length of valid data in bytes (runtime value, can be 0 to capacity).
    pub len_bytes: Wire,
    /// The data wires, each holding up to 8 bytes. The number of wires determines
    /// the capacity: capacity = data.len() * 8.
    pub data: Vec<Wire>,
}

impl ByteVec {
    /// Creates a new fixed byte vector using the given wires and wire
    /// containing the length of the data in bytes.
    pub fn new(data: Vec<Wire>, len_bytes: Wire) -> Self {
        Self { len_bytes, data }
    }

    /// Creates a new fixed byte vector with the given maximum length as inout wires.
    pub fn new_inout(b: &CircuitBuilder, max_len: usize) -> Self {
        let len_bytes = b.add_inout();
        let data = (0..max_len).map(|_| b.add_inout()).collect();
        Self { len_bytes, data }
    }

    /// Creates a new fixed byte vector with the given maximum length as witness wires.
    pub fn new_witness(b: &CircuitBuilder, max_len: usize) -> Self {
        let len_bytes = b.add_inout();
        let data = (0..max_len).map(|_| b.add_witness()).collect();
        Self { len_bytes, data }
    }

    /// Populate the length wire with the actual vector size in bytes.
    pub fn populate_len_bytes(&self, w: &mut WitnessFiller, len_bytes: usize) {
        w[self.len_bytes] = Word(len_bytes as u64);
    }

    /// Populate the [`ByteVec`] with bytes.
    ///
    /// This method packs bytes into 64-bit words using little-endian ordering,
    ///
    /// # Panics
    /// * If bytes.len() exceeds self.max_len
    pub fn populate_bytes_le(&self, w: &mut WitnessFiller, bytes: &[u8]) {
        pack_bytes_into_wires_le(w, &self.data, bytes);
        w[self.len_bytes] = Word(bytes.len() as u64);
    }

    /// Populate the vector's data from a byte slice.
    ///
    /// Packs the bytes into 64-bit words in little-endian order and ensures
    /// any unused words are zeroed out.
    ///
    /// # Panics
    /// Panics if `data_bytes.len()` > `self.max_len_bytes()`
    pub fn populate_data(&self, w: &mut WitnessFiller, data_bytes: &[u8]) {
        assert!(
            data_bytes.len() <= self.max_len_bytes(),
            "vector data length {} exceeds maximum {}",
            data_bytes.len(),
            self.max_len_bytes()
        );

        // Pack bytes into 64-bit words (little-endian)
        for (i, chunk) in data_bytes.chunks(8).enumerate() {
            if i < self.data.len() {
                let mut word = 0u64;
                for (j, &byte) in chunk.iter().enumerate() {
                    word |= (byte as u64) << (j * 8);
                }
                w[self.data[i]] = Word(word);
            }
        }

        // Zero out any remaining words beyond the actual data
        for i in data_bytes.len().div_ceil(8)..self.data.len() {
            w[self.data[i]] = Word::ZERO;
        }
    }

    /// Returns the maximum length of this vector in bytes.
    pub fn max_len_bytes(&self) -> usize {
        self.data.len() * 8
    }

    /// Construct a new [`ByteVec`] by truncating to `num_wires`.
    ///
    /// # Panics
    /// * If num_wires exceeds self.data.len()
    pub fn truncate(&self, b: &CircuitBuilder, num_wires: usize) -> ByteVec {
        assert!(
            num_wires <= self.data.len(),
            "num_wires must be less than self.data.len()"
        );

        let trimmed_wires = self.data[0..num_wires].to_vec();
        let len_bytes = b.add_constant_64((num_wires << 3) as u64);

        ByteVec::new(trimmed_wires, len_bytes)
    }

    /// Extracts a slice at a compile-time constant range.
    ///
    /// This operation is significantly more efficient than the dynamic `Slice` circuit
    /// because the range is known at circuit construction time, allowing for:
    /// - Direct computation of which words are needed (no multiplexers)
    /// - Compile-time shift amounts (no dynamic shift selection)
    /// - Reduced constraint count
    ///
    /// # Arguments
    /// * `b` - Circuit builder
    /// * `range` - Compile-time constant byte range to extract
    ///
    /// # Returns
    /// A new `ByteVec` containing the extracted slice with capacity rounded up to
    /// the next word boundary (8 bytes).
    ///
    /// # Constraints
    /// - Validates at runtime that `range.end <= self.len_bytes`
    /// - If the range is not aligned to 8-byte boundaries, words are shifted appropriately
    /// - Padding bytes beyond the slice length are guaranteed to be zero
    ///
    /// # Panics
    /// * If `range.start > range.end`
    /// * If `range.end > self.max_len_bytes()`
    ///
    /// # Example
    /// ```ignore
    /// // Extract bytes 3-11 from a ByteVec
    /// let slice = byte_vec.slice_const_range(&builder, 3..11);
    /// // slice will have capacity of 16 bytes (2 words) but length of 8 bytes
    /// ```
    pub fn slice_const_range(&self, b: &CircuitBuilder, range: Range<usize>) -> ByteVec {
        assert!(range.start <= range.end, "Invalid range: start > end");
        assert!(
            range.end <= self.max_len_bytes(),
            "Range end {} exceeds maximum capacity {}",
            range.end,
            self.max_len_bytes()
        );

        let slice_len = range.end - range.start;
        let len_bytes = b.add_constant_64(slice_len as u64);

        // Return early if slice is empty
        if slice_len == 0 {
            return ByteVec::new(Vec::new(), len_bytes);
        }

        let start_word_idx = range.start / WORD_SIZE_BYTES;
        // Word index containing the last byte of the sliced data.
        let last_word_index = (range.end - 1) / WORD_SIZE_BYTES;
        let byte_offset = range.start % WORD_SIZE_BYTES;

        // Calculate number of output words (rounded up to word boundary)
        let num_output_words = slice_len.div_ceil(WORD_SIZE_BYTES);

        // Validate that range.end <= self.len_bytes at runtime
        let range_end_const = b.add_constant_64(range.end as u64);
        let valid = b.icmp_ule(range_end_const, self.len_bytes);
        b.assert_true("slice_range_check", valid);

        // Extract words with shifting if needed
        let mut output_words = if byte_offset == 0 {
            // Aligned case: directly copy the words
            self.data[start_word_idx..start_word_idx + num_output_words].to_vec()
        } else {
            // Unaligned case: combine bytes from two adjacent words
            (0..num_output_words)
                .map(|i| {
                    let source_idx = start_word_idx + i;

                    let current_word = self.data[source_idx];
                    // Shift current word right by byte_offset bytes to align
                    let shifted_current = b.shr(current_word, (byte_offset * 8) as u32);

                    if source_idx < last_word_index {
                        let next_word = self.data[source_idx + 1];
                        // Shift next word left to fill in the high bytes
                        let shifted_next = b.shl(next_word, ((8 - byte_offset) * 8) as u32);
                        // Combine the two parts (XOR is cheaper than OR)
                        b.bxor(shifted_current, shifted_next)
                    } else {
                        shifted_current
                    }
                })
                .collect()
        };

        // Apply byte mask to final word if slice doesn't end on word boundary
        let final_byte_count = slice_len % 8;
        if final_byte_count != 0 {
            let last_idx = output_words.len() - 1;
            // Create mask with only the lower final_byte_count bytes set
            let mask_value = (1u64 << (final_byte_count * 8)) - 1;
            let mask = b.add_constant_64(mask_value);
            output_words[last_idx] = b.band(output_words[last_idx], mask);
        }

        // Create new ByteVec with constant length
        ByteVec::new(output_words, len_bytes)
    }
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;

    use super::{ByteVec, CircuitBuilder, Word};

    #[test]
    fn test_slice_const_range_aligned() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 32 bytes capacity (4 words)
        let byte_vec = ByteVec::new_witness(&b, 4);

        // Extract aligned slice: bytes 8-16 (word 1)
        let slice = byte_vec.slice_const_range(&b, 8..16);

        assert_eq!(slice.data.len(), 1, "Slice should have 1 word");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input with known data
        let input_data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Expected slice: bytes 8-15 (word 1)
        let expected_word = 0x0f0e0d0c0b0a0908u64;
        filler[slice.data[0]] = Word(expected_word);

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_slice_const_range_unaligned() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 32 bytes capacity (4 words)
        let byte_vec = ByteVec::new_witness(&b, 4);

        // Extract unaligned slice: bytes 3-11 (spans across words 0 and 1)
        let slice = byte_vec.slice_const_range(&b, 3..11);

        assert_eq!(slice.data.len(), 1, "Slice should have 1 word");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input with known data
        let input_data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Expected slice: bytes 3-10 (8 bytes total)
        // Bytes: 03 04 05 06 07 08 09 0a
        let expected_word = 0x0a09080706050403u64;
        filler[slice.data[0]] = Word(expected_word);

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_slice_const_range_partial_word() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 24 bytes capacity (3 words)
        let byte_vec = ByteVec::new_witness(&b, 3);

        // Extract partial word: bytes 0-5
        let slice = byte_vec.slice_const_range(&b, 0..5);

        assert_eq!(slice.data.len(), 1, "Slice should have 1 word (rounded up)");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input with known data
        let input_data: Vec<u8> = (0..24).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Expected slice: bytes 0-4 (5 bytes), padded to word
        // Bytes: 00 01 02 03 04 (and upper 3 bytes should be masked to 0)
        let expected_word = 0x0000000004030201u64; // Note: byte 0 is LSB
        filler[slice.data[0]] = Word(expected_word);

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_slice_const_range_unaligned_partial() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 24 bytes capacity (3 words)
        let byte_vec = ByteVec::new_witness(&b, 3);

        // Extract unaligned partial word: bytes 3-8 (5 bytes)
        let slice = byte_vec.slice_const_range(&b, 3..8);

        assert_eq!(slice.data.len(), 1, "Slice should have 1 word");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input with known data
        let input_data: Vec<u8> = (0..24).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Expected slice: bytes 3-7 (5 bytes)
        // Bytes: 03 04 05 06 07 (and upper 3 bytes masked to 0)
        let expected_word = 0x0000000007060504u64 | 0x03; // Construct little-endian
        filler[slice.data[0]] = Word(expected_word);

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_slice_const_range_empty() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 16 bytes capacity (2 words)
        let byte_vec = ByteVec::new_witness(&b, 2);

        // Extract empty slice
        let slice = byte_vec.slice_const_range(&b, 5..5);

        assert_eq!(slice.data.len(), 0, "Empty slice should have 0 words");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input
        let input_data: Vec<u8> = (0..16).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_slice_const_range_bounds_check_valid() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 16 bytes capacity (2 words)
        let byte_vec = ByteVec::new_witness(&b, 2);

        // Extract slice up to the full length
        let slice = byte_vec.slice_const_range(&b, 0..16);

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate with exactly 16 bytes
        let input_data: Vec<u8> = (0..16).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Manually populate slice data
        for (i, chunk) in input_data.chunks(8).enumerate() {
            let mut word = 0u64;
            for (j, &byte) in chunk.iter().enumerate() {
                word |= (byte as u64) << (j * 8);
            }
            filler[slice.data[i]] = Word(word);
        }

        // Should succeed: range.end (16) == len_bytes (16)
        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    fn test_slice_const_range_bounds_check_fail() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 16 bytes capacity (2 words)
        let byte_vec = ByteVec::new_witness(&b, 2);

        // Extract slice beyond capacity
        let slice = byte_vec.slice_const_range(&b, 0..16);

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate with only 10 bytes (less than range.end)
        let input_data: Vec<u8> = (0..10).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Manually populate slice data
        for i in 0..2 {
            let start = i * 8;
            let end = (start + 8).min(input_data.len());
            let mut word = 0u64;
            if start < input_data.len() {
                for (j, &byte) in input_data[start..end].iter().enumerate() {
                    word |= (byte as u64) << (j * 8);
                }
            }
            filler[slice.data[i]] = Word(word);
        }

        // Should fail: range.end (16) > len_bytes (10)
        let result = circuit.populate_wire_witness(&mut filler);
        assert!(
            result.is_err(),
            "Should fail bounds check when range.end > len_bytes"
        );
    }

    #[test]
    fn test_slice_const_range_multi_word() {
        let b = CircuitBuilder::new();

        // Create a ByteVec with 32 bytes capacity (4 words)
        let byte_vec = ByteVec::new_witness(&b, 4);

        // Extract multi-word slice: bytes 5-21 (16 bytes, spans 3 source words)
        let slice = byte_vec.slice_const_range(&b, 5..21);

        assert_eq!(slice.data.len(), 2, "Slice should have 2 words");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input with known data
        let input_data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Extract bytes 5-20 manually for verification
        let slice_bytes: Vec<u8> = input_data[5..21].to_vec();

        // Pack into words
        for (i, chunk) in slice_bytes.chunks(8).enumerate() {
            let mut word = 0u64;
            for (j, &byte) in chunk.iter().enumerate() {
                word |= (byte as u64) << (j * 8);
            }
            filler[slice.data[i]] = Word(word);
        }

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Invalid range")]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_slice_const_range_invalid_range() {
        let b = CircuitBuilder::new();
        let byte_vec = ByteVec::new_witness(&b, 4);

        // Should panic: start > end
        byte_vec.slice_const_range(&b, 10..5);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum capacity")]
    fn test_slice_const_range_exceeds_capacity() {
        let b = CircuitBuilder::new();
        let byte_vec = ByteVec::new_witness(&b, 2); // 16 bytes capacity

        // Should panic: range.end > capacity
        byte_vec.slice_const_range(&b, 0..20);
    }

    #[test]
    fn test_slice_const_range_unaligned_at_capacity_boundary() {
        // Test the edge case where we slice to the very end with non-zero byte offset
        // This exercises the next_word bounds check
        let b = CircuitBuilder::new();

        // Create a ByteVec with 16 bytes capacity (2 words)
        let byte_vec = ByteVec::new_witness(&b, 2);

        // Extract unaligned slice all the way to the end: bytes 5-16
        // This requires accessing a "next word" that doesn't exist
        let slice = byte_vec.slice_const_range(&b, 5..16);

        assert_eq!(slice.data.len(), 2, "Slice should have 2 words");

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();

        // Populate input with known data
        let input_data: Vec<u8> = (0..16).map(|i| i as u8).collect();
        byte_vec.populate_bytes_le(&mut filler, &input_data);

        // Expected slice: bytes 5-15 (11 bytes)
        let slice_bytes: Vec<u8> = input_data[5..16].to_vec();

        // Pack into words
        for (i, chunk) in slice_bytes.chunks(8).enumerate() {
            let mut word = 0u64;
            for (j, &byte) in chunk.iter().enumerate() {
                word |= (byte as u64) << (j * 8);
            }
            filler[slice.data[i]] = Word(word);
        }

        circuit.populate_wire_witness(&mut filler).unwrap();

        // Verify constraints
        let cs = circuit.constraint_system();
        verify_constraints(cs, &filler.into_value_vec()).unwrap();
    }
}
