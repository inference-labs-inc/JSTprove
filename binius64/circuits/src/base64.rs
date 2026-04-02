// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller, util::pack_bytes_into_wires_le};

use crate::multiplexer::single_wire_multiplex;

/// Base64 encoding (URL-safe, without trailing padding characters) encoding verification.
///
/// Verifies that encoded data is a valid base64 URL-safe encoding (without
/// trailing padding characters) of decoded data.
///
/// This encoding is defined in the JSON Web Signature (JWS) spec:
/// <https://datatracker.ietf.org/doc/html/rfc7515#section-2> (Base64url Encoding)
///
/// # Base64 URL-Safe Alphabet (RFC 4648 §5)
///
/// - Characters 0-61: Same as standard base64 (A-Z, a-z, 0-9)
/// - Character 62: '-' (minus) instead of '+'
/// - Character 63: '_' (underscore) instead of '/'
///
/// # Circuit Behavior
///
/// The circuit performs the following validations:
/// - encoded is valid base64 URL-safe encoding of decoded
/// - len_decoded is the actual length of data in decoded (in bytes)
/// - len_decoded ≤ max_len_decoded (compile-time maximum)
///
/// # Input Packing
///
/// - decoded: Pack 8 bytes per 64-bit word in little-endian format
/// - encoded: Pack 8 base64 characters per 64-bit word in little-endian format
/// - len_decoded: Single 64-bit word containing byte count
pub struct Base64UrlSafe {
    /// Decoded data array (packed 8 bytes per word).
    pub decoded: Vec<Wire>,
    /// Encoded base64 array (packed 8 chars per word).
    pub encoded: Vec<Wire>,
    /// Actual length of decoded data in bytes.
    pub len_bytes: Wire,
}

impl Base64UrlSafe {
    /// Creates a new Base64UrlSafe verifier.
    ///
    /// # Arguments
    ///
    /// * `builder` - Circuit builder for constructing constraints
    /// * `decoded` - raw byte array wires
    /// * `encoded` - Base64 encoded array wires
    /// * `len_bytes` - Wire containing actual length of raw data in bytes
    ///
    /// # Panics
    ///
    /// * If `decoded.len()` is not a multiple of 3
    ///
    /// # Implementation Notes
    ///
    /// The requirement that `decoded.len()` be a multiple of 3 ensures:
    /// - Word alignment: divisible by 8 for packing bytes into 64-bit words
    /// - Base64 group alignment: divisible by 3 for processing complete groups
    /// - Exact array sizing with no rounding needed
    pub fn new(
        builder: &CircuitBuilder,
        decoded: Vec<Wire>,
        encoded: Vec<Wire>,
        len_bytes: Wire,
    ) -> Self {
        // Verify length bounds
        verify_length_bounds(builder, len_bytes, decoded.len() << 3);

        // Process groups of 3 bytes -> 4 base64 chars
        let groups = (decoded.len() << 3).div_ceil(3); // how many 3-byte chunks are there?

        for group_idx in 0..groups {
            let b = builder.subcircuit(format!("group[{group_idx}]"));
            verify_base64_group(&b, &decoded, &encoded, len_bytes, group_idx);
        }

        Self {
            decoded,
            encoded,
            len_bytes,
        }
    }

    /// Populates the length wire with the actual decoded data length.
    ///
    /// # Arguments
    ///
    /// * `w` - Witness filler to populate
    /// * `length` - Actual length of decoded data in bytes
    pub fn populate_len_bytes(&self, w: &mut WitnessFiller, len_bytes: usize) {
        w[self.len_bytes] = Word(len_bytes as u64);
    }

    /// Populates the decoded data array from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `w` - Witness filler to populate
    /// * `data` - Decoded bytes
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` exceeds the maximum size specified during construction.
    pub fn populate_decoded(&self, w: &mut WitnessFiller, data: &[u8]) {
        pack_bytes_into_wires_le(w, &self.decoded, data);
    }

    /// Populates the encoded base64 array from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `w` - Witness filler to populate
    /// * `data` - Base64-encoded bytes
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` exceeds the maximum size specified during construction.
    pub fn populate_encoded(&self, w: &mut WitnessFiller, data: &[u8]) {
        pack_bytes_into_wires_le(w, &self.encoded, data);
    }
}

/// Verifies that the length is within bounds (0 < len_decoded <= max_len_decoded).
fn verify_length_bounds(builder: &CircuitBuilder, len_bytes: Wire, max_len_bytes: usize) {
    // Check if len_decoded > max_len_decoded (which should be false)
    let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
    builder.assert_false("length_check", too_long);
}

/// Verifies a single base64 group (3 decoded bytes -> 4 base64 chars).
///
/// # Base64 Encoding Rules
///
/// Three bytes: AAAAAAAA BBBBBBBB CCCCCCCC
/// Become four 6-bit values:
/// - val0 = AAAAAA (top 6 bits of byte0)
/// - val1 = AABBBB (bottom 2 bits of byte0 + top 4 bits of byte1)
/// - val2 = BBBBCC (bottom 4 bits of byte1 + top 2 bits of byte2)
/// - val3 = CCCCCC (bottom 6 bits of byte2)
fn verify_base64_group(
    builder: &CircuitBuilder,
    decoded: &[Wire],
    encoded: &[Wire],
    len_bytes: Wire,
    group_idx: usize,
) {
    let char_lookup_table = build_base64_char_lookup_table(builder);
    let base_byte_idx = group_idx * 3;
    let base_char_idx = group_idx * 4;

    // Extract 3 decoded bytes
    let byte0 = extract_byte(builder, decoded, base_byte_idx);
    let byte1 = extract_byte(builder, decoded, base_byte_idx + 1);
    let byte2 = extract_byte(builder, decoded, base_byte_idx + 2);

    let has_1 = builder.icmp_ult(builder.add_constant_64(base_byte_idx as u64), len_bytes);
    let has_2 = builder.icmp_ult(
        builder.add_constant_64((base_byte_idx + 1) as u64),
        len_bytes,
    );
    let has_3 = builder.icmp_ult(
        builder.add_constant_64((base_byte_idx + 2) as u64),
        len_bytes,
    );

    let zero = builder.add_constant(Word::ZERO);
    builder.assert_eq_cond(
        "past boundary should be empty",
        byte0,
        zero,
        builder.bnot(has_1),
    );
    builder.assert_eq_cond(
        "past boundary should be empty",
        byte1,
        zero,
        builder.bnot(has_2),
    );
    builder.assert_eq_cond(
        "past boundary should be empty",
        byte2,
        zero,
        builder.bnot(has_3),
    );

    // Convert 3 bytes to 4 6-bit values
    let val0 = extract_6bit_value_0(builder, byte0);
    let val1 = extract_6bit_value_1(builder, byte0, byte1);
    let val2 = extract_6bit_value_2(builder, byte1, byte2);
    let val3 = extract_6bit_value_3(builder, byte2);

    // Convert 6-bit values to base64 encoded chars
    let expected_char0 = compute_expected_base64_char(builder, val0, &char_lookup_table);
    let expected_char1 = compute_expected_base64_char(builder, val1, &char_lookup_table);
    let expected_char2 = compute_expected_base64_char(builder, val2, &char_lookup_table);
    let expected_char3 = compute_expected_base64_char(builder, val3, &char_lookup_table);

    // Extract 4 base64 characters
    let actual_char0 = extract_byte(builder, encoded, base_char_idx);
    let actual_char1 = extract_byte(builder, encoded, base_char_idx + 1);
    let actual_char2 = extract_byte(builder, encoded, base_char_idx + 2);
    let actual_char3 = extract_byte(builder, encoded, base_char_idx + 3);

    verify_base64_char(builder, expected_char0, actual_char0, has_1);
    verify_base64_char(builder, expected_char1, actual_char1, has_1);
    verify_base64_char(builder, expected_char2, actual_char2, has_2);
    verify_base64_char(builder, expected_char3, actual_char3, has_3);
}

/// Extracts a byte from a word array at the given byte index.
///
/// # Arguments
///
/// * `builder` - Circuit builder
/// * `words` - Array of 64-bit words, each containing 8 packed bytes in little-endian format
/// * `byte_idx` - Global byte index to extract
///
/// # Returns
///
/// Wire containing the extracted byte value (0-255), or 0 if out of bounds.
fn extract_byte(builder: &CircuitBuilder, words: &[Wire], byte_idx: usize) -> Wire {
    let word_idx = byte_idx / 8;
    let byte_offset = byte_idx % 8;

    let zero = builder.add_constant(Word::ZERO);
    let word = words.get(word_idx).copied().unwrap_or(zero);
    builder.extract_byte(word, byte_offset as u32)
}

/// Extracts the first 6-bit value (top 6 bits of byte0).
fn extract_6bit_value_0(builder: &CircuitBuilder, byte0: Wire) -> Wire {
    builder.shr(byte0, 2)
}

/// Extracts the second 6-bit value (bottom 2 bits of byte0 + top 4 bits of byte1).
fn extract_6bit_value_1(builder: &CircuitBuilder, byte0: Wire, byte1: Wire) -> Wire {
    let byte0_low = builder.band(byte0, builder.add_constant_64(0x03));
    builder.bxor(builder.shl(byte0_low, 4), builder.shr(byte1, 4))
}

/// Extracts the third 6-bit value (bottom 4 bits of byte1 + top 2 bits of byte2).
fn extract_6bit_value_2(builder: &CircuitBuilder, byte1: Wire, byte2: Wire) -> Wire {
    let byte1_low = builder.band(byte1, builder.add_constant_64(0x0F));
    builder.bxor(builder.shl(byte1_low, 2), builder.shr(byte2, 6))
}

/// Extracts the fourth 6-bit value (bottom 6 bits of byte2).
fn extract_6bit_value_3(builder: &CircuitBuilder, byte2: Wire) -> Wire {
    builder.band(byte2, builder.add_constant_64(0x3F))
}

/// Verifies that a base64 character matches the expected encoding.
///
/// # Arguments
///
/// * `builder` - Circuit builder
/// * `expected_encoded_char` - The expected encoding of the character
/// * `char_val` - The actual character value found
/// * `is_active` - Whether this check should be enforced
fn verify_base64_char(
    builder: &CircuitBuilder,
    expected_encoded_char: Wire,
    actual_encoded_char: Wire,
    is_active: Wire,
) {
    builder.assert_eq(
        "base64_char",
        actual_encoded_char,
        builder.select(
            is_active,
            expected_encoded_char,
            builder.add_constant(Word::ZERO),
        ),
    );
}

/// Builds a character lookup table for base64 URL-safe encoding.
///
/// # Base64 URL-Safe Mapping
///
/// - 0-25: 'A'-'Z' (65-90)
/// - 26-51: 'a'-'z' (97-122)
/// - 52-61: '0'-'9' (48-57)
/// - 62: '-' (45) [URL-safe variant]
/// - 63: '_' (95) [URL-safe variant]
fn build_base64_char_lookup_table(builder: &CircuitBuilder) -> Vec<Wire> {
    (0..64u64)
        .map(|i| {
            let char_val = match i {
                0..=25 => b'A' + i as u8,
                26..=51 => b'a' + (i - 26) as u8,
                52..=61 => b'0' + (i - 52) as u8,
                62 => b'-', // URL-safe: minus instead of plus
                63 => b'_', // URL-safe: underscore instead of slash
                _ => unreachable!(),
            };
            builder.add_constant_64(char_val as u64)
        })
        .collect()
}

/// Computes the expected base64 character for a 6-bit value using the provided lookup table.
///
/// # Arguments
///
/// * `builder` - Circuit builder
/// * `six_bit_val` - The 6-bit value to encode (0-63)
/// * `char_lookup_table` - base64 character encoding lookup table
fn compute_expected_base64_char(
    builder: &CircuitBuilder,
    six_bit_val: Wire,
    char_lookup_table: &[Wire],
) -> Wire {
    single_wire_multiplex(builder, char_lookup_table, six_bit_val)
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::CircuitBuilder;

    use super::{Base64UrlSafe, Wire};

    /// Encodes bytes to base64 using URL-safe alphabet without trailing padding
    /// '=" chars.
    fn encode_base64(input: &[u8]) -> Vec<u8> {
        const BASE64_CHARS: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

        let mut output = Vec::new();

        for chunk in input.chunks(3) {
            let b1 = chunk[0];
            let b2 = chunk.get(1).copied().unwrap_or(0);
            let b3 = chunk.get(2).copied().unwrap_or(0);

            let n = ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);

            output.push(BASE64_CHARS[((n >> 18) & 63) as usize]);
            output.push(BASE64_CHARS[((n >> 12) & 63) as usize]);

            if chunk.len() > 1 {
                output.push(BASE64_CHARS[((n >> 6) & 63) as usize]);
            };

            if chunk.len() > 2 {
                output.push(BASE64_CHARS[(n & 63) as usize]);
            };
        }

        output
    }

    /// Helper to create base64 circuit with given max size.
    fn create_base64_circuit(builder: &CircuitBuilder, max_len_decoded: usize) -> Base64UrlSafe {
        // Create input wires
        assert!(
            max_len_decoded.is_multiple_of(3),
            "max_len_decoded must be a multiple of 3, got {max_len_decoded}"
        );
        let decoded: Vec<Wire> = (0..max_len_decoded).map(|_| builder.add_inout()).collect();
        let max_len_encoded = (max_len_decoded / 3) * 4;
        let encoded: Vec<Wire> = (0..max_len_encoded).map(|_| builder.add_inout()).collect();

        let len_bytes = builder.add_inout();

        Base64UrlSafe::new(builder, decoded, encoded, len_bytes)
    }

    /// Core helper that tests base64 encoding verification and returns a Result.
    fn check_base64_encoding(
        input_bytes: &[u8],
        encoded: &[u8],
        max_len_decoded: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let builder = CircuitBuilder::new();
        let circuit = create_base64_circuit(&builder, max_len_decoded);
        let compiled = builder.build();

        // Create witness
        let mut witness = compiled.new_witness_filler();

        circuit.populate_len_bytes(&mut witness, input_bytes.len());
        circuit.populate_decoded(&mut witness, input_bytes);
        circuit.populate_encoded(&mut witness, encoded);

        // Verify circuit
        compiled.populate_wire_witness(&mut witness)?;

        // Verify constraints
        let cs = compiled.constraint_system();
        verify_constraints(cs, &witness.into_value_vec())?;

        Ok(())
    }

    /// Helper to test base64 encoding verification with specified padding mode.
    fn test_base64_encoding(input: &[u8], max_len_decoded: usize) {
        let expected_base64 = encode_base64(input);
        check_base64_encoding(input, &expected_base64, max_len_decoded).unwrap();
    }

    /// Assert that the base64 circuit fails to verify the specified inputs
    fn assert_base64_failure(input: &[u8], encoded: &[u8], max_len_decoded: usize) {
        check_base64_encoding(input, encoded, max_len_decoded).unwrap_err();
    }

    #[test]
    fn test_base64_hello_world() {
        test_base64_encoding(b"Hello World!", 189);
    }

    #[test]
    fn test_base64_empty() {
        test_base64_encoding(b"", 189);
    }

    #[test]
    fn test_base64_long_input() {
        let input =
			b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        test_base64_encoding(input, 189);
    }

    #[test]
    fn test_invalid_base64() {
        let input = b"ABC";
        let invalid_base64 = b"XXXX"; // Invalid base64 for "ABC"
        assert_base64_failure(input, invalid_base64, 15);
    }

    #[test]
    fn test_url_safe_characters() {
        // Test that URL-safe characters - and _ are used instead of + and /
        // Create input that will result in characters 62 and 63 in base64

        // For 111110 (62): we need top 6 bits = 111110
        let input1 = &[0b11111000]; // Top 6 bits = 111110 = 62
        let expected1 = encode_base64(input1);
        assert_eq!(expected1[0], b'-', "Index 62 should map to '-' not '+'");

        // For 111111 (63): we need top 6 bits = 111111
        let input2 = &[0b11111100]; // Top 6 bits = 111111 = 63
        let expected2 = encode_base64(input2);
        assert_eq!(expected2[0], b'_', "Index 63 should map to '_' not '/'");

        // Now test with the circuit
        test_base64_encoding(input1, 15);
        test_base64_encoding(input2, 15);
    }

    #[test]
    #[should_panic(expected = "max_len_decoded must be a multiple of 3")]
    fn test_panic_when_max_len_not_multiple_of_3() {
        // This test verifies that max_len_decoded must be a multiple of 3
        // Testing with max_len_decoded = 13 which is not a multiple of 3
        test_base64_encoding(b"test", 13);
    }

    #[test]
    fn test_encoding_with_padding_rejected() {
        let input = b"A";
        let encoding_with_padding = b"QQ==";
        // encoding with padding should be rejected
        assert_base64_failure(input, encoding_with_padding, 15);
    }
}
