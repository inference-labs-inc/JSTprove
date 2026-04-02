// Copyright 2025 Irreducible Inc.
//! HMAC (Hash-based Message Authentication Code) circuit implementation using SHA-512.
//!
//! This module provides circuit gadgets for computing HMAC-SHA512 on fixed-length messages.
//! HMAC is defined in RFC 2104 and provides message authentication using a shared secret key.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::sha512::sha512_fixed;

/// Computes HMAC-SHA512 on a fixed-length message with a given key.
///
/// Implements the HMAC construction: `H((K ⊕ opad) || H((K ⊕ ipad) || M))`
/// where H is SHA-512, and ipad/opad are the inner and outer padding constants.
///
/// # Arguments
/// * `builder` - The circuit builder to add constraints to
/// * `key` - The secret key as wires (words in big-endian format)
/// * `message` - The message to authenticate as wires (words in big-endian format)
/// * `message_len_bytes` - The actual message length in bytes
///
/// # Returns
/// * Array of 8 wires containing the 512-bit HMAC output
///
/// # Preconditions
/// * Key length must be at most 128 bytes (SHA-512 block size): `key.len() <= 16`
/// * Message length must match wire count: `message.len() == message_len_bytes.div_ceil(8)`
pub fn hmac_sha512_fixed(
    builder: &mut CircuitBuilder,
    key: &[Wire],
    message: &[Wire],
    message_len_bytes: usize,
) -> [Wire; 8] {
    const BLOCK_SIZE_WORDS: usize = 16; // 128 bytes = 16 * 8-byte words
    const IPAD_BYTE: u8 = 0x36;
    const OPAD_BYTE: u8 = 0x5C;

    // Precondition: key must be at most block size
    assert!(
        key.len() <= BLOCK_SIZE_WORDS,
        "Key length {} words exceeds maximum {} words (128 bytes)",
        key.len(),
        BLOCK_SIZE_WORDS
    );

    // Precondition: message wires must match message length
    assert_eq!(
        message.len(),
        message_len_bytes.div_ceil(8),
        "Message wire count {} doesn't match message length {} bytes",
        message.len(),
        message_len_bytes
    );

    // Create padded key (pad with zeros to block size)
    let mut padded_key = Vec::with_capacity(BLOCK_SIZE_WORDS);
    padded_key.extend_from_slice(key);
    let zero = builder.add_constant(Word::ZERO);
    padded_key.resize(BLOCK_SIZE_WORDS, zero);

    // Create ipad and opad constants
    let ipad_word = builder.add_constant(Word(u64::from_le_bytes([IPAD_BYTE; 8])));
    let opad_word = builder.add_constant(Word(u64::from_le_bytes([OPAD_BYTE; 8])));

    // Compute K ⊕ ipad
    let key_xor_ipad = padded_key
        .iter()
        .map(|&key_word| builder.bxor(key_word, ipad_word))
        .collect::<Vec<_>>();

    // Compute K ⊕ opad
    let key_xor_opad = padded_key
        .iter()
        .map(|&key_word| builder.bxor(key_word, opad_word))
        .collect::<Vec<_>>();

    // Inner hash: H((K ⊕ ipad) || M)
    let mut inner_input = Vec::with_capacity(BLOCK_SIZE_WORDS + message.len());
    inner_input.extend_from_slice(&key_xor_ipad);
    inner_input.extend_from_slice(message);
    let inner_len_bytes = 128 + message_len_bytes; // block size + message length
    let inner_hash = sha512_fixed(builder, &inner_input, inner_len_bytes);

    // Outer hash: H((K ⊕ opad) || inner_hash)
    let mut outer_input = Vec::with_capacity(BLOCK_SIZE_WORDS + 8);
    outer_input.extend_from_slice(&key_xor_opad);
    outer_input.extend_from_slice(&inner_hash);
    let outer_len_bytes = 128 + 64; // block size + SHA-512 output size
    sha512_fixed(builder, &outer_input, outer_len_bytes)
}

#[cfg(test)]
mod tests {
    use std::{array, iter::repeat_with};

    use binius_core::{verify::verify_constraints, word::Word};
    use hmac::{Hmac, Mac};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use sha2::Sha512;

    use super::*;

    type HmacSha512 = Hmac<Sha512>;

    #[test]
    fn test_hmac_sha512_random_messages() {
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..10 {
            // Generate random key (up to 128 bytes)
            let key_len = rng.random_range(1..=16) * 8; // Multiple of 8 bytes
            let mut key_bytes = vec![0u8; key_len];
            rng.fill(&mut key_bytes[..]);

            // Generate random message
            let message_len = rng.random_range(0..=256);
            let mut message_bytes = vec![0u8; message_len];
            rng.fill(&mut message_bytes[..]);

            // Compute expected HMAC using hmac crate
            let mut mac = HmacSha512::new_from_slice(&key_bytes).unwrap();
            mac.update(&message_bytes);
            let expected = mac.finalize().into_bytes();

            // Build circuit
            let mut builder = CircuitBuilder::new();

            // Create key witness wires
            let mut key_wires = Vec::new();
            for _ in 0..key_bytes.len().div_ceil(8) {
                key_wires.push(builder.add_witness());
            }

            // Create message witness wires
            let message_wires = repeat_with(|| builder.add_witness())
                .take(message_len.div_ceil(8))
                .collect::<Vec<_>>();

            // Compute HMAC
            let output = hmac_sha512_fixed(
                &mut builder,
                &key_wires,
                &message_wires,
                message_bytes.len(),
            );

            // Create expected output witness wires
            let expected_wires = array::from_fn::<_, 8, _>(|_| builder.add_witness());

            // Assert equality
            for (i, (&computed, &expected)) in output.iter().zip(expected_wires.iter()).enumerate()
            {
                builder.assert_eq(format!("hmac_output_{i}"), computed, expected);
            }

            let circuit = builder.build();
            let mut witness = circuit.new_witness_filler();

            // Populate key witness values
            for (i, chunk) in key_bytes.chunks(8).enumerate() {
                let mut word_bytes = [0u8; 8];
                word_bytes[..chunk.len()].copy_from_slice(chunk);
                let word = u64::from_be_bytes(word_bytes);
                witness[key_wires[i]] = Word(word);
            }

            // Populate message witness values
            for (i, chunk) in message_bytes.chunks(8).enumerate() {
                let mut word_bytes = [0u8; 8];
                word_bytes[..chunk.len()].copy_from_slice(chunk);
                let word = u64::from_be_bytes(word_bytes);
                witness[message_wires[i]] = Word(word);
            }

            // Populate expected output witness values
            for (i, chunk) in expected.chunks(8).enumerate() {
                let word = u64::from_be_bytes(chunk.try_into().unwrap());
                witness[expected_wires[i]] = Word(word);
            }

            circuit
                .populate_wire_witness(&mut witness)
                .expect("Circuit should populate witnesses successfully");

            // Verify constraints
            let cs = circuit.constraint_system();
            verify_constraints(cs, &witness.into_value_vec())
                .expect("Circuit constraints should be satisfied with random data");
        }
    }

    #[test]
    #[should_panic(expected = "Key length 17 words exceeds maximum 16 words")]
    fn test_hmac_key_too_large() {
        let mut builder = CircuitBuilder::new();

        // Create key that's too large (17 words = 136 bytes)
        let mut key_wires = Vec::new();
        for _ in 0..17 {
            key_wires.push(builder.add_constant(Word::ZERO));
        }

        let message = vec![builder.add_constant(Word::ZERO)];

        // This should panic
        hmac_sha512_fixed(&mut builder, &key_wires, &message, 8);
    }

    #[test]
    #[should_panic(expected = "Message wire count 2 doesn't match message length 7 bytes")]
    fn test_hmac_message_wire_mismatch() {
        let mut builder = CircuitBuilder::new();

        let key = vec![builder.add_constant(Word::ZERO)];
        let message = vec![
            builder.add_constant(Word::ZERO),
            builder.add_constant(Word::ZERO),
        ];

        // This should panic - 2 wires but claiming 7 bytes (needs only 1 wire)
        hmac_sha512_fixed(&mut builder, &key, &message, 7);
    }
}
