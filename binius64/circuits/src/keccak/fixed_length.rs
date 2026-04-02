// Copyright 2025 Irreducible Inc.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::{
    N_WORDS_PER_BLOCK, N_WORDS_PER_DIGEST, N_WORDS_PER_STATE, RATE_BYTES, permutation::Permutation,
};

/// Computes the Keccak-256 hash of a fixed-length message.
///
/// This function creates a circuit that computes the Keccak-256 digest of a message
/// with a known, fixed length at circuit construction time. It's more efficient than
/// the variable-length version as it doesn't need runtime length checks or multiplexing.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `message` - Input message as packed 64-bit words (8 bytes per wire)
/// * `len_bytes` - The exact length of the message in bytes
///
/// # Returns
/// * `[Wire; 4]` - The Keccak-256 digest as 4 wires of 64 bits each
///
/// # Panics
/// * If `message.len()` does not equal exactly `len_bytes.div_ceil(8)`
///
/// # Example
/// ```rust,ignore
/// use binius_circuits::keccak::fixed_length::keccak256;
/// use binius_frontend::CircuitBuilder;
///
/// let builder = CircuitBuilder::new();
///
/// // Create input wires for a 32-byte message
/// let message: Vec<_> = (0..4).map(|_| builder.add_witness()).collect();
///
/// // Compute Keccak-256 of the 32-byte message
/// let digest = keccak256(&builder, &message, 32);
/// ```
pub fn keccak256(
    builder: &CircuitBuilder,
    message: &[Wire],
    len_bytes: usize,
) -> [Wire; N_WORDS_PER_DIGEST] {
    // Validate that message.len() equals exactly len_bytes.div_ceil(8)
    assert_eq!(
        message.len(),
        len_bytes.div_ceil(8),
        "message.len() ({}) must equal len_bytes.div_ceil(8) ({})",
        message.len(),
        len_bytes.div_ceil(8)
    );

    // Calculate number of blocks needed
    let n_blocks = (len_bytes + 1).div_ceil(RATE_BYTES);
    let n_padded_words = n_blocks * N_WORDS_PER_BLOCK;

    // Create padded message
    let mut padded_message = Vec::with_capacity(n_padded_words);

    // Apply Keccak padding within the circuit
    // The padding consists of 0x01 byte after the message and 0x80 in the final byte of the block
    if len_bytes.is_multiple_of(8) {
        // Message ends on a word boundary - all words are complete
        padded_message.extend_from_slice(message);
        // The 0x01 byte goes at the start of the next word
        padded_message.push(builder.add_constant(Word(0x01)));
    } else {
        // Message ends mid-word - need to handle boundary word
        padded_message.extend_from_slice(&message[..message.len() - 1]);

        // Handle the last message word which is partial
        let last_idx = message.len() - 1;
        let byte_in_word = len_bytes % 8;

        // Mask out the invalid bytes from the original word
        // Create a mask with 1s for valid bytes and 0s for invalid bytes
        let mask = (1u64 << (byte_in_word * 8)) - 1;
        let masked_word = builder.band(message[last_idx], builder.add_constant(Word(mask)));

        // Add 0x01 padding byte right after the valid bytes
        let padding_bit = 1u64 << (byte_in_word * 8);
        let boundary_word = builder.bxor(masked_word, builder.add_constant(Word(padding_bit)));
        padded_message.push(boundary_word);
    }

    // Fill with zeros to complete the padded message
    let zero = builder.add_constant(Word::ZERO);
    padded_message.resize(n_padded_words, zero);

    // XOR 0x80 into the last byte of the last block
    // This correctly handles the case where 0x01 is already in that position
    let last_byte_mask = 0x80u64 << 56; // 0x80 in the most significant byte
    let last_idx = n_padded_words - 1;
    padded_message[last_idx] = builder.bxor(
        padded_message[last_idx],
        builder.add_constant(Word(last_byte_mask)),
    );

    // Initialize state to zeros
    let zero = builder.add_constant(Word::ZERO);
    let mut state = [zero; N_WORDS_PER_STATE];

    // Process each block
    for block in padded_message.chunks(N_WORDS_PER_BLOCK) {
        // XOR the block into the state (first N_WORDS_PER_BLOCK words)
        for (i, &word) in block.iter().enumerate() {
            state[i] = builder.bxor(state[i], word);
        }

        // Apply Keccak-f[1600] permutation
        Permutation::keccak_f1600(builder, &mut state);
    }

    // Return the first 4 words (256 bits) of the state as the digest
    [state[0], state[1], state[2], state[3]]
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::CircuitBuilder;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use rstest::rstest;
    use sha3::{Digest, Keccak256};

    use super::*;

    #[rstest]
    #[case(0)] // Empty message
    #[case(1)] // Single byte
    #[case(8)] // Exactly one word
    #[case(135)] // One byte before block boundary
    #[case(136)] // Exactly one block
    #[case(137)] // One byte over block boundary
    #[case(272)] // Exactly two blocks
    #[case(500)] // Arbitrary larger message
    fn test_keccak256_fixed(#[case] message_len_bytes: usize) {
        // Create test message with deterministic random bytes
        let seed = message_len_bytes as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut message = vec![0u8; message_len_bytes];
        rng.fill_bytes(&mut message);

        // Compute expected digest using sha3 crate
        let mut hasher = Keccak256::new();
        hasher.update(&message);
        let expected_digest: [u8; 32] = hasher.finalize().into();

        // Build circuit
        let builder = CircuitBuilder::new();

        // Create message wires
        let n_words = message_len_bytes.div_ceil(8);
        let message_wires: Vec<_> = (0..n_words).map(|_| builder.add_witness()).collect();

        // Create expected digest wires
        let expected_digest_wires: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());

        // Compute digest using fixed-length function
        let computed_digest = keccak256(&builder, &message_wires, message_len_bytes);

        // Assert computed digest equals expected
        for i in 0..4 {
            builder.assert_eq(
                format!("digest[{i}]"),
                computed_digest[i],
                expected_digest_wires[i],
            );
        }

        // Build and verify circuit
        let circuit = builder.build();
        let cs = circuit.constraint_system();
        let mut witness = circuit.new_witness_filler();

        // Populate message witness
        for (i, chunk) in message.chunks(8).enumerate() {
            let mut word_bytes = [0u8; 8];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            let word = u64::from_le_bytes(word_bytes);
            witness[message_wires[i]] = Word(word);
        }

        // Populate expected digest witness
        for (i, chunk) in expected_digest.chunks(8).enumerate() {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            witness[expected_digest_wires[i]] = Word(word);
        }

        circuit.populate_wire_witness(&mut witness).unwrap();
        verify_constraints(cs, &witness.into_value_vec())
            .expect("Circuit constraints should be satisfied");
    }

    #[test]
    #[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(8) (2)")]
    fn test_keccak256_fixed_wrong_wire_count() {
        let builder = CircuitBuilder::new();

        // Create only 1 wire but claim message is 10 bytes (needs 2 wires)
        let message_wires = vec![builder.add_witness()];

        // This should panic
        keccak256(&builder, &message_wires, 10);
    }

    #[test]
    fn test_keccak256_fixed_exact_wire_count() {
        let builder = CircuitBuilder::new();

        // Empty message: 0 bytes requires 0 wires
        let empty: Vec<Wire> = vec![];
        let _ = keccak256(&builder, &empty, 0);

        // 8 bytes requires exactly 1 wire
        let one_wire = vec![builder.add_witness()];
        let _ = keccak256(&builder, &one_wire, 8);

        // 10 bytes requires exactly 2 wires
        let two_wires = vec![builder.add_witness(), builder.add_witness()];
        let _ = keccak256(&builder, &two_wires, 10);

        // 17 bytes requires exactly 3 wires
        let three_wires = vec![
            builder.add_witness(),
            builder.add_witness(),
            builder.add_witness(),
        ];
        let _ = keccak256(&builder, &three_wires, 17);
    }
}
