// Copyright 2025 Irreducible Inc.
//! BLAKE2b circuit implementation for Binius64
//!
//! This module implements BLAKE2b as a zero-knowledge circuit using the Binius64
//! constraint system. It follows the RFC 7693 specification.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use super::constants::{BLOCK_BYTES, IV, ROUNDS, SIGMA};

/// BLAKE2b circuit following single-block pattern
/// Processes messages block-by-block like Blake2s
pub struct Blake2bCircuit {
    /// Message size in bytes this circuit supports
    pub length: usize,

    /// Witness wires for the input message (little-endian 64-bit words)
    pub message: Vec<Wire>,

    /// Witness wires for the expected 512-bit digest (8 × 64-bit words)
    pub digest: [Wire; 8],
}

impl Blake2bCircuit {
    /// Create a new BLAKE2b circuit with standard 64-byte output
    pub fn new(builder: &CircuitBuilder) -> Self {
        Self::new_with_length(builder, BLOCK_BYTES) // Default to 1 block
    }

    /// Create a new BLAKE2b circuit for messages up to `max_msg_len_bytes`
    pub fn new_with_length(builder: &CircuitBuilder, max_msg_len_bytes: usize) -> Self {
        Self::new_with_params(builder, max_msg_len_bytes, 64)
    }

    /// Create a new BLAKE2b circuit with specified message length and output length
    pub fn new_with_params(
        builder: &CircuitBuilder,
        max_msg_len_bytes: usize,
        outlen: usize,
    ) -> Self {
        assert!(
            outlen > 0 && outlen <= 64,
            "Output length must be 1-64 bytes"
        );
        // Allow zero-length messages

        // Create witness wires for message (packed as 64-bit words)
        // For empty messages, we still need at least one wire for padding
        let num_message_words = max_msg_len_bytes.div_ceil(8).max(1);
        let message: Vec<Wire> = (0..num_message_words)
            .map(|_| builder.add_witness())
            .collect();

        // Create witness wires for digest
        let digest = std::array::from_fn(|_| builder.add_witness());

        // Build the circuit
        Self::build_circuit(builder, max_msg_len_bytes, &message, digest, outlen);

        Self {
            length: max_msg_len_bytes,
            message,
            digest,
        }
    }

    /// Populate the message data into the witness
    pub fn populate_message(&self, w: &mut WitnessFiller, message: &[u8]) {
        assert!(
            message.len() <= self.length,
            "Message exceeds circuit capacity"
        );

        // Pack message bytes into 64-bit words (little-endian)
        for (i, chunk) in message.chunks(8).enumerate() {
            let mut word_value = 0u64;
            for (j, &byte) in chunk.iter().enumerate() {
                word_value |= (byte as u64) << (j * 8);
            }
            w[self.message[i]] = Word(word_value);
        }

        // Pad remaining message words with zeros
        for i in message.len().div_ceil(8)..self.message.len() {
            w[self.message[i]] = Word(0);
        }
    }

    /// Populate the expected digest output for verification
    pub fn populate_digest(&self, w: &mut WitnessFiller, digest: &[u8; 64]) {
        // Pack digest bytes into 64-bit words (little-endian)
        for i in 0..8 {
            let mut word_value = 0u64;
            for j in 0..8 {
                word_value |= (digest[i * 8 + j] as u64) << (j * 8);
            }
            w[self.digest[i]] = Word(word_value);
        }
    }

    /// Build the BLAKE2b circuit constraints.
    ///
    /// This constructs the circuit that verifies a fixed-length message
    /// produces the expected BLAKE2b digest. The circuit handles:
    ///
    /// 1. Message padding to 128-byte blocks
    /// 2. Sequential block processing (one compression per block)
    /// 3. Proper counter management for multi-block messages
    /// 4. Final block detection and processing
    fn build_circuit(
        builder: &CircuitBuilder,
        length: usize,
        message: &[Wire],
        expected_digest: [Wire; 8],
        outlen: usize,
    ) {
        // Calculate number of blocks needed
        let num_blocks = if length == 0 {
            1
        } else {
            length.div_ceil(BLOCK_BYTES)
        };
        let zero = builder.add_constant(Word::ZERO);

        // Initialize state with IVs XORed with parameter block
        // Parameter block: 0x0101kknn where nn=outlen, kk=keylen (0), fanout=depth=1
        let param_block = 0x01010000 | (outlen as u64);

        let init_state = [
            builder.add_constant(Word(IV[0] ^ param_block)),
            builder.add_constant(Word(IV[1])),
            builder.add_constant(Word(IV[2])),
            builder.add_constant(Word(IV[3])),
            builder.add_constant(Word(IV[4])),
            builder.add_constant(Word(IV[5])),
            builder.add_constant(Word(IV[6])),
            builder.add_constant(Word(IV[7])),
        ];

        let mut h = init_state;
        let mut final_digest = [zero; 8];

        // Process each block sequentially
        for block_idx in 0..num_blocks {
            // Prepare message block with proper padding
            let mut m = [zero; 16];

            // Fill message words from input
            for word_idx in 0..16 {
                let byte_start = block_idx * BLOCK_BYTES + word_idx * 8;

                if byte_start < length {
                    // Get the corresponding 64-bit word from message
                    let msg_word_idx = byte_start / 8;

                    if msg_word_idx < message.len() {
                        let msg_word = message[msg_word_idx];

                        // Handle partial word at message boundary
                        if byte_start + 8 > length {
                            // Need to mask off bytes beyond message length
                            let valid_bytes = length - byte_start;
                            let mask = builder.add_constant(Word((1u64 << (valid_bytes * 8)) - 1));
                            m[word_idx] = builder.band(msg_word, mask);
                        } else {
                            m[word_idx] = msg_word;
                        }
                    }
                }
                // Else m[word_idx] remains zero (padding)
            }

            // Determine if this is the final block
            let is_final_block = block_idx == num_blocks - 1;

            // Set up byte counter
            let t_low = if is_final_block {
                builder.add_constant(Word(length as u64))
            } else {
                builder.add_constant(Word(((block_idx + 1) * BLOCK_BYTES) as u64))
            };
            let t_high = zero; // Always 0 for messages < 2^64 bytes

            // Set finalization flag
            let last_flag = if is_final_block {
                builder.add_constant(Word(0xFFFFFFFFFFFFFFFF))
            } else {
                zero
            };

            // Process the block
            h = compress(builder, &h, &m, t_low, t_high, last_flag);

            // Save as final digest if this is the last block
            if is_final_block {
                final_digest.copy_from_slice(&h);
            }
        }

        // Assert that the computed digest matches the expected digest
        for i in 0..8 {
            builder.assert_eq(
                format!("digest[{}]", i),
                final_digest[i],
                expected_digest[i],
            );
        }
    }
}

/// BLAKE2b compression function - processes a single 128-byte block
fn compress(
    builder: &CircuitBuilder,
    h: &[Wire; 8],
    m: &[Wire; 16],
    t_low: Wire,
    t_high: Wire,
    last_block_flag: Wire,
) -> [Wire; 8] {
    // Initialize working vector
    let mut v = [builder.add_constant(Word::ZERO); 16];

    // v[0..8] = h[0..8]
    v[0..8].copy_from_slice(h);

    // v[8..16] = IV[0..8]
    for i in 0..8 {
        v[i + 8] = builder.add_constant(Word(IV[i]));
    }

    // Mix in counter
    v[12] = builder.bxor(v[12], t_low);
    v[13] = builder.bxor(v[13], t_high);

    // Conditionally invert v[14] for last block
    v[14] = builder.bxor(v[14], last_block_flag);

    // 12 rounds of mixing
    for round in 0..ROUNDS {
        // Column step
        g_mixing(
            builder,
            &mut v,
            0,
            4,
            8,
            12,
            m[SIGMA[round][0]],
            m[SIGMA[round][1]],
        );
        g_mixing(
            builder,
            &mut v,
            1,
            5,
            9,
            13,
            m[SIGMA[round][2]],
            m[SIGMA[round][3]],
        );
        g_mixing(
            builder,
            &mut v,
            2,
            6,
            10,
            14,
            m[SIGMA[round][4]],
            m[SIGMA[round][5]],
        );
        g_mixing(
            builder,
            &mut v,
            3,
            7,
            11,
            15,
            m[SIGMA[round][6]],
            m[SIGMA[round][7]],
        );

        // Diagonal step
        g_mixing(
            builder,
            &mut v,
            0,
            5,
            10,
            15,
            m[SIGMA[round][8]],
            m[SIGMA[round][9]],
        );
        g_mixing(
            builder,
            &mut v,
            1,
            6,
            11,
            12,
            m[SIGMA[round][10]],
            m[SIGMA[round][11]],
        );
        g_mixing(
            builder,
            &mut v,
            2,
            7,
            8,
            13,
            m[SIGMA[round][12]],
            m[SIGMA[round][13]],
        );
        g_mixing(
            builder,
            &mut v,
            3,
            4,
            9,
            14,
            m[SIGMA[round][14]],
            m[SIGMA[round][15]],
        );
    }

    // Finalization: h[i] = h[i] XOR v[i] XOR v[i+8]
    let mut h_new = [builder.add_constant(Word::ZERO); 8];
    for i in 0..8 {
        h_new[i] = builder.bxor_multi(&[h[i], v[i], v[i + 8]]);
    }

    h_new
}

/// BLAKE2b G mixing function
///
/// This implements the core mixing operation:
/// ```text
/// a = a + b + x
/// d = rotr64(d ^ a, 32)
/// c = c + d
/// b = rotr64(b ^ c, 24)
/// a = a + b + y
/// d = rotr64(d ^ a, 16)
/// c = c + d
/// b = rotr64(b ^ c, 63)
/// ```
///
/// Cost: 8 AND constraints (4 additions × 2 constraints each)
#[allow(clippy::too_many_arguments)]
pub fn g_mixing(
    builder: &CircuitBuilder,
    v: &mut [Wire; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    x: Wire,
    y: Wire,
) {
    let zero = builder.add_constant(Word::ZERO);

    // a = a + b + x
    let (temp1, _) = builder.iadd_cin_cout(v[a], v[b], zero);
    let (v_a_new1, _) = builder.iadd_cin_cout(temp1, x, zero);
    v[a] = v_a_new1;

    // d = rotr64(d ^ a, 32)
    let xor1 = builder.bxor(v[d], v[a]);
    v[d] = builder.rotr(xor1, 32);

    // c = c + d
    let (v_c_new1, _) = builder.iadd_cin_cout(v[c], v[d], zero);
    v[c] = v_c_new1;

    // b = rotr64(b ^ c, 24)
    let xor2 = builder.bxor(v[b], v[c]);
    v[b] = builder.rotr(xor2, 24);

    // a = a + b + y
    let (temp2, _) = builder.iadd_cin_cout(v[a], v[b], zero);
    let (v_a_new2, _) = builder.iadd_cin_cout(temp2, y, zero);
    v[a] = v_a_new2;

    // d = rotr64(d ^ a, 16)
    let xor3 = builder.bxor(v[d], v[a]);
    v[d] = builder.rotr(xor3, 16);

    // c = c + d
    let (v_c_new2, _) = builder.iadd_cin_cout(v[c], v[d], zero);
    v[c] = v_c_new2;

    // b = rotr64(b ^ c, 63)
    let xor4 = builder.bxor(v[b], v[c]);
    v[b] = builder.rotr(xor4, 63);
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use binius_frontend::CircuitBuilder;

    use crate::blake2b::{circuit::g_mixing, reference};

    /// Test the G mixing function with known values
    #[test]
    fn test_g_mixing_function() {
        let builder = CircuitBuilder::new();

        let mut v = core::array::from_fn(|_| builder.add_inout());
        let x = builder.add_inout();
        let y = builder.add_inout();

        // Expected outputs
        let expected: [_; 16] = core::array::from_fn(|_| builder.add_inout());

        // Save the initial wires before G mixing
        let v_initial = v;

        // Apply G mixing
        g_mixing(&builder, &mut v, 0, 4, 8, 12, x, y);

        // The g_mixing function has updated v[0], v[4], v[8], v[12] with new wires
        // Assert equality between the new values and expected
        for i in [0, 4, 8, 12] {
            builder.assert_eq(format!("v[{}]", i), v[i], expected[i]);
        }

        let circuit = builder.build();

        // Test with simple values
        let mut w = circuit.new_witness_filler();

        // Initial state (simple test values)
        let initial_v = [
            0x0000000000000001u64, // v[0]
            0x0000000000000002u64, // v[1]
            0x0000000000000003u64, // v[2]
            0x0000000000000004u64, // v[3]
            0x0000000000000005u64, // v[4]
            0x0000000000000006u64, // v[5]
            0x0000000000000007u64, // v[6]
            0x0000000000000008u64, // v[7]
            0x0000000000000009u64, // v[8]
            0x000000000000000Au64, // v[9]
            0x000000000000000Bu64, // v[10]
            0x000000000000000Cu64, // v[11]
            0x000000000000000Du64, // v[12]
            0x000000000000000Eu64, // v[13]
            0x000000000000000Fu64, // v[14]
            0x0000000000000010u64, // v[15]
        ];

        let x_val = 0x123456789ABCDEFu64;
        let y_val = 0xFEDCBA9876543210u64;

        for i in 0..16 {
            w[v_initial[i]] = Word(initial_v[i]);
        }
        w[x] = Word(x_val);
        w[y] = Word(y_val);

        // Run reference G to get expected values
        let mut expected_v = initial_v;
        reference::g(&mut expected_v, 0, 4, 8, 12, x_val, y_val);

        for i in [0, 4, 8, 12] {
            w[expected[i]] = Word(expected_v[i]);
        }

        // Populate witness and verify constraints
        circuit.populate_wire_witness(&mut w).unwrap();

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }
}
