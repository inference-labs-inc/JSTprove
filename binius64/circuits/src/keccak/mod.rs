// Copyright 2025 Irreducible Inc.

pub mod fixed_length;
pub mod permutation;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use permutation::Permutation;

use crate::multiplexer::{multi_wire_multiplex, single_wire_multiplex};

pub const N_WORDS_PER_DIGEST: usize = 4;
pub const N_WORDS_PER_STATE: usize = 25;
pub const RATE_BYTES: usize = 136;
pub const N_WORDS_PER_BLOCK: usize = RATE_BYTES / 8;

/// Keccak-256 circuit that can handle variable-length inputs up to a specified maximum length.
///
/// # Arguments
///
/// * `len_bytes` - A wire representing the input message length in bytes
/// * `digest` - Array of 4 wires representing the 256-bit output digest
/// * `message` - Vector of wires representing the input message
pub struct Keccak256 {
    pub len_bytes: Wire,
    pub digest: [Wire; N_WORDS_PER_DIGEST],
    pub message: Vec<Wire>,
    padded_message: Vec<Wire>,
    n_blocks: usize,
}

impl Keccak256 {
    /// Create a new keccak circuit using the circuit builder
    ///
    /// # Arguments
    ///
    /// * `builder` - circuit builder object
    /// * `max_len` - max message length in bytes for this circuit instance
    /// * `len` - wire representing the claimed input message length in bytes
    /// * `digest` - array of 4 wires representing the claimed 256-bit output digest
    /// * `message` - vector of wires representing the claimed input message
    ///
    /// ## Preconditions
    /// * max_len > 0
    pub fn new(
        b: &CircuitBuilder,
        len_bytes: Wire,
        digest: [Wire; N_WORDS_PER_DIGEST],
        message: Vec<Wire>,
    ) -> Self {
        let max_len_bytes = message.len() << 3;
        // number of blocks needed for the maximum sized message
        let n_blocks = (max_len_bytes + 1).div_ceil(RATE_BYTES);

        // constrain the message length claim to be explicitly within bounds
        let len_check = b.icmp_ugt(len_bytes, b.add_constant_64(max_len_bytes as u64)); // len_bytes > max_len_bytes
        b.assert_false("len_check", len_check);

        let padded_message: Vec<Wire> = (0..n_blocks * N_WORDS_PER_BLOCK)
            .map(|_| b.add_witness())
            .collect();

        // zero initialized keccak state
        let mut states: Vec<[Wire; N_WORDS_PER_STATE]> = Vec::with_capacity(n_blocks + 1);
        let zero = b.add_constant(Word::ZERO);
        states.push([zero; N_WORDS_PER_STATE]);

        // xor next message block into state and permute
        for block_no in 0..n_blocks {
            let state_in = states[block_no];
            let mut xored_state = state_in;
            for i in 0..N_WORDS_PER_BLOCK {
                xored_state[i] = b.bxor(
                    state_in[i],
                    padded_message[block_no * N_WORDS_PER_BLOCK + i],
                );
            }

            Permutation::keccak_f1600(b, &mut xored_state);

            states.push(xored_state);
        }

        // begin "constrain claimed digest".
        // want to do: `let block_index = (len_bytes + 1).divceil(136)`.
        // royal pain in the ass that 136 is not a power of 2, so we can't compute this in circuit
        // still though, i believe that there might be tricks better than what we're doing below.
        let mut end_block_index = b.add_constant(Word::ZERO);
        let mut is_not_last_column = b.add_constant(Word::ZERO);
        // `is_not_last_column` will be true if and only if `len_bytes >> 3` != 16 (mod 17).
        // true iff the WORD w/ the very first post-message byte is NOT the last word in its block.
        for block_no in 0..n_blocks {
            // start of this block
            let block_start = b.add_constant_64((block_no * RATE_BYTES) as u64);
            let block_end = b.add_constant_64(((block_no + 1) * RATE_BYTES) as u64);
            let last_word_start = b.add_constant_64(((block_no + 1) * RATE_BYTES - 8) as u64);

            let gte_start = b.icmp_ule(block_start, len_bytes);
            let lt_end = b.icmp_ult(len_bytes, block_end);
            let lt_last_word = b.icmp_ult(len_bytes, last_word_start);
            let is_final_block = b.band(gte_start, lt_end);

            // flag that this block is the final block per the claimed length
            end_block_index = b.select(
                is_final_block,
                b.add_constant_64(block_no as u64),
                end_block_index,
            );
            is_not_last_column = b.select(is_final_block, lt_last_word, is_not_last_column);
        }

        let inputs: Vec<&[Wire]> = states[1..].iter().map(|arr| &arr[..]).collect();
        let computed_digest_vec = multi_wire_multiplex(b, &inputs, end_block_index);
        let computed_digest = computed_digest_vec[..N_WORDS_PER_DIGEST]
            .try_into()
            .unwrap();
        b.assert_eq_v("claimed digest is correct", digest, computed_digest);

        // begin treatment of boundary word.
        let word_boundary = b.shr(len_bytes, 3);
        let boundary_word = single_wire_multiplex(b, &message, word_boundary);
        let boundary_padded_word = single_wire_multiplex(b, &padded_message, word_boundary);
        // When the last word of the message is not full, we expect a padding byte to be
        // somewhere within the word. Since the top bit will also be in this word.
        let candidates: Vec<Wire> = (0..8)
            .map(|i| {
                let mask = b.add_constant_64(0x00FFFFFFFFFFFFFF >> ((7 - i) << 3));
                let padding_byte = b.add_constant_64(1 << (i << 3));
                let message_low = b.band(boundary_word, mask);
                b.bxor(message_low, padding_byte)
            })
            .collect();

        let zero = b.add_constant(Word::ZERO);
        let msb_one = b.add_constant(Word::MSB_ONE);
        let len_bytes_mod_8 = b.band(len_bytes, b.add_constant_64(7));
        let expected_partial = single_wire_multiplex(b, &candidates, len_bytes_mod_8);
        let with_possible_end = b.bxor(
            expected_partial,
            b.select(is_not_last_column, zero, msb_one),
        );

        b.assert_eq("expected partial", with_possible_end, boundary_padded_word);

        // Within the final rate block, ensure that the pad byte and top bit are where they are
        // supposed to be
        for block_index in 0..n_blocks {
            let is_end_block = b.icmp_eq(end_block_index, b.add_constant_64(block_index as u64));
            for column_index in 0..N_WORDS_PER_BLOCK {
                let word_index = block_index * N_WORDS_PER_BLOCK + column_index;

                let padded_word = padded_message[word_index];

                // a potentially padded word is at this index
                let word_idx_wire = b.add_constant_64(word_index as u64);
                if word_index < message.len() {
                    let message_word = message[word_index];
                    let is_before_end = b.icmp_ult(word_idx_wire, word_boundary);
                    b.assert_eq_cond("full", padded_word, message_word, is_before_end);
                }

                let is_past_message = b.icmp_ugt(word_idx_wire, word_boundary);

                if column_index == 16 {
                    // last word in the block
                    let must_check_delimiter = b.band(is_end_block, is_not_last_column);
                    b.assert_eq_cond("delim", padded_word, msb_one, must_check_delimiter);
                    // the case we need to deal with: we're in end block but `is_not_last_column`.
                    // this means that the `boundary_message_word` is not the last word in its block
                    // then the presence of the 0x80 delimiter is NOT treated with the boundary word
                    // thus we must separately check that the ACTUAL last word in the block has it

                    // if `is_end_block` is true but NOT `is_not_last_column`, then we're fine.
                    // indeed: if `!is_not_last_column`, boundary message word IS in last column,
                    // so we already handled the validity of that word, and there is nothing to do.

                    // if NOT in end block, then again i claim there is nothing we need to check.
                    // if we're in the last column but strictly before the end block, then we're
                    // still in the message, by definition of `end_block`. indeed, the `0x80` byte
                    // happens in the soonest possible block after the message ends, and no later.
                    // thus we already checked the validity of this word above (a `message_word`).
                    // the other case is that we're strictly after the end block. in this case,
                    // we can just leave the `padded_word` completely unconstrained. after all,
                    // said word will have no effect on `digest` whatsoever, so we just leave it.
                } else {
                    b.assert_eq_cond("after-message padding", padded_word, zero, is_past_message);
                    // we're strictly after the word w/ the 0x01 byte and not in the last column.
                    // there are two cases: either we're within the end block or strictly after it.
                    // if the former, we're after the boundary word but before the word w/ 0x80.
                    // in that case, we must for the sake of correctness assert that this word is 0.
                    // if strictly after the end block, this word will have no effect on `digest`;
                    // thus we're free to assert that it's 0, but it's not necessary for soundness.
                }
            }
        }

        Self {
            len_bytes,
            digest,
            message,
            padded_message,
            n_blocks,
        }
    }

    pub fn max_len_bytes(&self) -> usize {
        self.message.len() << 3
    }

    /// Populates the witness with the actual message length
    ///
    /// ## Arguments
    ///
    /// * w - The witness filler to populate
    /// * len_bytes - The actual byte length of the message
    pub fn populate_len_bytes(&self, w: &mut WitnessFiller<'_>, len_bytes: usize) {
        assert!(
            len_bytes <= self.max_len_bytes(),
            "Message length {} exceeds maximum {}",
            len_bytes,
            self.max_len_bytes()
        );
        w[self.len_bytes] = Word(len_bytes as u64);
    }

    /// Populates the witness with the expected digest value packed into 4 64-bit words
    ///
    /// ## Arguments
    ///
    /// * w - The witness filler to populate
    /// * digest - The expected 32-byte Keccak-256 digest
    pub fn populate_digest(&self, w: &mut WitnessFiller<'_>, digest: [u8; 32]) {
        for (i, bytes) in digest.chunks(8).enumerate() {
            let word = u64::from_le_bytes(bytes.try_into().unwrap());
            w[self.digest[i]] = Word(word);
        }
    }

    /// Populates the witness with padded byte message packed into 64-bit words
    ///
    /// ## Arguments
    ///
    /// * w - The witness filler to populate
    /// * message_bytes - The input message as a byte slice
    pub fn populate_message(&self, w: &mut WitnessFiller<'_>, message_bytes: &[u8]) {
        assert!(
            message_bytes.len() <= self.max_len_bytes(),
            "Message length {} exceeds maximum {}",
            message_bytes.len(),
            self.max_len_bytes()
        );

        // populate message words from input bytes
        let words = self.pack_bytes_into_words(message_bytes, self.max_len_bytes().div_ceil(8));
        for (i, word) in words.iter().enumerate() {
            if i < self.message.len() {
                w[self.message[i]] = Word(*word);
            }
        }

        let mut padded_bytes = vec![0u8; self.n_blocks * RATE_BYTES];

        padded_bytes[..message_bytes.len()].copy_from_slice(message_bytes);

        let msg_len = message_bytes.len();
        let num_full_blocks = msg_len / RATE_BYTES;
        let padding_block_start = num_full_blocks * RATE_BYTES;

        padded_bytes[msg_len] = 0x01;

        let padding_block_end = padding_block_start + RATE_BYTES - 1;
        padded_bytes[padding_block_end] |= 0x80;

        for block_idx in 0..self.n_blocks {
            for (i, chunk) in padded_bytes[block_idx * RATE_BYTES..(block_idx + 1) * RATE_BYTES]
                .chunks(8)
                .enumerate()
            {
                let word = u64::from_le_bytes(chunk.try_into().unwrap());
                w[self.padded_message[block_idx * N_WORDS_PER_BLOCK + i]] = Word(word);
            }
        }
    }

    fn pack_bytes_into_words(&self, bytes: &[u8], n_words: usize) -> Vec<u64> {
        let mut words = Vec::with_capacity(n_words);
        for i in 0..n_words {
            if i * 8 < bytes.len() {
                // to handle messages that are not multiples of 64, bytes are copied into
                // a little endian byte array and then converted to a u64
                let start = i * 8;
                let end = ((i + 1) * 8).min(bytes.len());
                let mut word_bytes = [0u8; 8];
                word_bytes[..end - start].copy_from_slice(&bytes[start..end]);
                let word = u64::from_le_bytes(word_bytes);
                words.push(word);
            }
        }

        words
    }
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::{CircuitBuilder, Wire};
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use rstest::rstest;
    use sha3::Digest;

    use super::*;

    #[rstest]
    #[case(0, 100)] // Empty message
    #[case(1, 100)] // Single byte - minimal non-empty
    #[case(1, 144)] // Single byte - minimal non-empty
    #[case(135, 136)] // 135 bytes - one byte before block boundary
    #[case(136, 136)] // 136 bytes - exactly one block
    #[case(137, 272)] // 137 bytes - crosses block boundary
    #[case(271, 272)] // 271 bytes - one byte before two blocks
    #[case(272, 272)] // 272 bytes - exactly two blocks
    fn test_keccak_circuit(#[case] message_len_bytes: usize, #[case] max_message_len_bytes: usize) {
        // Create test message with deterministic random bytes seeded by the length inputs
        let seed = ((message_len_bytes as u64) << 32) | (max_message_len_bytes as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut message = vec![0u8; message_len_bytes];
        rng.fill_bytes(&mut message);

        // Compute expected digest using sha3 crate
        let mut hasher = sha3::Keccak256::new();
        hasher.update(&message);
        let expected_digest: [u8; 32] = hasher.finalize().into();

        // Build circuit
        assert!(
            message_len_bytes <= max_message_len_bytes,
            "Message length {} exceeds max capacity {} bytes",
            message_len_bytes,
            max_message_len_bytes
        );

        let b = CircuitBuilder::new();
        let len = b.add_witness();
        let digest: [Wire; N_WORDS_PER_DIGEST] = std::array::from_fn(|_| b.add_inout());
        let n_words = max_message_len_bytes.div_ceil(8);
        let message_wires = (0..n_words).map(|_| b.add_inout()).collect();

        let keccak = Keccak256::new(&b, len, digest, message_wires);
        let circuit = b.build();

        // Create and populate witness
        let mut witness = circuit.new_witness_filler();
        keccak.populate_len_bytes(&mut witness, message.len());
        keccak.populate_message(&mut witness, &message);
        keccak.populate_digest(&mut witness, expected_digest);

        // Verify circuit accepts the witness
        circuit
            .populate_wire_witness(&mut witness)
            .expect("Circuit should accept valid witness");

        // Verify all constraints are satisfied
        let cs = circuit.constraint_system();
        verify_constraints(cs, &witness.into_value_vec())
            .expect("All constraints should be satisfied");
    }
}
