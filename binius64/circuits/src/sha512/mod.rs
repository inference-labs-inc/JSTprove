// Copyright 2025 Irreducible Inc.
pub mod compress;

use binius_core::{
    consts::{LOG_BYTE_BITS, LOG_WORD_SIZE_BITS},
    word::Word,
};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
pub use compress::{Compress, State};

use crate::{
    bytes::swap_bytes,
    multiplexer::{multi_wire_multiplex, single_wire_multiplex},
};

/// Verifies that a message produces a specific SHA-512 digest.
///
/// This circuit validates that the provided message, when hashed using SHA-512,
/// produces exactly the expected digest. It implements the full SHA-512 algorithm
/// including proper padding and compression.
///
/// # Wire Layout
///
/// The circuit uses the following wire organization:
/// - Message words: 16 bytes per wire
/// - Digest: 8 wires of 64 bits each, representing the 512-bit hash in big-endian order
///
/// # Limitations
///
/// The message bitlength must be less than 2^32 bits (2^29 bytes) due to SHA-512's
/// length encoding using a 64-bit integer where we only support the lower 32 bits.
pub struct Sha512 {
    /// The actual length of the input message in bytes.
    ///
    /// Must be less than or equal to `max_len_bytes`.
    pub len_bytes: Wire,
    /// The expected SHA-512 digest packed as 8x64-bit words in big-endian order.
    ///
    /// - digest\[0\]: Highest 64 bits (bytes 0-7 of the hash)
    /// - .....
    /// - digest\[7\]: Lowest 64 bits (bytes 56-63 of the hash)
    pub digest: [Wire; 8],
    /// The input message packed as 64-bit words.
    ///
    /// Each wire contains 8 bytes of the message.
    /// This circuit will run enough hash blocks to process the entire message.
    pub message: Vec<Wire>,

    /// Compression gadgets for each 1024-bit block.
    ///
    /// Each compression gadget processes one 1024-bit (128-byte) block of the padded message.
    /// The gadgets are chained together, with each taking the output state from the previous
    /// compression as input. The first compression starts from the SHA-512 initialization vector.
    ///
    /// The number of compression gadgets is `ceil((max_len_bytes + 17) / 128)`, accounting for
    /// the minimum 17 bytes of padding (1 byte for 0x80 delimiter + 16 bytes for length).
    compress: Vec<Compress>,
}

impl Sha512 {
    /// Creates a new SHA-512 verifier circuit.
    ///
    /// # Arguments
    /// * `builder` - Circuit builder for constructing constraints
    /// * `len_bytes` - Wire containing the actual message length in bytes
    /// * `digest` - Expected SHA-512 digest as 8 wires of 64 bits each
    /// * `message` - Input message as packed 64-bit words (16 bytes per wire)
    ///
    /// # Panics If the total number of bits of content contained in `message` cannot be represented
    ///   in 64 bits; i.e., if `message.len() << LOG_WORD_SIZE_BITS > u64::MAX`
    ///
    /// # Circuit Structure
    /// The circuit performs the following validations:
    /// 1. Ensures the actual length is within bounds (len <= max_len_bytes)
    /// 2. Pads the message according to SHA-512 specifications
    /// 3. Computes the hash through chained compression functions
    /// 4. Verifies the computed digest matches the expected digest
    pub fn new(
        builder: &CircuitBuilder,
        len_bytes: Wire,
        digest: [Wire; 8],
        message: Vec<Wire>,
    ) -> Self {
        // ---- Circuit construction overview
        //
        // This function builds a SHA-512 circuit with the following structure:
        //
        // 1. Input validation and setup
        //    - Validate maximum length constraints
        //    - Ensure actual length <= max_len_bytes
        //
        // 2. Message padding and compression setup
        //    - Create padded message blocks following SHA-512 rules
        //    - Chain compression functions from IV to final state
        //    2a. SHA-512 padding position calculation
        //    2b. Final digest selection
        //
        // 3. Message padding constraints (per-word validation)
        //    - 3a: Full message words
        //    - 3b: Boundary word byte checks
        //    - 3c: Zero padding constraints
        //    - 3d: Length field placement

        // ---- 1. Input validation and setup
        //
        // SHA-512 padding routine uses a 64-bit integer to specify the length of the message
        // in bits. This is impractical for a circuit, so we cap the maximum bit length by 2^32,
        // allowing us to check a single word instead of two.
        //
        // We calculate the number of compression blocks needed, including the last padding block
        // for the maximum sized message. Each block is 128 bytes, and we need 17 extra bytes for
        // padding (1 byte for 0x80 delimiter + 16 bytes for length field).
        //
        // We also verify that the actual input length len is within bounds.
        assert!(
            message.len() << LOG_WORD_SIZE_BITS <= u64::MAX as usize,
            "length of message in bits must fit in 64-bit wire"
        );

        let max_len_bytes = message.len() << (LOG_WORD_SIZE_BITS - LOG_BYTE_BITS); // can't call function
        let n_blocks = (message.len() + 3).div_ceil(16);
        // number of 16-word blocks necessary to hold the message and padding.
        // equivalent to `(message.len() * 3 + 17).div_ceil(128)`.
        let n_words: usize = n_blocks << 4; // 16 words per block

        // Check that len_bytes > max_len_bytes is false.
        let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
        builder.assert_false("1.len_check", too_long);

        // ---- 2. Message padding and compression setup
        //
        // Create padded message blocks and compression gadgets.
        //
        // The padded message follows SHA-512 padding requirements and is passed directly to
        // the compression function.
        //
        // Compression gadgets are daisy-chained: each takes the output state from the previous
        // compression as input, with the first compression starting from the SHA-512 IV.

        let padded_message: Vec<Wire> = (0..n_words).map(|_| builder.add_witness()).collect();

        let mut compress = Vec::with_capacity(n_blocks);
        let mut states = Vec::with_capacity(n_blocks + 1);
        states.push(State::iv(builder));
        for block_no in 0..n_blocks {
            let c = Compress::new(
                &builder.subcircuit(format!("compress[{block_no}]")),
                states[block_no].clone(),
                padded_message[block_no << 4..(block_no + 1) << 4]
                    .try_into()
                    .unwrap(),
            );
            states.push(c.state_out.clone());
            compress.push(c);
        }

        // ---- 2a. SHA-512 padding position calculation
        //
        // Calculate where padding elements go. SHA-512 padding has three parts:
        // 1. The 0x80 delimiter byte immediately after the message
        // 2. Zero bytes to fill up to the length field
        // 3. 128-bit length field in the last 16 bytes of a block
        //
        // The length field fits in the same block as the message if there's room for at least
        // 17 bytes of padding (1 delimiter + 16 length). This happens when len % 128 <= 111.
        // Special case: if len % 128 = 0, the message fills the block exactly, so
        // padding goes in the next block.
        //
        // We calculate:
        // - w_bd: word boundary (which word contains the delimiter byte)
        // - end_block_index (which block contains the length field)
        let zero = builder.add_constant(Word::ZERO);
        let w_bd = builder.shr(len_bytes, 3);
        let len_mod_8 = builder.band(len_bytes, builder.add_constant_zx_8(7));
        let bitlen = builder.shl(len_bytes, 3);
        // For SHA-512, the length field is 128 bits. We only support messages < 2^64 bits,
        // so the high 64 bits are zero. We keep `bitlen` as the low 64-bit portion.

        // end_block_index = floor((len + 16) / 128) using 64-bit add
        let (sum, _carry) = builder.iadd_cin_cout(len_bytes, builder.add_constant_64(16), zero);
        let end_block_index = builder.shr(sum, 7);
        // ---- 2b. Final digest selection
        //
        // Select the correct final digest from all compression outputs. The final digest is
        // the state after processing the end_block (the block containing the length field).
        // We use masking and OR operations to conditionally select the right digest.

        let inputs: Vec<&[Wire]> = states[1..].iter().map(|s| &s.0[..]).collect();
        let final_digest_vec = multi_wire_multiplex(builder, &inputs, end_block_index);
        let final_digest: [Wire; 8] = final_digest_vec.try_into().unwrap();

        builder.assert_eq_v("2b.digest", digest, final_digest);

        // ---- 3. Message padding constraints
        //
        // This section validates that the padded message follows SHA-512 padding rules.
        // For each 64-bit word in the padded message, we check:
        //
        // 1. Message words: Must match the input message exactly
        // 2. Boundary word: Contains both message bytes and the start of padding (0x80)
        // 3. Zero padding: All zeros between the delimiter and length field
        // 4. Length field: 64-bit message length in bits (last 8 bytes of a block)
        //
        // Subsections:
        // - 3a: Boundary word byte-level checks
        // - 3a: Full message words (before the boundary)
        // - 3c: Zero padding constraints
        // - 3d: Length field placement

        // ---- 3a. Boundary word byte checks
        //
        // This block implements the boundary word byte checks. That means they are only
        // active for the word in which the boundary between the message and padding occurs.
        //
        // Therefore, every constraint is protected by `is_boundary_word` defined above.
        //
        // For each index of a byte `j` within a word we check whether that byte falls into
        // one of the following categories:
        //
        // 1. message byte. Still part of the input.
        // 2. delimiter byte, placed right after the input.
        // 3. zero byte. Placed after the delimiter byte.

        let boundary_padded_word = single_wire_multiplex(builder, &padded_message, w_bd);
        let boundary_message_word = single_wire_multiplex(builder, &message, w_bd);
        // for the multiplexer above to be sound, we need `sel < inputs.len()` to be true.
        // since we constrained `len_bytes ≤ max_len_bytes ≔ message.len() << 3`, above,
        // we necessarily have `w_bd ≔ len_bytes >> 3 ≤ max_len_bytes >> 3 == message.len()`.
        // in the exceptional case w_bd ≔ len_bytes >> 3 == max_len_bytes >> 3 == message.len().
        // this case can indeed happen. but i claim that we will still get soundness in this case.
        // the only way w_bd = message.len() and len_bytes ≤ max_len_bytes can both be true is if
        // len_bytes = max_len_bytes. in this case, len_bytes is a multiple of 8, so len_mod_8 = 0.
        // in this case, `data_b` will thus be false for each j ∈ {0, … , 7}, ergo, "3b.1" will be
        // dummy'd out for each j, and `boundary_message_word` will be completely ignored.
        // thus it truly doesn't matter what the multiplexer returns; in this case,
        // we are simply asserting that `boundary_padded_word` == 0x 80 00 ...... 00.

        let candidates: Vec<Wire> = (0..8)
            .map(|i| {
                let mask = builder.add_constant_64(0xFFFFFFFFFFFFFF00 << ((7 - i) << 3));
                let padding_byte = builder.add_constant_64(0x8000000000000000 >> (i << 3));
                let message_low = builder.band(boundary_message_word, mask);
                builder.bxor(message_low, padding_byte)
            })
            .collect();
        let expected_padded = single_wire_multiplex(builder, &candidates, len_mod_8);
        builder.assert_eq("expected padded", expected_padded, boundary_padded_word);

        for block_index in 0..n_blocks {
            let builder = builder.subcircuit(format!("word[{block_index}]"));
            let is_length_block =
                builder.icmp_eq(builder.add_constant_64(block_index as u64), end_block_index);

            for column_index in 0..16 {
                let word_index = block_index << 4 | column_index;

                let padded_message_word = padded_message[word_index];

                // flags that help us classify our current position.
                //
                //     1. w     < w_bd - pure message word
                //     2. w    == w_bd - message word at boundary. Mix of message and padding.
                //     3. w_bd  < w    - pure padding word.
                //
                let is_message_word =
                    builder.icmp_ult(builder.add_constant_64(word_index as u64 + 1), w_bd);
                let is_past_message =
                    builder.icmp_ugt(builder.add_constant_64(word_index as u64), w_bd);

                // ---- 3b. Full message words
                if word_index < message.len() {
                    // it is safe to exempt the following check when word_index ≥ message.len().
                    // proof: we constrained above that len_bytes ≤ max_len_bytes.
                    // thus, w_bd ≔ len_bytes >> 3 ≤ max_len_bytes >> 3 == message.len().
                    // so if word_index ≥ message.len() held, then word_index ≥ w_bd also would;
                    // equivalently, is_message_word ≔ (word_index < w_bd) would be false,
                    // so the below constraint would be perma-disabled and we can feely omit / skip.
                    builder.assert_eq_cond(
                        "3b.full_word",
                        message[word_index],
                        padded_message_word,
                        is_message_word,
                    );
                }

                // ---- 3c. Zero padding constraints
                //
                // SHA-512 padding fills the space between the delimiter byte (0x80) and the
                // length field with zeros. We need to ensure all padding words are zero,
                // except for the final two 64-bit words of the length block which contains the
                // message bit length.
                //
                // The length field occupies the last 16 bytes (128 bits) of a block, which
                // corresponds to 64-bit words 14 and 15.
                // We identify padding words as those that are:
                // 1. Past the message boundary (is_past_message = true)
                // 2. NOT the length field location (last two 64-bit words of the length block)

                // ---- 3d. Length field placement
                //
                // When idx == 15, we're looking at the last 64-bit word of a block
                // If this block contains the length field:
                // - Word 15 contains the message bit length
                // Otherwise, if it's a padding word (not message, not length), it must be zero.
                if column_index == 15 {
                    builder.assert_eq_cond(
                        "3d.w15_len",
                        padded_message_word,
                        builder.select(is_length_block, bitlen, zero),
                        is_past_message,
                    );
                } else {
                    builder.assert_eq_cond(
                        "3c.zero_pad",
                        padded_message_word,
                        zero,
                        is_past_message,
                    );
                }
            }
        }

        Self {
            len_bytes,
            digest,
            message,
            compress,
        }
    }

    /// Returns the maximum message length, in bytes.
    pub fn max_len_bytes(&self) -> usize {
        self.message.len() << (LOG_WORD_SIZE_BITS - LOG_BYTE_BITS)
    }

    /// Populates the length wire with the actual message size in bytes.
    ///
    /// # Panics
    /// The method panics if `len_bytes` exceeds `max_len_bytes`.
    pub fn populate_len_bytes(&self, w: &mut WitnessFiller<'_>, len_bytes: usize) {
        assert!(len_bytes <= self.max_len_bytes());
        w[self.len_bytes] = Word(len_bytes as u64);
    }

    /// Populates the digest wires with the expected SHA-512 hash.
    pub fn populate_digest(&self, w: &mut WitnessFiller<'_>, digest: [u8; 64]) {
        for (i, bytes) in digest.chunks(8).enumerate() {
            let word = u64::from_be_bytes(bytes.try_into().unwrap());
            w[self.digest[i]] = Word(word);
        }
    }

    /// Returns digest wires in little-endian packed format.
    ///
    /// The SHA512 digest is stored as 8 wires, each containing 8 bytes of the hash
    /// as a 64-bit big-endian value. This method converts from big-endian to little-endian format.
    ///
    /// # Returns
    /// An array of 8 wires containing the 64-byte digest repacked in little-endian format (8 bytes
    /// per wire)
    pub fn digest_to_le_wires(&self, builder: &CircuitBuilder) -> [Wire; 8] {
        self.digest.map(|be_wire| swap_bytes(builder, be_wire))
    }

    /// Returns message wires in little-endian packed format.
    ///
    /// This method converts from big-endian to little-endian format using the
    /// efficient Hacker's Delight byte swapping algorithm, which is useful for
    /// interfacing with other circuits that expect LE format (e.g., zklogin).
    ///
    /// # Returns
    /// A vector of wires containing the message repacked in little-endian format (8 bytes per wire)
    pub fn message_to_le_wires(&self, builder: &CircuitBuilder) -> Vec<Wire> {
        self.message
            .iter()
            .map(|&be_wire| swap_bytes(builder, be_wire))
            .collect()
    }

    /// Populates the message wires and internal compression blocks with the input message.
    ///
    /// This method handles the complete message preparation including:
    /// 1. Packing the message bytes into 64-bit words
    /// 2. Applying SHA-512 padding (0x80 delimiter + zeros + length)
    /// 3. Populating the compression gadgets with properly formatted blocks
    ///
    /// # Panics
    /// * If `message_bytes.len()` > `max_len_bytes`
    pub fn populate_message(&self, w: &mut WitnessFiller<'_>, message_bytes: &[u8]) {
        assert!(
            message_bytes.len() <= self.max_len_bytes(),
            "message length {} exceeds maximum {}",
            message_bytes.len(),
            self.max_len_bytes()
        );

        let n_blocks = self.compress.len();
        let mut padded_message_bytes = vec![0u8; n_blocks * 128];

        // Apply SHA-512 padding
        //
        // Create padded message following SHA-512 rules:
        // 1. Copy original message
        // 2. Add 0x80 delimiter byte
        // 3. Add zero padding to fill to 112 bytes in the appropriate block
        // 4. Add 64-bit length field in big-endian format
        //
        // The length field placement logic must match the circuit's calculation.
        padded_message_bytes[..message_bytes.len()].copy_from_slice(message_bytes);
        padded_message_bytes[message_bytes.len()] = 0x80;

        let bitlen = (message_bytes.len() as u64) * 8;
        let len_bytes = bitlen.to_be_bytes();

        // SHA-512 requires 17 bytes of padding minimum (1 byte for 0x80 delimiter + 16 bytes for
        // length). The length field must be placed in the last 16 bytes of a 128-byte block.
        // So we can fit the length in the current block only if position after message + 0x80 <=
        // 112. This means len % 128 must be <= 111 to fit everything in the same block.
        let len = message_bytes.len() as u64;
        let end_block_index = (len + 16) / 128;
        // even though there are 16 bytes devoted to the length field, we will only write 8.
        let len_offset = (end_block_index as usize) * 128 + 120;
        padded_message_bytes[len_offset..len_offset + 8].copy_from_slice(&len_bytes);

        // Populate witness wires
        //
        // Pack the padded message into the witness format expected by the circuit:
        // 1. Message wires: 8 bytes per wire
        // 2. Compression inputs: 128-byte blocks passed to each compression gadget
        for (i, wire) in self.message.iter().enumerate() {
            let byte_start = i * 8;

            let mut word = 0u64;
            for j in 0..8 {
                word |= (padded_message_bytes[byte_start + j] as u64) << (56 - j * 8);
            }
            w[*wire] = Word(word);
        }

        for (i, compress) in self.compress.iter().enumerate() {
            let block_start = i * 128;
            let mut block_arr = [0u8; 128];
            block_arr.copy_from_slice(&padded_message_bytes[block_start..block_start + 128]);
            compress.populate_m(w, block_arr);
        }
    }
}

/// Computes SHA-512 hash of a fixed-length message.
///
/// This function creates a subcircuit that computes the SHA-512 hash of a message
/// with a compile-time known length. Unlike the `Sha512` struct which handles
/// variable-length inputs, this function is optimized for fixed-length inputs where
/// the length is known at circuit construction time.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `message` - Input message as packed 64-bit words (8 bytes per wire) in big-endian format. The
///   words should already be encoded from bytes in big-endian order, matching SHA-512's
///   byte-to-word conversion.
/// * `len_bytes` - The fixed length of the message in bytes (known at compile time)
///
/// # Returns
/// * `[Wire; 8]` - The SHA-512 digest as 8 wires of 64 bits each in big-endian order
///
/// # Panics
/// * If `message.len()` does not equal exactly `len_bytes.div_ceil(8)`
/// * If the message length in bits cannot fit in 64 bits
///
/// # Example
/// ```rust,ignore
/// use binius_frontend::crate::sha512::sha512_fixed;
/// use binius_frontend::compiler::CircuitBuilder;
///
/// let mut builder = CircuitBuilder::new();
///
/// // Create input wires for a 32-byte message
/// let message: Vec<_> = (0..4).map(|_| builder.add_witness()).collect();
///
/// // Compute SHA-512 of the 32-byte message
/// let digest = sha512_fixed(&builder, &message, 32);
/// ```
pub fn sha512_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 8] {
    // Validate that message.len() equals exactly len_bytes.div_ceil(8)
    assert_eq!(
        message.len(),
        len_bytes.div_ceil(8),
        "message.len() ({}) must equal len_bytes.div_ceil(8) ({})",
        message.len(),
        len_bytes.div_ceil(8)
    );

    // Ensure message length in bits fits in 64 bits
    assert!(
        (len_bytes as u64).checked_mul(8).is_some(),
        "Message length in bits must fit in 64 bits"
    );

    // Calculate padding requirements
    // SHA-512 requires: message || 0x80 || zeros || 128-bit length field
    // The 128-bit length field goes in the last 16 bytes of a block
    // We need at least 17 bytes for padding (1 for 0x80 + 16 for length)
    let n_blocks = (len_bytes + 17).div_ceil(128);
    let n_padded_words = n_blocks * 16; // 16 words per block

    // Create padded message wires
    let mut padded_message = Vec::with_capacity(n_padded_words);
    if len_bytes.is_multiple_of(8) {
        // Message ends at a word boundary - all words are complete
        padded_message.extend_from_slice(message);
        // Next word starts with 0x80 delimiter
        padded_message.push(builder.add_constant(Word(0x8000000000000000)));
    } else {
        // Message ends mid-word - need to handle boundary word
        padded_message.extend_from_slice(&message[..message.len() - 1]);

        // Handle the last message word which is partial
        let last_idx = message.len() - 1;
        let boundary_byte_in_word = len_bytes % 8;

        // Use shift operations to extract valid bytes and add delimiter
        // Shift right to remove unwanted bytes, then shift left to restore position
        let shift_amount = (8 - boundary_byte_in_word) * 8;
        let shifted_right = builder.shr(message[last_idx], shift_amount as u32);
        let shifted_back = builder.shl(shifted_right, shift_amount as u32);

        // Add 0x80 delimiter at the right position
        let delimiter_shift = (7 - boundary_byte_in_word) * 8;
        let delimiter = builder.add_constant(Word(0x80u64 << delimiter_shift));
        let boundary_word = builder.bxor(shifted_back, delimiter);
        padded_message.push(boundary_word);
    }

    // Fill with zeros until we reach the length field position
    let zero = builder.add_constant(Word::ZERO);
    padded_message.resize(n_padded_words - 2, zero);

    // Add the length field (128 bits, but high 64 bits are always 0 for us)
    padded_message.push(zero);

    let bitlen = (len_bytes as u64) * 8;
    padded_message.push(builder.add_constant(Word(bitlen))); // Low 64 bits of length

    // Process compression blocks
    let state_out = padded_message.chunks(16).enumerate().fold(
        State::iv(builder),
        |state, (block_idx, block)| {
            let block_message: [Wire; 16] = block
                .try_into()
                .expect("padded_message.len() must be divisible by 16");

            let compress = Compress::new(
                &builder.subcircuit(format!("sha512_fixed_compress[{}]", block_idx)),
                state.clone(),
                block_message,
            );

            compress.state_out
        },
    );

    // Return the final state as the digest
    state_out.0
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::{CircuitBuilder, Wire};
    use hex_literal::hex;
    use sha2::Digest;

    use super::{Sha512, sha512_fixed};

    fn mk_circuit(b: &mut CircuitBuilder, max_len: usize) -> Sha512 {
        let len = b.add_witness();
        let digest: [Wire; 8] = std::array::from_fn(|_| b.add_inout());
        let message = (0..max_len).map(|_| b.add_inout()).collect();
        Sha512::new(b, len, digest, message)
    }

    #[test]
    fn full_sha512() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();
        c.populate_len_bytes(&mut w, 3);
        c.populate_message(&mut w, b"abc");
        c.populate_digest(
			&mut w,
			hex!("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"),
		);
        circuit.populate_wire_witness(&mut w).unwrap();
    }

    #[test]
    fn full_sha512_multi_block() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();

        let message_bytes = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopqabcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        c.populate_len_bytes(&mut w, message_bytes.len());
        c.populate_message(&mut w, message_bytes);
        c.populate_digest(
			&mut w,
			hex!("7361ec4a617b6473fb751c44d1026db9442915a5fcea1a419e615d2f3bc5069494da28b8cf2e4412a1dc97d6848f9c84a254fb884ad0720a83eaa0434aeafd8c"),
		);
        circuit.populate_wire_witness(&mut w).unwrap();
    }

    // Helper function to run SHA-512 test with given input and expected digest
    fn test_sha512_with_input(message_bytes: &[u8], expected_digest: [u8; 64]) {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        c.populate_len_bytes(&mut w, message_bytes.len());
        c.populate_message(&mut w, message_bytes);
        c.populate_digest(&mut w, expected_digest);

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_empty_message() {
        test_sha512_with_input(
            b"",
            hex!(
                "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
            ),
        );
    }

    #[test]
    fn test_single_byte() {
        test_sha512_with_input(
            b"a",
            hex!(
                "1f40fc92da241694750979ee6cf582f2d5d7d28e18335de05abc54d0560e0f5302860c652bf08d560252aa5e74210546f369fbbbce8c12cfc7957b2652fe9a75"
            ),
        );
    }

    #[test]
    fn test_two_bytes() {
        test_sha512_with_input(
            b"ab",
            hex!(
                "2d408a0717ec188158278a796c689044361dc6fdde28d6f04973b80896e1823975cdbf12eb63f9e0591328ee235d80e9b5bf1aa6a44f4617ff3caf6400eb172d"
            ),
        );
    }

    #[test]
    fn test_ten_bytes() {
        test_sha512_with_input(
            b"abcdefghij",
            hex!(
                "ef6b97321f34b1fea2169a7db9e1960b471aa13302a988087357c520be957ca119c3ba68e6b4982c019ec89de3865ccf6a3cda1fe11e59f98d99f1502c8b9745"
            ),
        );
    }

    #[test]
    fn test_size_111_bytes() {
        // 111 bytes - maximum that fits in one block with padding
        test_sha512_with_input(
            &[b'a'; 111],
            hex!(
                "fa9121c7b32b9e01733d034cfc78cbf67f926c7ed83e82200ef86818196921760b4beff48404df811b953828274461673c68d04e297b0eb7b2b4d60fc6b566a2"
            ),
        );
    }

    #[test]
    fn test_size_112_bytes() {
        // 112 bytes - critical boundary, forces two blocks
        test_sha512_with_input(
            &[b'a'; 112],
            hex!(
                "c01d080efd492776a1c43bd23dd99d0a2e626d481e16782e75d54c2503b5dc32bd05f0f1ba33e568b88fd2d970929b719ecbb152f58f130a407c8830604b70ca"
            ),
        );
    }

    #[test]
    fn test_size_127_bytes() {
        // 127 bytes - one byte from block boundary
        test_sha512_with_input(
            &[b'a'; 127],
            hex!(
                "828613968b501dc00a97e08c73b118aa8876c26b8aac93df128502ab360f91bab50a51e088769a5c1eff4782ace147dce3642554199876374291f5d921629502"
            ),
        );
    }

    #[test]
    fn test_size_128_bytes() {
        // 128 bytes - exactly one complete block
        test_sha512_with_input(
            &[b'a'; 128],
            hex!(
                "b73d1929aa615934e61a871596b3f3b33359f42b8175602e89f7e06e5f658a243667807ed300314b95cacdd579f3e33abdfbe351909519a846d465c59582f321"
            ),
        );
    }

    #[test]
    fn test_size_200_bytes() {
        // 2200 bytes - tests two-block processing
        test_sha512_with_input(
            &[b'a'; 200],
            hex!(
                "4b11459c33f52a22ee8236782714c150a3b2c60994e9acee17fe68947a3e6789f31e7668394592da7bef827cddca88c4e6f86e4df7ed1ae6cba71f3e98faee9f"
            ),
        );
    }

    #[test]
    fn test_size_239_bytes() {
        // 239 bytes - maximum that fits in two blocks with padding
        test_sha512_with_input(
            &[b'a'; 239],
            hex!(
                "52c853cb8d907f3d4d6b889beb027985d7c273486d75f8baf26f80d24e90c74c6c3de3e22131582380a7d14d43f2941a31385439cd6ddc469f628015e50bf286"
            ),
        );
    }

    #[test]
    fn test_size_240_bytes() {
        // 240 bytes - minimum that needs three blocks
        test_sha512_with_input(
            &[b'a'; 240],
            hex!(
                "4c296d90c61052a62ffb1dd196f1b7b09373b1f93e71836baebf89690546b7595684dbe9467a8e484fa0d1094272b4344a7c24f5fee8daedeb0bf549c985ab5f"
            ),
        );
    }

    #[test]
    fn test_size_256_bytes() {
        // 256 bytes - exactly two complete blocks
        test_sha512_with_input(
            &[b'a'; 256],
            hex!(
                "6a9169eb662f136d87374070e8828b3e615a7eca32a89446e9225b02832709be095e635c824a2bb70213ba2ea0ababac0809827843992c851903b7ac0c136699"
            ),
        );
    }

    #[test]
    fn test_size_512_bytes() {
        // 512 bytes - exactly four complete blocks
        test_sha512_with_input(
            &[b'a'; 512],
            hex!(
                "0210d27bcbe05c2156627c5f136ade1338ab98e06a4591a00b0bcaa61662a5931d0b3bd41a67b5c140627923f5f6307669eb508d8db38b2a8cd41aebd783394b"
            ),
        );
    }

    #[test]
    fn test_size_1024_bytes() {
        test_sha512_with_input(
            &[b'a'; 1024],
            hex!(
                "74b22492e3b9a86a9c93c23a69f821ebafa429302c1f4054b4bc37356a4bae056d9ccbc6f24093a25704faaa72bd21a5f337ca9ec92f32369d24e6b9fae954d8"
            ),
        );
    }

    #[test]
    fn test_realistic_text() {
        // Realistic text around boundary
        test_sha512_with_input(
            b"The quick brown fox jumps over the lazy dog!!!!!",
            hex!(
                "2a8c8f82b62291fdb06439d90b799a6bf63bfa3acc3d627b06151099b54df6b9f860e22c84534033523f4a723d49ffb46ca059157d5cbda70ec878e4f692a38f"
            ),
        );
    }

    #[test]
    fn test_abc_again() {
        // Test the classic 3-byte case to make sure basic functionality still works
        test_sha512_with_input(
            b"abc",
            hex!(
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
            ),
        );
    }

    #[test]
    fn test_mid_range_sizes() {
        // Test various sizes in the 80–111 byte range (single block with varying padding)
        test_sha512_with_input(
            &[b'a'; 95],
            hex!(
                "89e0446c3ff5a04b6d707ef43a77e2b349791f402930dbdb74bbab73d5215e294146ba7bd2fa269aee38564ef11a9ccaf5278f9e82687126dcf20d481d470617"
            ),
        );
        test_sha512_with_input(
            &[b'a'; 100],
            hex!(
                "70ff99fd241905992cc3fff2f6e3f562c8719d689bfe0e53cbc75e53286d82d8767aed0959b8c63aadf55b5730babee75ea082e88414700d7507b988c44c47bc"
            ),
        );
        test_sha512_with_input(
            &[b'a'; 105],
            hex!(
                "3b6dd73c9552f2381107bf206b49c7967fdc5f5011d877d9c576bb4da6d74fbbabf46a1105242d7c645978e54c0b44adaf06d9f7aa4703e8a58829f6d87c5168"
            ),
        );
        test_sha512_with_input(
            &[b'a'; 110],
            hex!(
                "c825949632e509824543f7eaf159fb6041722fce3c1cdcbb613b3d37ff107c519417baac32f8e74fe29d7f4823bf6886956603dca5354a6ed6e4a542e06b7d28"
            ),
        );
    }

    #[test]
    fn test_bogus_length_rejection() {
        // Test that providing wrong length causes circuit to reject
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();

        let message = b"abc";
        // Populate with wrong length (should be 3, but we'll use 5)
        c.populate_len_bytes(&mut w, 5);
        c.populate_message(&mut w, message);
        c.populate_digest(
			&mut w,
			hex!("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"),
		);

        // This should fail when the circuit checks constraints
        let result = circuit.populate_wire_witness(&mut w);
        assert!(result.is_err());
    }

    #[test]
    fn test_length_exceeds_max_rejection() {
        // Test that providing a len > max_len_bytes causes circuit to reject
        let max_len = 1; // 1 Wire's worth of content will be supplied
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, max_len);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();

        let message = b"abc";
        // Bypass the API safety check and set the length wire directly
        w[c.len_bytes] = Word(c.max_len_bytes() as u64 + 1);
        c.populate_message(&mut w, message);
        c.populate_digest(
			&mut w,
			hex!("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"),
		);

        // This should fail at the length check assertion in the circuit
        let result = circuit.populate_wire_witness(&mut w);
        assert!(
            result.is_err(),
            "Circuit should reject length > max_len_bytes"
        );
    }

    #[test]
    fn test_invalid_digest_rejection() {
        // Test that providing wrong digest causes circuit to reject
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();

        let message_bytes = b"abc";
        c.populate_len_bytes(&mut w, message_bytes.len());
        c.populate_message(&mut w, message_bytes);
        // Provide wrong digest (all zeros instead of correct hash)
        c.populate_digest(&mut w, [0u8; 64]);

        // This should fail when the circuit checks constraints
        let result = circuit.populate_wire_witness(&mut w);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_message_content() {
        // Test that providing wrong message content causes circuit to reject
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();

        // Populate with "abc" message but "def" digest
        c.populate_len_bytes(&mut w, 3);
        c.populate_message(&mut w, b"abc");
        // This is the digest for "def", not "abc"
        c.populate_digest(
			&mut w,
			hex!("40a855bf0a93c1019d75dd5b59cd8157608811dd75c5977e07f3bc4be0cad98b22dde4db9ddb429fc2ad3cf9ca379fedf6c1dc4d4bb8829f10c2f0ee04a66663"),
		);

        // This should fail when the circuit checks constraints
        let result = circuit.populate_wire_witness(&mut w);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_len_bytes_edge_cases() {
        // Test that SHA512 circuit construction works correctly for various max_len_bytes values
        // This specifically tests the fix for indexing issues when word_index >= message.len()

        let test_cases = vec![
            // (max_len, description)
            (0, "no input"),
            (13, "fits in one block with padding"),
            (14, "just past boundary"),
            (15, "one word before block boundary"),
            (16, "exactly one block"),
            (29, "fits in two blocks with padding"),
            (30, "just past two-block boundary"),
            (32, "two blocks exactly"),
            (64, "four blocks - previously caused index out of bounds"),
            (128, "eight blocks"),
            (256, "sixteen blocks"),
        ];

        for (max_len, description) in test_cases {
            let mut b = CircuitBuilder::new();
            let c = mk_circuit(&mut b, max_len);
            let circuit = b.build();

            assert_eq!(
                c.message.len(),
                max_len,
                "Wrong number of message wires for max_len={max_len} ({description})"
            );

            // Test with a simple case: empty message
            let mut w = circuit.new_witness_filler();
            c.populate_len_bytes(&mut w, 0);
            c.populate_message(&mut w, b"");
            // SHA512 of empty string
            c.populate_digest(
				&mut w,
				hex!("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"),
			);

            let result = circuit.populate_wire_witness(&mut w);
            assert!(
                result.is_ok(),
                "Failed for max_len={max_len} ({description}) with empty message: {result:?}"
            );
        }
    }

    #[test]
    fn test_sha512_to_le_wires() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 64);

        // Obtain LE-packed wires for the digest wires
        let le_wires = c.digest_to_le_wires(&b);
        assert_eq!(le_wires.len(), 8);

        let circuit = b.build();
        let mut w = circuit.new_witness_filler();
        let message = b"abc";
        let hash = hex!(
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );

        c.populate_len_bytes(&mut w, message.len());
        c.populate_message(&mut w, message);
        c.populate_digest(&mut w, hash);
        circuit.populate_wire_witness(&mut w).unwrap();

        // Extract the LE-packed bytes from the wires
        let mut le_bytes = Vec::with_capacity(64);
        for i in 0..8 {
            let word = w[le_wires[i]].0;
            for j in 0..8 {
                le_bytes.push((word >> (j * 8)) as u8);
            }
        }

        assert_eq!(&le_bytes, &hash);
    }

    #[test]
    fn test_message_to_le_wires() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 16); // Small circuit for simple test

        // Obtain LE-packed wires for the message
        let le_message_wires = c.message_to_le_wires(&b);

        let circuit = b.build();
        let mut w = circuit.new_witness_filler();
        // Use a simple 8-byte message that fits in one wire
        let message = b"abcdefgh";
        let hash = hex!(
            "a3a8c81bc97c2560010d7389bc88aac974a104e0e2381220c6e084c4dccd1d2d17d4f86db31c2a851dc80e6681d74733c55dcd03dd96f6062cdda12a291ae6ce"
        );

        c.populate_len_bytes(&mut w, message.len());
        c.populate_message(&mut w, message);
        c.populate_digest(&mut w, hash);
        circuit.populate_wire_witness(&mut w).unwrap();

        // Check the converted LE wire
        let le_wire = w[le_message_wires[0]].0;

        // Verify the bytes match
        let mut extracted_bytes = Vec::new();
        for j in 0..8 {
            let byte = (le_wire >> (j * 8)) as u8;
            extracted_bytes.push(byte);
        }
        assert_eq!(&extracted_bytes, message);
    }

    // ---- Tests for sha512_fixed function ----

    /// Helper function to test sha512_fixed with a specific message
    fn test_sha512_fixed_with_input(message_bytes: &[u8], expected_digest: [u8; 64]) {
        let builder = CircuitBuilder::new();

        // Create message wires
        let n_words = message_bytes.len().div_ceil(8);
        let message_wires: Vec<Wire> = (0..n_words).map(|_| builder.add_witness()).collect();

        // Create digest output wires
        let expected_digest_wires: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());

        // Call sha512_fixed
        let computed_digest = sha512_fixed(&builder, &message_wires, message_bytes.len());

        // Assert computed digest equals expected
        for i in 0..8 {
            builder.assert_eq(
                format!("digest[{i}]"),
                computed_digest[i],
                expected_digest_wires[i],
            );
        }

        let circuit = builder.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        // Populate message wires
        for (i, wire) in message_wires.iter().enumerate() {
            let byte_start = i * 8;
            let byte_end = ((i + 1) * 8).min(message_bytes.len());

            let mut word = 0u64;
            for j in byte_start..byte_end {
                word |= (message_bytes[j] as u64) << (56 - (j - byte_start) * 8);
            }
            w[*wire] = Word(word);
        }

        // Populate expected digest wires
        for (i, bytes) in expected_digest.chunks(8).enumerate() {
            let word = u64::from_be_bytes(bytes.try_into().unwrap());
            w[expected_digest_wires[i]] = Word(word);
        }

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    #[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(8) (2)")]
    fn test_sha512_fixed_with_insufficient_wires() {
        let builder = CircuitBuilder::new();

        // Create only 1 wire but claim message is 10 bytes (which needs 2 wires)
        let message_wires: Vec<Wire> = vec![builder.add_witness()];

        // This should panic because message.len() (1) != len_bytes.div_ceil(8) (2)
        sha512_fixed(&builder, &message_wires, 10);
    }

    #[test]
    fn test_sha512_fixed_exact_wire_count() {
        let builder = CircuitBuilder::new();

        // Test that the function requires exact wire count

        // Empty message: 0 bytes requires 0 wires
        let empty: Vec<Wire> = vec![];
        let _ = sha512_fixed(&builder, &empty, 0);

        // 8 bytes requires exactly 1 wire
        let one_wire: Vec<Wire> = vec![builder.add_witness()];
        let _ = sha512_fixed(&builder, &one_wire, 8);

        // 10 bytes requires exactly 2 wires (10.div_ceil(8) = 2)
        let two_wires: Vec<Wire> = vec![builder.add_witness(), builder.add_witness()];
        let _ = sha512_fixed(&builder, &two_wires, 10);

        // 16 bytes requires exactly 2 wires
        let two_wires_full: Vec<Wire> = vec![builder.add_witness(), builder.add_witness()];
        let _ = sha512_fixed(&builder, &two_wires_full, 16);

        // 17 bytes requires exactly 3 wires (17.div_ceil(8) = 3)
        let three_wires: Vec<Wire> = vec![
            builder.add_witness(),
            builder.add_witness(),
            builder.add_witness(),
        ];
        let _ = sha512_fixed(&builder, &three_wires, 17);
    }

    #[test]
    fn test_sha512_fixed_various_sizes() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // Test various message sizes to ensure padding works correctly
        let sizes = vec![
            0,   // empty
            1,   // single byte
            7,   // just under word boundary
            8,   // exactly one word
            9,   // just over word boundary
            63,  // just under half block
            64,  // exactly half block
            65,  // just over half block
            111, // max single block
            112, // forces two blocks
            127, // one byte from block boundary
            128, // exactly one block
            129, // just over one block
            239, // max two blocks
            240, // forces three blocks
            256, // exactly two blocks
        ];

        let mut rng = StdRng::seed_from_u64(0);

        for size in sizes {
            // Generate random payload
            let mut message = vec![0u8; size];
            rng.fill(&mut message[..]);

            // Compute expected hash using sha2 crate
            let expected = sha2::Sha512::digest(&message);
            let expected_bytes: [u8; 64] = expected.into();

            // Test with our circuit
            test_sha512_fixed_with_input(&message, expected_bytes);
        }
    }
}
