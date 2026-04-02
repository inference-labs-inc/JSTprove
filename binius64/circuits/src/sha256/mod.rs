// Copyright 2025 Irreducible Inc.
pub mod compress;

use binius_core::{
    consts::{LOG_BYTE_BITS, LOG_WORD_SIZE_BITS},
    word::Word,
};
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
pub use compress::{Compress, State};

use crate::{
    bytes::{swap_bytes, swap_bytes_32},
    multiplexer::{multi_wire_multiplex, single_wire_multiplex},
};

/// Verifies that a message produces a specific SHA-256 digest.
///
/// This circuit validates that the provided message, when hashed using SHA-256,
/// produces exactly the expected digest. It implements the full SHA-256 algorithm
/// including proper padding and compression.
///
/// # Wire Layout
///
/// The circuit uses the following wire organization:
/// - Message words: 8 bytes per wire, packed as two XORed 32-bit big-endian words
/// - Digest: 4 wires of 64 bits each, representing the 256-bit hash in big-endian order
///
/// # Limitations
///
/// The message bitlength must be less than 2^32 bits (2^29 bytes) due to SHA-256's
/// length encoding using a 64-bit integer where we only support the lower 32 bits.
pub struct Sha256 {
    /// The actual length of the input message in bytes.
    ///
    /// Must be less than or equal to `max_len_bytes`.
    pub len_bytes: Wire,
    /// The expected SHA-256 digest packed as 4x64-bit words in big-endian order.
    ///
    /// - digest\[0\]: High 64 bits (bytes 0-7 of the hash)
    /// - digest\[1\]: Next 64 bits (bytes 8-15 of the hash)
    /// - digest\[2\]: Next 64 bits (bytes 16-23 of the hash)
    /// - digest\[3\]: Low 64 bits (bytes 24-31 of the hash)
    pub digest: [Wire; 4],
    /// The input message packed as 64-bit words.
    ///
    /// Each wire contains 8 bytes of the message packed as two XORed 32-bit big-endian words.
    /// This circuit will run enough hash blocks to process the entire message.
    pub message: Vec<Wire>,

    /// Compression gadgets for each 512-bit block.
    ///
    /// Each compression gadget processes one 512-bit (64-byte) block of the padded message.
    /// The gadgets are chained together, with each taking the output state from the previous
    /// compression as input. The first compression starts from the SHA-256 initialization vector.
    ///
    /// The number of compression gadgets is `ceil((max_len_bytes + 9) / 64)`, accounting for
    /// the minimum 9 bytes of padding (1 byte for 0x80 delimiter + 8 bytes for length).
    compress: Vec<Compress>,
}

impl Sha256 {
    /// Creates a new SHA-256 verifier circuit.
    ///
    /// # Arguments
    /// * `builder` - Circuit builder for constructing constraints
    /// * `len_bytes` - Wire containing the actual message length in bytes
    /// * `digest` - Expected SHA-256 digest as 4 wires of 64 bits each
    /// * `message` - Input message as packed 64-bit words (8 bytes per wire)
    ///
    /// # Panics If the total number of bits of content contained in `message` cannot be represented in 32
    ///   bits; i.e., if `message.len() << LOG_WORD_SIZE_BITS > u32::MAX`
    ///
    /// # Circuit Structure
    /// The circuit performs the following validations:
    /// 1. Ensures the actual length is within bounds (len_bytes <= max_len_bytes)
    /// 2. Pads the message according to SHA-256 specifications
    /// 3. Computes the hash through chained compression functions
    /// 4. Verifies the computed digest matches the expected digest
    pub fn new(
        builder: &CircuitBuilder,
        len_bytes: Wire,
        digest: [Wire; 4],
        message: Vec<Wire>,
    ) -> Self {
        // ---- Circuit construction overview
        //
        // This function builds a SHA-256 circuit with the following structure:
        //
        // 1. Input validation and setup
        //    - Validate maximum length constraints
        //    - Ensure actual length <= max_len_bytes
        //
        // 2. Message padding and compression setup
        //    - Create padded message blocks following SHA-256 rules
        //    - Chain compression functions from IV to final state
        //    2a. SHA-256 padding position calculation
        //    2b. Final digest selection
        //
        // 3. Message padding constraints (per-word validation)
        //    - 3a: Full message words
        //    - 3b: Boundary word byte checks
        //    - 3c: Zero padding constraints
        //    - 3d: Length field placement

        // ---- 1. Input validation and setup
        //
        // SHA-256 padding routine uses a 64-bit integer to specify the length of the message
        // in bits. This is impractical for a circuit, so we cap the maximum bit length by 2^32,
        // allowing us to check a single word instead of two.
        //
        // We calculate the number of compression blocks needed, including the last padding block
        // for the maximum sized message. Each block is 64 bytes, and we need 9 extra bytes for
        // padding (1 byte for 0x80 delimiter + 8 bytes for length field).
        //
        // We also verify that the actual input length len is within bounds.
        assert!(
            message.len() << LOG_WORD_SIZE_BITS <= u32::MAX as usize,
            "length of message in bits must fit within 32 bits"
        );

        let max_len_bytes = message.len() << (LOG_WORD_SIZE_BITS - LOG_BYTE_BITS);
        let n_blocks = (message.len() + 2).div_ceil(8);
        let n_words = n_blocks << 3; // 8 message words per compression gadget block

        // Check that len_bytes > max_len_bytes is false.
        let too_long = builder.icmp_ugt(len_bytes, builder.add_constant_64(max_len_bytes as u64));
        builder.assert_false("1.len_check", too_long);

        // ---- 2. Message padding and compression setup
        //
        // Create padded message blocks and compression gadgets. The padded message is packed
        // differently from the input message words:
        //     16 × 32-bit schedule words = 8 × 64-bit message words
        //
        // The padded message follows SHA-256 padding requirements and is passed directly to
        // the compression function.
        //
        // Compression gadgets are daisy-chained: each takes the output state from the previous
        // compression as input, with the first compression starting from the SHA-256 IV.

        // for each compression gadget, we need to digest 16 32-bit words' worth of content.
        // since message is packed into 64-bit wires, we need to do a few slight tricks.
        // each chunk we feed into compression needs to contain 16 × 32 = 512 bits of information.
        // we represent this as 16 64-bit wires, each of which is completely empty on the high half.
        // it winds up being convenient to split the total `n_blocks` × 16 wires into two halves,
        // `padded_evens` and `padded_odds`, each containing `n_blocks` × 8 total wires.
        // for each `n_block`, the interleaving of padded_evens' and padded_odds' respective
        // [n_block << 3 .. (n_block + 1) << 3] slices will give us the 16 half-empty wires we need.
        let padded_evens: Vec<Wire> = (0..n_words).map(|_| builder.add_witness()).collect();
        let padded_odds: Vec<Wire> = (0..n_words).map(|_| builder.add_witness()).collect();

        let mut compress = Vec::with_capacity(n_blocks);
        let mut states = Vec::with_capacity(n_blocks + 1);
        states.push(State::iv(builder));
        for block_no in 0..n_blocks {
            let c = Compress::new(
                &builder.subcircuit(format!("compress[{block_no}]")),
                states[block_no].clone(),
                // grab appropriate interleaved wires, in order to feed into compression gadget.
                std::array::from_fn(|i| {
                    if i & 1 == 0 {
                        padded_evens[block_no << 3 | i >> 1]
                    } else {
                        padded_odds[block_no << 3 | i >> 1]
                    }
                }),
            );
            states.push(c.state_out.clone());
            compress.push(c);
        }

        // ---- 2a. SHA-256 padding position calculation
        //
        // Calculate where padding elements go. SHA-256 padding has three parts:
        // 1. The 0x80 delimiter byte immediately after the message
        // 2. Zero bytes to fill up to the length field
        // 3. 64-bit length field in the last 8 bytes of a block
        //
        // The length field fits in the same block as the message if there's room for at least
        // 9 bytes of padding (1 delimiter + 8 length). This happens when len_bytes % 64 <= 55.
        // Special case: if len_bytes % 64 = 0 and len_bytes > 0, the message fills the block
        // exactly, so padding goes in the next block.
        //
        // We calculate:
        // - w_bd: word boundary (which word contains the delimiter byte)
        // - end_block_index (which block contains the length field)
        let w_bd = builder.shr(len_bytes, 3);
        let len_mod_8 = builder.band(len_bytes, builder.add_constant_zx_8(7));
        let bitlen = builder.shl(len_bytes, 3);

        let zero = builder.add_constant(Word::ZERO);
        let end_block_index =
            builder.shr(builder.iadd_32(len_bytes, builder.add_constant_64(8)), 6);
        let m32 = builder.add_constant(Word::MASK_32);
        // ---- 2b. Final digest selection
        //
        // Select the correct final digest from all compression outputs. The final digest is
        // the state after processing the end_block (the block containing the length field).
        // We use masking and OR operations to conditionally select the right digest.

        let block_digests = states[1..]
            .iter()
            .map(|s| s.pack_4x64b(builder))
            .collect::<Vec<_>>();
        let inputs: Vec<&[Wire]> = block_digests.iter().map(|arr| &arr[..]).collect();
        let final_digest_vec = multi_wire_multiplex(builder, &inputs, end_block_index);
        let final_digest: [Wire; 4] = final_digest_vec.try_into().unwrap();

        builder.assert_eq_v("2b.digest", digest, final_digest);

        // ---- 3. Message padding constraints
        //
        // This section validates that the padded message follows SHA-256 padding rules.
        // For each 64-bit word in the padded message, we check:
        //
        // 1. Message words: Must match the input message exactly
        // 2. Boundary word: Contains both message bytes and the start of padding (0x80)
        // 3. Zero padding: All zeros between the delimiter and length field
        // 4. Length field: 64-bit message length in bits (last 8 bytes of a block)
        //
        // Subsections:
        // - 3a: Boundary word byte-level checks
        // - 3b: Full message words (before the boundary)
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

        let boundary_padded_lo32 = single_wire_multiplex(builder, &padded_evens, w_bd);
        let boundary_padded_hi32 = single_wire_multiplex(builder, &padded_odds, w_bd);
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
                let temp_mask = 0xFFFFFFFFFFFFFF00 << ((7 - i) << 3);
                let halves_swap = temp_mask << 32 ^ temp_mask >> 32;
                let mask = builder.add_constant_64(halves_swap);
                let padding_byte = builder.add_constant_64(0x8000000000000000 >> ((i ^ 4) << 3));
                let message_low = builder.band(boundary_message_word, mask);
                builder.bxor(message_low, padding_byte)
            })
            .collect();

        let expected_padded = single_wire_multiplex(builder, &candidates, len_mod_8);
        builder.assert_eq(
            "first half",
            builder.band(expected_padded, m32),
            boundary_padded_lo32,
        );
        builder.assert_eq(
            "second half",
            builder.shr(expected_padded, 32),
            boundary_padded_hi32,
        );
        // also guarantees as a side effect that `boundary_padded` hi and lo have empty high halves.

        for block_index in 0..n_blocks {
            let builder = builder.subcircuit(format!("word[{block_index}]"));
            let is_length_block =
                builder.icmp_eq(builder.add_constant_64(block_index as u64), end_block_index);

            for column_index in 0..8 {
                // From two adjacent 32-bit message schedule words get a packed 64-bit message word.
                let word_index = block_index << 3 | column_index;
                let w_lo32 = padded_evens[word_index];
                let w_hi32 = padded_odds[word_index];

                // flags that help us classify our current position.
                //
                //     1. w     < w_bd - pure message word
                //     2. w    == w_bd - message word at boundary. Mix of message and padding.
                //     3. w_bd  < w    - pure padding word.

                let is_message_word =
                    builder.icmp_ult(builder.add_constant_64(word_index as u64 + 1), w_bd);
                let is_past_message =
                    builder.icmp_ugt(builder.add_constant_64(word_index as u64), w_bd);

                // ---- 3a. Full message words
                if word_index < message.len() {
                    // independently check both halves of message.

                    // it is safe to exempt the following check when word_index ≥ message.len().
                    // proof: we constrained above that len_bytes ≤ max_len_bytes.
                    // thus, w_bd ≔ len_bytes >> 3 ≤ max_len_bytes >> 3 == message.len().
                    // so if word_index ≥ message.len() held, then word_index ≥ w_bd also would;
                    // equivalently, is_message_word ≔ (word_index < w_bd) would be false,
                    // so the below constraint would be perma-disabled and we can feely omit / skip.
                    builder.assert_eq_cond(
                        "3a.full_word",
                        builder.band(message[word_index], m32),
                        w_lo32,
                        is_message_word,
                    );
                    builder.assert_eq_cond(
                        "3a.full_word",
                        builder.shr(message[word_index], 32),
                        w_hi32,
                        is_message_word,
                    );
                }

                // ---- 3c. Zero padding constraints
                //
                // SHA-256 padding fills the space between the delimiter byte (0x80) and the
                // length field with zeros. We need to ensure all padding words are zero,
                // except for the final 64-bit word of the length block which contains the
                // message bit length.
                //
                // The length field occupies the last 8 bytes (64 bits) of a block, which
                // corresponds to 32-bit words 14 and 15 (packed as 64-bit word index 7).
                // We identify padding words as those that are:
                // 1. Past the message boundary (is_past_message = true)
                // 2. NOT the length field location (last 64-bit word of the length block)

                // ---- 3d. Length field placement
                //
                // When column_index == 7, we're looking at the last two 32-bit words of a block.
                // If this block contains the length field:
                // - the low part of the 7th word should be zero;
                // - the high part of the 7th word should contain the message bit length.
                // Otherwise, if it's a padding word (not message, not length), it must be zero.
                if column_index == 7 {
                    builder.assert_eq_cond("3d.w_lo32_7_zero", w_lo32, zero, is_past_message);
                    builder.assert_eq_cond(
                        "3d.w_hi32_7_len",
                        w_hi32,
                        builder.select(is_length_block, bitlen, zero),
                        is_past_message,
                    );
                } else {
                    builder.assert_eq_cond("3d.zero_pad", w_lo32, zero, is_past_message);
                    builder.assert_eq_cond("3d.zero_pad", w_hi32, zero, is_past_message);
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

    /// Populates the digest wires with the expected SHA-256 hash.
    pub fn populate_digest(&self, w: &mut WitnessFiller<'_>, digest: [u8; 32]) {
        for (i, bytes) in digest.chunks(8).enumerate() {
            let word = u64::from_be_bytes(bytes.try_into().unwrap());
            w[self.digest[i]] = Word(word);
        }
    }

    /// Returns digest wires in little-endian packed format.
    ///
    /// The SHA256 digest is stored as 4 wires, each containing 8 bytes of the hash
    /// as a 64-bit big-endian value. This method converts from big-endian to little-endian format.
    ///
    /// # Returns
    /// An array of 4 wires containing the 32-byte digest repacked in little-endian format (8 bytes
    /// per wire)
    pub fn digest_to_le_wires(&self, builder: &CircuitBuilder) -> [Wire; 4] {
        self.digest.map(|be_wire| swap_bytes(builder, be_wire))
    }

    /// Returns message wires in little-endian packed format.
    ///
    /// The SHA256 message is stored with each wire containing 8 bytes packed as two XORed 32-bit
    /// big-endian words: `lo_word ^ (hi_word << 32)`.
    ///
    /// This method converts both 32-bit big-endian words to little-endian format simultaneously,
    /// which is useful for interfacing with other circuits that expect LE format (e.g., zklogin).
    ///
    /// # Returns
    /// A vector of wires containing the message repacked in little-endian format (8 bytes per wire)
    pub fn message_to_le_wires(&self, builder: &CircuitBuilder) -> Vec<Wire> {
        self.message
            .iter()
            .map(|&sha256_wire| swap_bytes_32(builder, sha256_wire))
            .collect()
    }

    /// Populates the message wires and internal compression blocks with the input message.
    ///
    /// This method handles the complete message preparation including:
    /// 1. Packing the message bytes into 64-bit words
    /// 2. Applying SHA-256 padding (0x80 delimiter + zeros + length)
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
        let mut padded_message_bytes = vec![0u8; n_blocks * 64];

        // Apply SHA-256 padding
        //
        // Create padded message following SHA-256 rules:
        // 1. Copy original message
        // 2. Add 0x80 delimiter byte
        // 3. Add zero padding to fill to 56 bytes in the appropriate block
        // 4. Add 64-bit length field in big-endian format
        //
        // The length field placement logic must match the circuit's calculation.
        padded_message_bytes[..message_bytes.len()].copy_from_slice(message_bytes);
        padded_message_bytes[message_bytes.len()] = 0x80;

        let bitlen = (message_bytes.len() as u64) * 8;
        let len_bytes = bitlen.to_be_bytes();

        // SHA-256 requires 9 bytes of padding minimum (1 byte for 0x80 delimiter + 8 bytes for
        // length). The length field must be placed in the last 8 bytes of a 64-byte block.
        // So we can fit the length in the current block only if position after message + 0x80 <=
        // 56. This means len % 64 must be <= 55 to fit everything in the same block.
        let len = message_bytes.len() as u64;
        let end_block_index = (len + 8) / 64;
        // Length field always starts at byte offset 56 within its block (64 - 8 = 56)
        let len_offset = (end_block_index as usize) * 64 + 56;
        padded_message_bytes[len_offset..len_offset + 8].copy_from_slice(&len_bytes);

        // Populate witness wires
        //
        // Pack the padded message into the witness format expected by the circuit:
        // 1. Message wires: 8 bytes per wire, packed as two 32-bit big-endian words XORed together
        // 2. Compression inputs: 64-byte blocks passed to each compression gadget
        for (i, wire) in self.message.iter().enumerate() {
            let byte_start = i * 8;

            let mut lo_word = 0u32;
            for j in 0..4 {
                lo_word |= (padded_message_bytes[byte_start + j] as u32) << (24 - j * 8);
            }

            let mut hi_word = 0u32;
            for j in 4..8 {
                hi_word |= (padded_message_bytes[byte_start + j] as u32) << (24 - (j - 4) * 8);
            }

            let word = (lo_word as u64) ^ ((hi_word as u64) << 32);
            w[*wire] = Word(word);
        }

        for (i, compress) in self.compress.iter().enumerate() {
            let block_start = i * 64;
            let block = &padded_message_bytes[block_start..block_start + 64];
            compress.populate_m(w, block.try_into().unwrap());
        }
    }
}

/// Computes SHA-256 hash of a fixed-length message.
///
/// This function creates a subcircuit that computes the SHA-256 hash of a message
/// with a compile-time known length. Unlike the `Sha256` struct which handles
/// variable-length inputs, this function is optimized for fixed-length inputs where
/// the length is known at circuit construction time.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `message` - Input message as 32-bit words (4 bytes per wire) in big-endian format. Each wire
///   must have the high 32 bits set to zero (enforced as a precondition).
/// * `len_bytes` - The fixed length of the message in bytes (known at compile time)
///
/// # Returns
/// * `[Wire; 8]` - The SHA-256 digest as 8 wires, each containing a 32-bit word in the low 32 bits
///   (high 32 bits are zero) in big-endian order
///
/// # Panics
/// * If `message.len()` does not equal exactly `len_bytes.div_ceil(4)`
/// * If the message length in bits cannot fit in 32 bits
///
/// # Example
/// ```rust,ignore
/// use binius_frontend::crate::sha256::sha256_fixed;
/// use binius_frontend::compiler::CircuitBuilder;
///
/// let mut builder = CircuitBuilder::new();
///
/// // Create input wires for a 32-byte message (8 32-bit words)
/// let message: Vec<_> = (0..8).map(|_| builder.add_witness()).collect();
///
/// // Compute SHA-256 of the 32-byte message
/// let digest = sha256_fixed(&builder, &message, 32);
/// ```
pub fn sha256_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 8] {
    // Validate that message.len() equals exactly len_bytes.div_ceil(4)
    assert_eq!(
        message.len(),
        len_bytes.div_ceil(4),
        "message.len() ({}) must equal len_bytes.div_ceil(4) ({})",
        message.len(),
        len_bytes.div_ceil(4)
    );

    // Ensure message length in bits fits in 32 bits
    assert!(
        (len_bytes as u64)
            .checked_mul(8)
            .is_some_and(|bits| bits <= u32::MAX as u64),
        "Message length in bits must fit in 32 bits"
    );

    // Calculate padding requirements
    // SHA-256 requires: message || 0x80 || zeros || 64-bit length field
    // The 64-bit length field goes in the last 8 bytes of a block
    // We need at least 9 bytes for padding (1 for 0x80 + 8 for length)
    let n_blocks = (len_bytes + 9).div_ceil(64);
    let n_padded_words = n_blocks * 16; // 16 32-bit words per block

    // Create padded message
    let mut padded_message = Vec::with_capacity(n_padded_words);

    // Add message words
    let n_message_words = len_bytes / 4;
    let boundary_bytes = len_bytes % 4;

    // Add complete message words
    padded_message.extend_from_slice(&message[0..n_message_words]);

    // Handle partial word at boundary
    if boundary_bytes > 0 {
        // The last message word contains partial data
        let last_word = message[n_message_words];

        // Mask out the unused bytes and add delimiter
        let shift_amount = (4 - boundary_bytes) * 8;
        let mask = builder.add_constant(Word((0xFFFFFFFFu64 >> shift_amount) << shift_amount));
        let masked = builder.band(last_word, mask);

        // Add 0x80 delimiter at the right position
        let delimiter_shift = (3 - boundary_bytes) * 8;
        let delimiter = builder.add_constant(Word(0x80u64 << delimiter_shift));
        let boundary_word = builder.bxor(masked, delimiter);

        padded_message.push(boundary_word);
    } else {
        // Message ends at word boundary - delimiter goes in new word
        padded_message.push(builder.add_constant(Word(0x80000000)));
    }

    // Fill with zeros until we reach the length field position
    let zero = builder.add_constant(Word::ZERO);
    padded_message.resize(n_padded_words - 2, zero);

    // Add the length field (64 bits total)
    padded_message.push(zero); // High 32 bits of length (always 0 for us)
    let bitlen = (len_bytes as u64) * 8;
    padded_message.push(builder.add_constant(Word(bitlen)));

    // Process compression blocks
    let state_out = padded_message.chunks_exact(16).enumerate().fold(
        State::iv(builder),
        |state, (block_idx, block)| {
            let block_message: [Wire; 16] = block.try_into().unwrap();

            let compress = Compress::new(
                &builder.subcircuit(format!("sha256_fixed_compress[{}]", block_idx)),
                state.clone(),
                block_message,
            );

            compress.state_out
        },
    );

    // Return the final state as 8 32-bit words
    state_out.0
}

#[cfg(test)]
mod tests {
    use std::array;

    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::{CircuitBuilder, Wire};
    use hex_literal::hex;
    use sha2::Digest;

    use super::*;

    fn mk_circuit(b: &mut CircuitBuilder, max_len: usize) -> Sha256 {
        let len = b.add_witness();
        let digest: [Wire; 4] = std::array::from_fn(|_| b.add_inout());
        let message = (0..max_len).map(|_| b.add_inout()).collect();
        Sha256::new(b, len, digest, message)
    }

    #[test]
    fn full_sha256() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();
        c.populate_len_bytes(&mut w, 3);
        c.populate_message(&mut w, b"abc");
        c.populate_digest(
            &mut w,
            hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        );
        circuit.populate_wire_witness(&mut w).unwrap();
    }

    #[test]
    fn full_sha256_multi_block() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 256);
        let circuit = b.build();
        let mut w = circuit.new_witness_filler();

        let message_bytes = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        c.populate_len_bytes(&mut w, message_bytes.len());
        c.populate_message(&mut w, message_bytes);
        c.populate_digest(
            &mut w,
            hex!("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        );
        circuit.populate_wire_witness(&mut w).unwrap();
    }

    // Helper function to run SHA-256 test with given input and expected digest
    fn test_sha256_with_input(message_bytes: &[u8], expected_digest: [u8; 32]) {
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
        test_sha256_with_input(
            b"",
            hex!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        );
    }

    #[test]
    fn test_single_byte() {
        test_sha256_with_input(
            b"a",
            hex!("ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"),
        );
    }

    #[test]
    fn test_two_bytes() {
        test_sha256_with_input(
            b"ab",
            hex!("fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603"),
        );
    }

    #[test]
    fn test_ten_bytes() {
        test_sha256_with_input(
            b"abcdefghij",
            hex!("72399361da6a7754fec986dca5b7cbaf1c810a28ded4abaf56b2106d06cb78b0"),
        );
    }

    #[test]
    fn test_size_55_bytes() {
        // 55 bytes - maximum that fits in one block with padding
        test_sha256_with_input(
            &[b'a'; 55],
            hex!("9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"),
        );
    }

    #[test]
    fn test_size_56_bytes() {
        // 56 bytes - critical boundary, forces two blocks
        test_sha256_with_input(
            &[b'a'; 56],
            hex!("b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"),
        );
    }

    #[test]
    fn test_size_63_bytes() {
        // 63 bytes - one byte from block boundary
        test_sha256_with_input(
            &[b'a'; 63],
            hex!("7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"),
        );
    }

    #[test]
    fn test_size_64_bytes() {
        // 64 bytes - exactly one complete block
        test_sha256_with_input(
            &[b'a'; 64],
            hex!("ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"),
        );
    }

    #[test]
    fn test_size_100_bytes() {
        // 100 bytes - tests two-block processing
        test_sha256_with_input(
            &[b'a'; 100],
            hex!("2816597888e4a0d3a36b82b83316ab32680eb8f00f8cd3b904d681246d285a0e"),
        );
    }

    #[test]
    fn test_size_119_bytes() {
        // 119 bytes - maximum that fits in two blocks with padding
        test_sha256_with_input(
            &[b'a'; 119],
            hex!("31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb"),
        );
    }

    #[test]
    fn test_size_120_bytes() {
        // 120 bytes - minimum that needs three blocks
        test_sha256_with_input(
            &[b'a'; 120],
            hex!("2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c"),
        );
    }

    #[test]
    fn test_size_128_bytes() {
        // 128 bytes - exactly two complete blocks
        test_sha256_with_input(
            &[b'a'; 128],
            hex!("6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e"),
        );
    }

    #[test]
    fn test_size_256_bytes() {
        // 256 bytes - exactly four complete blocks
        test_sha256_with_input(
            &[b'a'; 256],
            hex!("02d7160d77e18c6447be80c2e355c7ed4388545271702c50253b0914c65ce5fe"),
        );
    }

    #[test]
    fn test_size_512_bytes() {
        test_sha256_with_input(
            &[b'a'; 512],
            hex!("471be6558b665e4f6dd49f1184814d1491b0315d466beea768c153cc5500c836"),
        );
    }

    #[test]
    fn test_realistic_text() {
        // Realistic text around boundary
        test_sha256_with_input(
            b"The quick brown fox jumps over the lazy dog!!!!!",
            hex!("1042cd9153723d8e9124a60f2817843711a5c6b10170c80bdec99cd0c82e3dfe"),
        );
    }

    #[test]
    fn test_abc_again() {
        // Test the classic 3-byte case to make sure basic functionality still works
        test_sha256_with_input(
            b"abc",
            hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        );
    }

    #[test]
    fn test_mid_range_sizes() {
        // Test various sizes in the 49-54 byte range (single block with varying padding)
        test_sha256_with_input(
            &[b'a'; 49],
            hex!("8f9bec6a62dd28ebd36d1227745592de6658b36974a3bb98a4c582f683ea6c42"),
        );
        test_sha256_with_input(
            &[b'a'; 50],
            hex!("160b4e433e384e05e537dc59b467f7cb2403f0214db15c5db58862a3f1156d2e"),
        );
        test_sha256_with_input(
            &[b'a'; 53],
            hex!("abe346a7259fc90b4c27185419628e5e6af6466b1ae9b5446cac4bfc26cf05c4"),
        );
        test_sha256_with_input(
            &[b'a'; 54],
            hex!("a3f01b6939256127582ac8ae9fb47a382a244680806a3f613a118851c1ca1d47"),
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
            hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        );

        // This should fail when the circuit checks constraints
        let result = circuit.populate_wire_witness(&mut w);
        assert!(result.is_err());
    }

    #[test]
    fn test_length_exceeds_max_rejection() {
        // Test that providing a length > max_len_bytes causes circuit to reject
        let max_len = 3;
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
            hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
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

        let message = b"abc";
        c.populate_len_bytes(&mut w, message.len());
        c.populate_message(&mut w, message);
        // Provide wrong digest (all zeros instead of correct hash)
        c.populate_digest(&mut w, [0u8; 32]);

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
            hex!("89c8a3b0a2eb7b275eb983c1c3f22cb5cb6b07b962e2d60ccd4b8d88bb7306c4"),
        );

        // This should fail when the circuit checks constraints
        let result = circuit.populate_wire_witness(&mut w);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_len_bytes_edge_cases() {
        // Test that SHA256 circuit construction works correctly for various max_len_bytes values
        // This specifically tests the fix for indexing issues when word_index >= message.len()

        let test_cases = vec![
            // (max_len, description)
            (0, "no input"),
            (6, "fits in one block with padding"),
            (7, "just past boundary"),
            (8, "exactly one block"),
            (14, "fits in two blocks with padding"),
            (15, "just past two-block boundary"),
            (16, "two blocks exactly"),
            (32, "four blocks - previously caused index out of bounds"),
            (64, "eight blocks"),
            (128, "sixteen blocks"),
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
            // SHA256 of empty string
            c.populate_digest(
                &mut w,
                hex!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            );

            let result = circuit.populate_wire_witness(&mut w);
            assert!(
                result.is_ok(),
                "Failed for max_len={max_len} ({description}) with empty message: {result:?}"
            );
        }
    }

    #[test]
    fn test_sha256_to_le_wires() {
        let mut b = CircuitBuilder::new();
        let c = mk_circuit(&mut b, 64);

        // Obtain LE-packed wires for the digest wires
        let le_wires = c.digest_to_le_wires(&b);
        assert_eq!(le_wires.len(), 4);

        let circuit = b.build();
        let mut w = circuit.new_witness_filler();
        let message = b"abc";
        let hash = hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

        c.populate_len_bytes(&mut w, message.len());
        c.populate_message(&mut w, message);
        c.populate_digest(&mut w, hash);
        circuit.populate_wire_witness(&mut w).unwrap();

        // Extract the LE-packed bytes from the wires
        let mut le_bytes = Vec::with_capacity(32);
        for i in 0..4 {
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
        let hash = hex!("9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab");

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

    // Helper function for sha256_fixed tests
    fn test_sha256_fixed_with_input(message: &[u8], expected_bytes: [u8; 32]) {
        let b = CircuitBuilder::new();

        // Pack message into 32-bit words
        let n_words = message.len().div_ceil(4);
        let mut message_wires = Vec::new();

        for word_idx in 0..n_words {
            let mut packed = 0u32;
            for i in 0..4 {
                let byte_idx = word_idx * 4 + i;
                if byte_idx < message.len() {
                    packed |= (message[byte_idx] as u32) << (24 - i * 8);
                }
            }
            message_wires.push(b.add_constant(Word(packed as u64)));
        }

        // Create expected digest wires (8 32-bit words)
        let expected_digest_wires = array::from_fn::<_, 8, _>(|_| b.add_inout());

        // Compute the digest
        let computed_digest = sha256_fixed(&b, &message_wires, message.len());

        // Assert that computed digest equals expected digest
        for i in 0..8 {
            b.assert_eq(
                format!("digest[{}]", i),
                computed_digest[i],
                expected_digest_wires[i],
            );
        }

        let circuit = b.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        // Populate the expected digest wires
        for i in 0..8 {
            let mut word = 0u32;
            for j in 0..4 {
                word |= (expected_bytes[i * 4 + j] as u32) << (24 - j * 8);
            }
            w[expected_digest_wires[i]] = Word(word as u64);
        }

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    #[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(4) (2)")]
    fn test_sha256_fixed_with_insufficient_wires() {
        use super::sha256_fixed;
        let builder = CircuitBuilder::new();

        // Create only 1 wire but claim message is 5 bytes (which needs 2 wires)
        let message_wires: Vec<Wire> = vec![builder.add_witness()];

        // This should panic because message.len() (1) != len_bytes.div_ceil(4) (2)
        sha256_fixed(&builder, &message_wires, 5);
    }

    #[test]
    fn test_sha256_fixed_various_sizes() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // Test various message sizes to ensure padding works correctly
        let sizes = vec![
            0,   // empty
            1,   // single byte
            3,   // "abc" test vector
            4,   // exactly one word
            5,   // just over word boundary
            31,  // just under half block
            32,  // exactly half block
            33,  // just over half block
            55,  // max single block
            56,  // forces two blocks
            63,  // one byte from block boundary
            64,  // exactly one block
            65,  // just over one block
            119, // max two blocks
            120, // forces three blocks
            128, // exactly two blocks
            256, // exactly four blocks
        ];

        let mut rng = StdRng::seed_from_u64(0);

        for size in sizes {
            // Generate random payload
            let mut message = vec![0u8; size];
            rng.fill(&mut message[..]);

            // Compute expected hash using sha2 crate
            let expected = sha2::Sha256::digest(&message);
            let expected_bytes: [u8; 32] = expected.into();

            // Test with our circuit
            test_sha256_fixed_with_input(&message, expected_bytes);
        }
    }
}
