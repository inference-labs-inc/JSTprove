// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

/// Computes the RIPEMD-160 compression function on a single 512-bit block.
///
/// This function implements the core compression function of RIPEMD-160, which processes
/// a single 512-bit (64-byte) message block and updates the 160-bit internal state.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `state` - Current hash state as 5 wires, each containing a 32-bit word in the low 32 bits
///   (high 32 bits must be zero). The state words are in little-endian byte order.
/// * `message_block` - Message block as 16 wires, each containing a 32-bit word in the low 32 bits
///   (high 32 bits must be zero). The words are in little-endian byte order.
///
/// # Returns
/// * `[Wire; 5]` - Updated hash state as 5 wires, each containing a 32-bit word in the low 32 bits
///   (high 32 bits are zero). The state words are in little-endian byte order.
///
/// # Preconditions
/// * All input wires must have their high 32 bits set to zero (i.e., `wire & 0xFFFFFFFF == wire`)
/// * This is the caller's responsibility to ensure
///
/// # Implementation Notes
/// * RIPEMD-160 uses two parallel computation paths (left and right lines) that are combined
/// * Each line performs 80 rounds (5 groups of 16 rounds each)
/// * The final state is computed by adding the results of both lines to the input state
fn ripemd160_compress(
    builder: &CircuitBuilder,
    state: [Wire; 5],
    message_block: [Wire; 16],
) -> [Wire; 5] {
    // Constants for left line rounds
    const KL: [u32; 5] = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E];

    // Constants for right line rounds
    const KR: [u32; 5] = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000];

    // Message permutation for left line
    const ZL: [usize; 80] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // rounds 0-15
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8, // rounds 16-31
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, // rounds 32-47
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2, // rounds 48-63
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13, // rounds 64-79
    ];

    // Message permutation for right line
    const ZR: [usize; 80] = [
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, // rounds 0-15
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2, // rounds 16-31
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, // rounds 32-47
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14, // rounds 48-63
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11, // rounds 64-79
    ];

    // Rotation amounts for left line
    const SL: [u32; 80] = [
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, // rounds 0-15
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12, // rounds 16-31
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, // rounds 32-47
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12, // rounds 48-63
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6, // rounds 64-79
    ];

    // Rotation amounts for right line
    const SR: [u32; 80] = [
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, // rounds 0-15
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11, // rounds 16-31
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, // rounds 32-47
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8, // rounds 48-63
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11, // rounds 64-79
    ];

    // Initialize working variables for left line
    let mut al = state[0];
    let mut bl = state[1];
    let mut cl = state[2];
    let mut dl = state[3];
    let mut el = state[4];

    // Initialize working variables for right line
    let mut ar = state[0];
    let mut br = state[1];
    let mut cr = state[2];
    let mut dr = state[3];
    let mut er = state[4];

    // Process 80 rounds for each line
    for round_num in 0..80 {
        let round_group = round_num / 16;

        // Left line
        let func_l = match round_group {
            0 => f,
            1 => g,
            2 => h,
            3 => i,
            4 => j,
            _ => unreachable!(),
        };

        let new_bl = round(
            builder,
            al,
            bl,
            cl,
            dl,
            el,
            message_block[ZL[round_num]],
            SL[round_num],
            KL[round_group],
            func_l,
        );

        // Rotate variables for left line
        al = el;
        el = dl;
        dl = builder.rotl32(cl, 10);
        cl = bl;
        bl = new_bl;

        // Right line
        let func_r = match round_group {
            0 => j,
            1 => i,
            2 => h,
            3 => g,
            4 => f,
            _ => unreachable!(),
        };

        let new_br = round(
            builder,
            ar,
            br,
            cr,
            dr,
            er,
            message_block[ZR[round_num]],
            SR[round_num],
            KR[round_group],
            func_r,
        );

        // Rotate variables for right line
        ar = er;
        er = dr;
        dr = builder.rotl32(cr, 10);
        cr = br;
        br = new_br;
    }

    // Combine results: state[i] = state[i] + cl + dr (and appropriate permutation)
    [
        builder.iadd_32(builder.iadd_32(state[1], cl), dr),
        builder.iadd_32(builder.iadd_32(state[2], dl), er),
        builder.iadd_32(builder.iadd_32(state[3], el), ar),
        builder.iadd_32(builder.iadd_32(state[4], al), br),
        builder.iadd_32(builder.iadd_32(state[0], bl), cr),
    ]
}

// Selection function f(x, y, z) = x XOR y XOR z
fn f(b: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    b.bxor(b.bxor(x, y), z)
}

// Selection function g(x, y, z) = (x AND y) OR (NOT x AND z) = z XOR (x AND (y XOR z))
fn g(b: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    b.bxor(z, b.band(x, b.bxor(y, z)))
}

// Selection function h(x, y, z) = (x OR NOT y) XOR z
fn h(b: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    b.bxor(b.bor(x, b.bnot(y)), z)
}

// Selection function i(x, y, z) = (x AND z) OR (y AND NOT z) = y XOR (z AND (x XOR y))
fn i(b: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    b.bxor(y, b.band(z, b.bxor(x, y)))
}

// Selection function j(x, y, z) = x XOR (y OR NOT z)
fn j(b: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    b.bxor(x, b.bor(y, b.bnot(z)))
}

// RIPEMD-160 round function
#[allow(clippy::too_many_arguments)]
fn round(
    builder: &CircuitBuilder,
    a: Wire,
    b: Wire,
    c: Wire,
    d: Wire,
    e: Wire,
    x: Wire,
    s: u32,
    k: u32,
    func: fn(&CircuitBuilder, Wire, Wire, Wire) -> Wire,
) -> Wire {
    // T = A + func(B, C, D) + X + K
    let f_val = func(builder, b, c, d);
    let t1 = builder.iadd_32(a, f_val);
    let t2 = builder.iadd_32(t1, x);
    let t = builder.iadd_32(t2, builder.add_constant_64(k as u64));

    // T = (T << s) | (T >> (32 - s)) (rotate left by s)
    let t_rot = builder.rotl32(t, s);

    // T = T + E (this becomes the new B value)
    builder.iadd_32(t_rot, e)
}

/// Computes RIPEMD-160 hash of a fixed-length message.
///
/// This function creates a subcircuit that computes the RIPEMD-160 hash of a message
/// with a compile-time known length. Unlike a variable-length RIPEMD-160 implementation,
/// this function is optimized for fixed-length inputs where the length is known at
/// circuit construction time.
///
/// See [Pseudo-code for RIPEMD-160](https://homes.esat.kuleuven.be/~bosselae/ripemd/rmd160.txt)
/// for reference.
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `message` - Input message as 32-bit words (4 bytes per wire) in little-endian format. Each
///   wire must have the high 32 bits set to zero (precondition).
/// * `len_bytes` - The fixed length of the message in bytes (known at compile time)
///
/// # Returns
/// * `[Wire; 5]` - The RIPEMD-160 digest as 5 wires, each containing a 32-bit word in the low 32
///   bits (high 32 bits are zero) in little-endian order
///
/// # Panics
/// * If `message.len()` does not equal exactly `len_bytes.div_ceil(4)`
/// * If the message length in bits cannot fit in 32 bits
///
/// # Preconditions
/// * The caller must constrain the `message` wires to have their high 32 bits set to zero.
///
/// # Example
/// ```rust,ignore
/// use binius_circuits::ripemd::ripemd160_fixed;
/// use binius_frontend::compiler::CircuitBuilder;
///
/// let mut builder = CircuitBuilder::new();
///
/// // Create input wires for a 32-byte message (8 32-bit words)
/// let message: Vec<_> = (0..8).map(|_| builder.add_witness()).collect();
///
/// // Compute RIPEMD-160 of the 32-byte message
/// let digest = ripemd160_fixed(&builder, &message, 32);
/// ```
pub fn ripemd160_fixed(builder: &CircuitBuilder, message: &[Wire], len_bytes: usize) -> [Wire; 5] {
    // RIPEMD-160 initialization vector
    const H0: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];

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
    // RIPEMD-160 requires: message || 0x80 || zeros || 64-bit length field
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
        // For little-endian, the delimiter goes in the byte after the message
        let shift_amount = boundary_bytes * 8;
        let mask = builder.add_constant(Word((1u64 << shift_amount) - 1));
        let masked = builder.band(last_word, mask);

        // Add 0x80 delimiter at the right position (little-endian)
        let delimiter = builder.add_constant(Word(0x80u64 << shift_amount));
        let boundary_word = builder.bxor(masked, delimiter);

        padded_message.push(boundary_word);
    } else {
        // Message ends at word boundary - delimiter goes in new word
        padded_message.push(builder.add_constant(Word(0x80)));
    }

    // Fill with zeros until we reach the length field position
    let zero = builder.add_constant(Word::ZERO);
    padded_message.resize(n_padded_words - 2, zero);

    // Add the length field (64 bits total, little-endian)
    let bitlen = (len_bytes as u64) * 8;
    padded_message.push(builder.add_constant(Word(bitlen & 0xFFFFFFFF))); // Low 32 bits
    padded_message.push(builder.add_constant(Word(bitlen >> 32))); // High 32 bits (always 0 for us)

    // Initialize state with RIPEMD-160 IV
    let initial_state = [
        builder.add_constant(Word(H0[0] as u64)),
        builder.add_constant(Word(H0[1] as u64)),
        builder.add_constant(Word(H0[2] as u64)),
        builder.add_constant(Word(H0[3] as u64)),
        builder.add_constant(Word(H0[4] as u64)),
    ];

    // Process compression blocks
    padded_message
        .chunks_exact(16)
        .enumerate()
        .fold(initial_state, |state, (block_idx, block)| {
            let block_message: [Wire; 16] = block
                .try_into()
                .expect("length of padded_message is a multiple of 16 by construction");

            ripemd160_compress(
                &builder.subcircuit(format!("ripemd160_compress[{block_idx}]")),
                state,
                block_message,
            )
        })
}

#[cfg(test)]
mod tests {
    use std::{array, iter, iter::repeat_with};

    use binius_core::verify::verify_constraints;
    use rand::prelude::*;
    use ripemd::Digest;

    use super::*;

    // Helper function for ripemd160_fixed tests
    fn test_ripemd160_fixed_with_input(message: &[u8], expected_bytes: [u8; 20]) {
        let b = CircuitBuilder::new();

        // Pack message into 32-bit words (little-endian)
        let n_words = message.len().div_ceil(4);

        let message_wires = repeat_with(|| b.add_inout())
            .take(n_words)
            .collect::<Vec<_>>();

        // Create expected digest wires (5 32-bit words)
        let expected_digest_wires = array::from_fn::<_, 5, _>(|_| b.add_inout());

        // Compute the digest
        assert_eq!(message_wires.len(), message.len().div_ceil(4));
        assert_eq!(message_wires.len(), n_words);
        let computed_digest = ripemd160_fixed(&b, &message_wires, message.len());

        // Assert that computed digest equals expected digest
        for i in 0..5 {
            b.assert_eq(
                format!("digest[{i}]"),
                computed_digest[i],
                expected_digest_wires[i],
            );
        }

        let circuit = b.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        // Populate the message wires (little-endian)
        for (&wire, chunk) in iter::zip(&message_wires, message.chunks(4)) {
            let mut word_bytes = [0u8; 4];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            let word = u32::from_le_bytes(word_bytes);
            w[wire] = Word(word as u64);
        }

        // Populate the expected digest wires (little-endian)
        for (&wire, chunk) in iter::zip(&expected_digest_wires, expected_bytes.chunks(4)) {
            let mut word_bytes = [0u8; 4];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            let word = u32::from_le_bytes(word_bytes);
            w[wire] = Word(word as u64);
        }

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_ripemd160_fixed_various_sizes() {
        let mut rng = StdRng::seed_from_u64(0);

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

        for size in sizes {
            // Generate random payload
            let mut message = vec![0u8; size];
            rng.fill(&mut message[..]);

            // Compute expected hash using ripemd crate
            let expected = ripemd::Ripemd160::digest(&message);
            let expected_bytes: [u8; 20] = expected.into();

            // Test with our circuit
            test_ripemd160_fixed_with_input(&message, expected_bytes);
        }
    }

    #[test]
    #[should_panic(expected = "message.len() (1) must equal len_bytes.div_ceil(4) (2)")]
    fn test_ripemd160_fixed_with_insufficient_wires() {
        let builder = CircuitBuilder::new();

        // Create only 1 wire but claim message is 5 bytes (which needs 2 wires)
        let message_wires: Vec<Wire> = vec![builder.add_witness()];

        // This should panic because message.len() (1) != len_bytes.div_ceil(4) (2)
        ripemd160_fixed(&builder, &message_wires, 5);
    }
}
