// Copyright 2025 Irreducible Inc.
mod constants;
#[cfg(test)]
mod reference;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use constants::{C240, R512, TWEAK_TYPE_CFG, TWEAK_TYPE_MSG, TWEAK_TYPE_OUT};

/// Skein512 circuit that processes a fixed number of 64-byte message blocks
///
/// This circuit implements the full Skein-512 hash function for fixed-length inputs,
/// following the reference implementation. Unlike variable-length hash circuits,
/// this expects exactly n_blocks of 64 bytes each as input.
///
/// # Circuit Structure
///
/// 1. **Configuration Block**: Process fixed config block to establish initial chaining value
/// 2. **Message Blocks**: Process each of the n_blocks message blocks in sequence
/// 3. **Output Block**: Generate final 512-bit digest
///
/// Each stage uses UBI (Unique Block Iteration) compression with appropriate tweak values.
///
/// # Arguments
///
/// * `n_blocks` - Number of 64-byte message blocks this circuit will process
/// * `message` - Vector of message block wires, each containing 8 × 64-bit words
/// * `digest` - Array of 8 wires representing the 512-bit output digest
pub struct Skein512 {
    /// Number of 64-byte blocks this circuit processes
    pub n_blocks: usize,
    /// Input message as blocks of 8 × 64-bit words each
    pub message: Vec<[Wire; 8]>,
    /// Expected 512-bit digest as 8 × 64-bit words
    pub digest: [Wire; 8],
}

impl Skein512 {
    pub fn new(builder: &CircuitBuilder, n_blocks: usize) -> Self {
        assert!(n_blocks > 0, "n_blocks must be positive");

        // Create message block wires - each block is 8 × 64-bit words
        let message: Vec<[Wire; 8]> = (0..n_blocks)
            .map(|_| std::array::from_fn(|_| builder.add_witness()))
            .collect();

        // Create digest wires - 8 × 64-bit words for 512-bit output
        let digest: [Wire; 8] = std::array::from_fn(|_| builder.add_witness());

        // Build the circuit following Skein-512 algorithm
        Self::build_circuit(builder, &message, digest, n_blocks);

        Self {
            n_blocks,
            message,
            digest,
        }
    }

    fn build_circuit(
        builder: &CircuitBuilder,
        message: &[[Wire; 8]],
        digest: [Wire; 8],
        n_blocks: usize,
    ) {
        // ---- Stage 1: Configuration UBI Block ----
        //
        // Process the fixed Skein-512 configuration block to establish initial chaining value.
        // Config block contains: "SHA3", version, output length (512 bits), etc.

        // Build 64-byte configuration block (same as reference implementation)
        let mut cfg = [0u8; 64];
        cfg[0..4].copy_from_slice(&0x3341_4853u32.to_le_bytes()); // "SHA3"
        cfg[4..6].copy_from_slice(&1u16.to_le_bytes()); // version 1
        cfg[6..8].copy_from_slice(&0u16.to_le_bytes()); // reserved
        cfg[8..16].copy_from_slice(&512u64.to_le_bytes()); // output length in bits
        cfg[16] = 0; // Yl
        cfg[17] = 0; // Yf
        cfg[18] = 0; // Ym
        // rest already zero (bytes 19..31 and 32..63)

        // Convert config bytes to 8 × 64-bit words
        let cfg_words: [Wire; 8] = std::array::from_fn(|i| {
            let mut word_bytes = [0u8; 8];
            word_bytes.copy_from_slice(&cfg[i * 8..(i + 1) * 8]);
            let word_val = u64::from_le_bytes(word_bytes);
            builder.add_constant_64(word_val)
        });

        // Create configuration tweak: T_TYPE_CFG, position=32, FIRST|FINAL
        let pos_lo = builder.add_constant_64(32);
        let pos_hi = builder.add_constant_64(0);
        let (t_low, t_high) = tweak(builder, pos_lo, pos_hi, true, true, TWEAK_TYPE_CFG);
        let t_cfg_wires = [t_low, t_high];

        // Initial chaining value is zero for Skein config UBI
        let cv0: [Wire; 8] = std::array::from_fn(|_| builder.add_constant_64(0));

        // Config UBI: CV1 = UBI(CV0=0, T_cfg, config_block)
        let config_ubi_out = ubi_block(builder, cv0, t_cfg_wires, cfg_words);

        // ---- Stage 2: Message UBI Blocks ----
        //
        // Process each message block with appropriate tweak values

        let mut current_cv = config_ubi_out;

        // Process each message block
        for block_idx in 0..n_blocks {
            // Calculate position: number of bytes processed so far INCLUDING this block
            let pos_end = ((block_idx + 1) * 64) as u64;

            // Tweak flags - for the reference semantics, no message block is final except the empty
            // one
            let is_first = block_idx == 0;
            let is_final = false; // Message blocks are never final in reference implementation

            // Create message tweak: T_TYPE_MSG, position, first/final flags
            let pos_lo = builder.add_constant_64(pos_end);
            let pos_hi = builder.add_constant_64(0); // high 32 bits of position (we assume < 4GB messages)
            let (t_low, t_high) =
                tweak(builder, pos_lo, pos_hi, is_first, is_final, TWEAK_TYPE_MSG);
            let t_msg_wires = [t_low, t_high];

            // Message UBI: CV_{i+1} = UBI(CV_i, T_msg, message_block_i)
            current_cv = ubi_block(builder, current_cv, t_msg_wires, message[block_idx]);
        }

        // ---- Final Message Block (Empty) ----
        //
        // According to reference implementation, we always need a final empty message block
        // when the message length is a multiple of 64 bytes

        // Calculate final position (total message length)
        let final_pos = (n_blocks * 64) as u64;

        // Create empty final block (all zeros)
        let empty_block: [Wire; 8] = std::array::from_fn(|_| builder.add_constant_64(0));

        // Create final message tweak: first=false, final=true
        let pos_lo_final = builder.add_constant_64(final_pos);
        let pos_hi_final = builder.add_constant_64(0);
        let (t_low, t_high) = tweak(
            builder,
            pos_lo_final,
            pos_hi_final,
            false,
            true, // first=false, final=true
            TWEAK_TYPE_MSG,
        );
        let t_msg_final_wires = [t_low, t_high];

        // Final message UBI with empty block
        current_cv = ubi_block(builder, current_cv, t_msg_final_wires, empty_block);

        // ---- Stage 3: Output UBI Block ----
        //
        // Generate final digest by processing counter block

        // Output block: 8 bytes counter (0) + 56 bytes zero padding
        // Match reference implementation: bytes first, then convert to words
        let mut out_bytes = [0u8; 64];
        out_bytes[0..8].copy_from_slice(&0u64.to_le_bytes()); // counter=0 in first 8 bytes, little-endian
        // rest already zero (bytes 8..63)

        // Convert to words using the same byte-to-word conversion as reference
        let out_block: [Wire; 8] = std::array::from_fn(|i| {
            let mut word_bytes = [0u8; 8];
            word_bytes.copy_from_slice(&out_bytes[i * 8..(i + 1) * 8]);
            let word_val = u64::from_le_bytes(word_bytes);
            builder.add_constant_64(word_val)
        });

        // Create output tweak: T_TYPE_OUT, position=8 (counter size), FIRST|FINAL
        let pos_lo = builder.add_constant_64(8);
        let pos_hi = builder.add_constant_64(0);
        let (t_low, t_high) = tweak(builder, pos_lo, pos_hi, true, true, TWEAK_TYPE_OUT);
        let t_out_wires = [t_low, t_high];

        // Output UBI: final_digest = UBI(CV_final, T_out, output_block)
        let computed_digest = ubi_block(builder, current_cv, t_out_wires, out_block);

        // ---- Stage 4: Digest Verification ----
        //
        // Verify that computed digest matches expected digest

        builder.assert_eq_v("skein512_digest", computed_digest, digest);
    }

    /// Populate the message wires with input message blocks
    pub fn populate_message(&self, w: &mut WitnessFiller<'_>, message_blocks: &[[u8; 64]]) {
        assert_eq!(
            message_blocks.len(),
            self.n_blocks,
            "Message blocks length {} != expected {}",
            message_blocks.len(),
            self.n_blocks
        );

        // Pack each 64-byte block into 8 × 64-bit words (little-endian)
        for (block_idx, block_bytes) in message_blocks.iter().enumerate() {
            for (word_idx, word_bytes) in block_bytes.chunks(8).enumerate() {
                let mut padded_bytes = [0u8; 8];
                padded_bytes[..word_bytes.len()].copy_from_slice(word_bytes);
                let word_val = u64::from_le_bytes(padded_bytes);
                w[self.message[block_idx][word_idx]] = Word(word_val);
            }
        }
    }

    /// Populate the digest wire with expected hash output
    pub fn populate_digest(&self, w: &mut WitnessFiller<'_>, expected_digest: [u8; 64]) {
        // Pack 64 bytes into 8 × 64-bit words (little-endian)
        for (i, word_bytes) in expected_digest.chunks(8).enumerate() {
            let word_val = u64::from_le_bytes(word_bytes.try_into().unwrap());
            w[self.digest[i]] = Word(word_val);
        }
    }

    /// Get the digest wires
    pub fn digest_wires(&self) -> [Wire; 8] {
        self.digest
    }

    /// Get the message block wires
    pub fn message_wires(&self) -> &[[Wire; 8]] {
        &self.message
    }
}

/// Mix component for Skein-512 hash function
///
/// Performs the Threefish MIX operation, which is the core building block of Threefish rounds.
///
/// The MIX algorithm is:
/// - a' = a + b (64-bit addition)
/// - b' = ROTL(b, r) ^ a' (rotate left b by r bits, then XOR with new a)
///
/// Where:
/// - a, b: input 64-bit words
/// - r: rotation amount (compile-time constant, 0-63)
/// - a', b': output 64-bit words
///
/// This operation provides both diffusion (through rotation) and confusion (through XOR)
/// while maintaining the avalanche property essential for cryptographic security.
fn mix(circuit: &CircuitBuilder, a: Wire, b: Wire, r: u32) -> (Wire, Wire) {
    // a' = a + b (64-bit addition, ignoring carry)
    let zero = circuit.add_constant_64(0);
    let (a_out, _) = circuit.iadd_cin_cout(a, b, zero);

    // b' = ROTL(b, r) ^ a'
    let b_rotated = circuit.rotl(b, r);
    let b_out = circuit.bxor(b_rotated, a_out);

    (a_out, b_out)
}

/// Permute512 component for Skein-512 hash function
///
/// Performs the Threefish-512 word permutation used between rounds.
///
/// The permutation algorithm is a simple rearrangement of 8 words:
/// - output\[i\] = input\[π\[i\]\] where π = \[2, 1, 4, 7, 6, 5, 0, 3\]
///
/// This corresponds to the Threefish permutation from Table 3 of the Skein specification
/// for Nw=8 (8 words). The permutation is applied after the MIX operations in each round
/// to ensure proper diffusion across the state.
fn permute_512(_circuit: &CircuitBuilder, x: [Wire; 8]) -> [Wire; 8] {
    [x[2], x[1], x[4], x[7], x[6], x[5], x[0], x[3]]
}

/// ThreefishRound component for Skein-512 hash function
///
/// Performs one round of the Threefish-512 block cipher, which consists of:
/// 1. Four parallel MIX operations on word pairs (0,1), (2,3), (4,5), (6,7)
/// 2. Word permutation to ensure proper diffusion
///
/// Each round uses round-specific rotation constants from the R512 table.
/// Threefish-512 uses 72 rounds total, with the rotation pattern repeating every 8 rounds.
///
/// The round algorithm:
/// - Apply MIX(v\[0\], v\[1\], r\[0\]) → (v0', v1')
/// - Apply MIX(v\[2\], v\[3\], r\[1\]) → (v2', v3')
/// - Apply MIX(v\[4\], v\[5\], r\[2\]) → (v4', v5')
/// - Apply MIX(v\[6\], v\[7\], r\[3\]) → (v6', v7')
/// - Apply permutation π to [v0', v1', v2', v3', v4', v5', v6', v7']
fn threefish_round(circuit: &CircuitBuilder, v_in: [Wire; 8], round_idx: usize) -> [Wire; 8] {
    // Get rotation constants for this round (pattern repeats every 8 rounds)
    let r = R512[round_idx % 8];

    // Apply 4 parallel MIX operations on word pairs
    // MIX pairs for 512-bit: (0,1), (2,3), (4,5), (6,7)
    let (mix0_a, mix0_b) = mix(circuit, v_in[0], v_in[1], r[0]);
    let (mix1_a, mix1_b) = mix(circuit, v_in[2], v_in[3], r[1]);
    let (mix2_a, mix2_b) = mix(circuit, v_in[4], v_in[5], r[2]);
    let (mix3_a, mix3_b) = mix(circuit, v_in[6], v_in[7], r[3]);

    // Reassemble state from MIX outputs
    let mixed_state = [
        mix0_a, mix0_b, // (v0, v1) from MIX(v[0], v[1])
        mix1_a, mix1_b, // (v2, v3) from MIX(v[2], v[3])
        mix2_a, mix2_b, // (v4, v5) from MIX(v[4], v[5])
        mix3_a, mix3_b, // (v6, v7) from MIX(v[6], v[7])
    ];

    // Apply permutation to complete the round
    permute_512(circuit, mixed_state)
}

/// ThreefishSubkey component for Skein-512 hash function
///
/// Generates subkeys for the Threefish-512 block cipher rounds.
///
/// The subkey generation algorithm is:
/// - sk\[i\] = k\[(s + i) % 9\] for i in 0..8 (base subkey from extended key)
/// - sk\[5\] += t\[s % 3\] (add tweak component)
/// - sk\[6\] += t\[(s + 1) % 3\] (add next tweak component)
/// - sk\[7\] += s (add round number)
///
/// Where:
/// - s: subkey/round index (0-18 for 72-round Threefish-512)
/// - k: extended key array [k0, k1, ..., k7, k8] where k8 = C240 ^ (k0 ^ k1 ^ ... ^ k7)
/// - t: extended tweak array [t0, t1, t2] where t2 = t0 ^ t1
fn threefish_subkey(circuit: &CircuitBuilder, s: usize, k: [Wire; 9], t: [Wire; 3]) -> [Wire; 8] {
    // Create base subkey by rotating through extended key
    let mut subkey = std::array::from_fn(|i| k[(s + i) % 9]);

    // Add tweak components to specific positions
    // sk[5] += t[s % 3] (64-bit addition, ignoring carry)
    let zero = circuit.add_constant_64(0);
    let (sum5, _) = circuit.iadd_cin_cout(subkey[5], t[s % 3], zero);
    subkey[5] = sum5;

    // sk[6] += t[(s + 1) % 3] (64-bit addition, ignoring carry)
    let (sum6, _) = circuit.iadd_cin_cout(subkey[6], t[(s + 1) % 3], zero);
    subkey[6] = sum6;

    // sk[7] += s (add round number as constant)
    let round_constant = circuit.add_constant_64(s as u64);
    let (sum7, _) = circuit.iadd_cin_cout(subkey[7], round_constant, zero);
    subkey[7] = sum7;

    subkey
}

/// Tweak component for Skein-512 hash function
///
/// The tweak is a 128-bit value used in the Threefish block cipher that underlies Skein.
/// It encodes position information, type flags, and first/final block indicators.
///
/// According to the Skein specification, the tweak bit layout is:
/// - Bits 0-95: Position (number of bytes processed so far, including this block)
/// - Bits 96-111: Reserved (must be zero)
/// - Bits 112-118: Tree level (0 for sequential hashing)
/// - Bit 119: Bit pad flag (0 for byte-aligned messages)
/// - Bits 120-125: Type field (CFG=4, MSG=48, OUT=63)
/// - Bit 126: First block flag
/// - Bit 127: Final block flag
fn tweak(
    circuit: &CircuitBuilder,
    pos_bytes_lo: Wire,
    mut pos_bytes_hi: Wire,
    is_first: bool,
    is_final: bool,
    cfg: u64,
) -> (Wire, Wire) {
    let low_bytes_mask = circuit.add_constant_64(u32::MAX as u64);
    pos_bytes_hi = circuit.band(pos_bytes_hi, low_bytes_mask);

    let t_low = pos_bytes_lo;
    let mut t_high = circuit.bor(pos_bytes_hi, circuit.add_constant_64(cfg << 56));

    if is_first {
        t_high = circuit.bor(t_high, circuit.add_constant_64(1 << 62));
    }

    if is_final {
        t_high = circuit.bor(t_high, circuit.add_constant_64(1 << 63));
    }

    (t_low, t_high)
}

/// UbiBlock component for Skein-512 hash function
///
/// Implements the UBI (Unique Block Iteration) compression for Skein-512.
///
/// Inputs:
/// - chaining_value: [Wire; 8] (input chaining value)
/// - tweak: [Wire; 2] (tweak words)
/// - block: [Wire; 8] (message block)
///
/// Outputs:
/// - g_out: [Wire; 8] (output chaining value)
fn ubi_block(
    circuit: &CircuitBuilder,
    chaining_value: [Wire; 8],
    tweak: [Wire; 2],
    block: [Wire; 8],
) -> [Wire; 8] {
    // G' = Threefish(K = CV, T = tweak, M = block) XOR M
    let out = Threefish512Block::new(circuit, chaining_value, tweak, block).v_out;
    std::array::from_fn(|i| circuit.bxor(out[i], block[i]))
}

/// Threefish4RoundsWithInjection component for Skein-512 hash function
///
/// Performs a group of 4 Threefish-512 rounds with subkey injection before the group.
/// This matches the reference implementation's `threefish_4rounds_with_injection`.
///
/// - Inject subkey (computed from extended key and tweak) before the 4 rounds
/// - Run 4 consecutive Threefish rounds (using round indices base..base+3)
///
/// Inputs:
/// - v_in: [Wire; 8] (input state)
/// - k: [Wire; 9] (extended key)
/// - t: [Wire; 3] (extended tweak)
/// - group_idx: usize (group index, 0..=18 for 72 rounds)
///
/// Outputs:
/// - v_out: [Wire; 8] (output state after 4 rounds and subkey injection)
struct Threefish4RoundsWithInjection {
    v_out: [Wire; 8],
}

impl Threefish4RoundsWithInjection {
    fn new(
        circuit: &CircuitBuilder,
        v_in: [Wire; 8],
        k: [Wire; 9],
        t: [Wire; 3],
        group_idx: usize,
    ) -> Self {
        // Inject subkey before the 4 rounds
        let subkey = threefish_subkey(circuit, group_idx, k, t);
        let zero = circuit.add_constant_64(0);
        let mut v_out = std::array::from_fn(|i| {
            let (sum, _) = circuit.iadd_cin_cout(v_in[i], subkey[i], zero);
            sum
        });

        // Do 4 rounds
        let base = group_idx * 4;
        for round in 0..4 {
            v_out = threefish_round(circuit, v_out, base + round);
        }

        Self { v_out }
    }
}

/// Threefish512Block component for Skein-512 hash function
///
/// Implements the full Threefish-512 block function as in the reference implementation.
///
/// Inputs:
/// - key: [Wire; 8] (key words)
/// - tweak: [Wire; 2] (tweak words)
/// - block: [Wire; 8] (plaintext/message block)
///
/// Outputs:
/// - v_out: [Wire; 8] (ciphertext/output block)
struct Threefish512Block {
    v_out: [Wire; 8],
}

impl Threefish512Block {
    fn new(circuit: &CircuitBuilder, key: [Wire; 8], tweak: [Wire; 2], block: [Wire; 8]) -> Self {
        // Expand key to 9 words: k[8] = C240 ^ (k0 ^ ... ^ k7)
        let c240 = circuit.add_constant_64(C240);
        let mut k_vec = Vec::with_capacity(9);
        let mut sum = key[0];
        k_vec.push(key[0]);
        for i in 1..8 {
            k_vec.push(key[i]);
            sum = circuit.bxor(sum, key[i]);
        }
        k_vec.push(circuit.bxor(c240, sum));
        let k: [Wire; 9] = k_vec.try_into().expect("Vec to array conversion");

        // Expand tweak to 3 words: t2 = t0 ^ t1
        let t0 = tweak[0];
        let t1 = tweak[1];
        let t2 = circuit.bxor(t0, t1);
        let t = [t0, t1, t2];

        // Initial state is the plaintext/message block
        let mut v = block;

        // 72 rounds = 18 groups of 4 rounds, with subkey injections
        for g in 0..18 {
            let group = Threefish4RoundsWithInjection::new(circuit, v, k, t, g);
            v = group.v_out;
        }

        // Final subkey injection (18th injection after the 72 rounds)
        let subkey = threefish_subkey(circuit, 18, k, t);
        let zero = circuit.add_constant_64(0);
        let v_out = std::array::from_fn(|i| {
            let (sum, _) = circuit.iadd_cin_cout(v[i], subkey[i], zero);
            sum
        });

        Self { v_out }
    }
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::CircuitBuilder;

    use super::*;
    use crate::skein512::reference;

    // Tests from skein512.rs
    fn test_skein512_with_blocks(message_blocks: &[[u8; 64]]) {
        let n_blocks = message_blocks.len();
        let builder = CircuitBuilder::new();

        let skein = Skein512::new(&builder, n_blocks);
        let circuit = builder.build();

        let expected_digest = reference::skein512(message_blocks.as_flattened());

        let mut w = circuit.new_witness_filler();
        skein.populate_message(&mut w, message_blocks);
        skein.populate_digest(&mut w, expected_digest);

        circuit.populate_wire_witness(&mut w).unwrap();
        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_skein512_single_block() {
        let mut block = [0u8; 64];
        block[0..3].copy_from_slice(b"abc");

        test_skein512_with_blocks(&[block]);
    }

    #[test]
    fn test_skein512_64_byte_zeros() {
        test_skein512_with_blocks(&[[0u8; 64]]);
    }

    #[test]
    fn test_skein512_two_blocks() {
        let mut block1 = [0xAAu8; 64];
        let mut block2 = [0x55u8; 64];

        block1[0] = 0x01;
        block2[63] = 0xFF;

        test_skein512_with_blocks(&[block1, block2]);
    }

    #[test]
    fn test_skein512_multiple_blocks() {
        // Test with 4 blocks of different patterns
        let blocks = [
            [0x00u8; 64], // all zeros
            [0xFFu8; 64], // all ones
            {
                let mut block = [0u8; 64];
                for (i, byte) in block.iter_mut().enumerate() {
                    *byte = (i % 256) as u8;
                }
                block
            },
            {
                let mut block = [0u8; 64];
                block[0..26].copy_from_slice(b"abcdefghijklmnopqrstuvwxyz");
                block
            },
        ];

        test_skein512_with_blocks(&blocks);
    }

    // Tests from mix.rs
    #[test]
    fn test_mix_correctness() {
        let test_cases = [
            // (a, b, r, description)
            (
                0x0123456789ABCDEFu64,
                0xFEDCBA9876543210u64,
                0,
                "Zero rotation",
            ),
            (
                0x0123456789ABCDEFu64,
                0xFEDCBA9876543210u64,
                1,
                "Single bit rotation",
            ),
            (
                0x0123456789ABCDEFu64,
                0xFEDCBA9876543210u64,
                8,
                "Byte rotation",
            ),
            (
                0x0123456789ABCDEFu64,
                0xFEDCBA9876543210u64,
                32,
                "Half-word rotation",
            ),
            (0u64, 0u64, 0, "Both zero, no rotation"),
            (u64::MAX, u64::MAX, 0, "Both max, no rotation"),
            (0u64, u64::MAX, 32, "Zero and max"),
            (u64::MAX, 0u64, 32, "Max and zero"),
            (
                0x8000000000000000u64,
                0x0000000000000001u64,
                1,
                "MSB and LSB",
            ),
            (
                0x0000000000000001u64,
                0x8000000000000000u64,
                16,
                "LSB and MSB with rotation",
            ),
        ];

        for (a_val, b_val, r, description) in test_cases {
            let (expected_a, expected_b) = reference::mix(a_val, b_val, r);

            let circuit = CircuitBuilder::new();

            let a_wire = circuit.add_witness();
            let b_wire = circuit.add_witness();

            let (a_out, b_out) = mix(&circuit, a_wire, b_wire, r);

            let expected_a_wire = circuit.add_constant(Word(expected_a));
            let expected_b_wire = circuit.add_constant(Word(expected_b));

            circuit.assert_eq(format!("{}_a_out", description), a_out, expected_a_wire);
            circuit.assert_eq(format!("{}_b_out", description), b_out, expected_b_wire);

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            witness[a_wire] = Word(a_val);
            witness[b_wire] = Word(b_val);

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();

            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    // Tests from permute_512.rs
    #[test]
    fn test_permute512_correctness() {
        let test_cases = [
            // (input_vals, description)
            (
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Sequential pattern",
            ),
            ([0xAAAAAAAAAAAAAAAAu64; 8], "All same values"),
            (
                [0u64, 1u64, 2u64, 3u64, 4u64, 5u64, 6u64, 7u64],
                "Simple incremental",
            ),
            (
                [
                    u64::MAX,
                    0u64,
                    u64::MAX,
                    0u64,
                    u64::MAX,
                    0u64,
                    u64::MAX,
                    0u64,
                ],
                "Alternating pattern",
            ),
        ];

        for (input_vals, description) in test_cases {
            let expected = reference::permute_512(input_vals);

            let circuit = CircuitBuilder::new();

            let input_wires = std::array::from_fn(|_| circuit.add_witness());

            let permuted = permute_512(&circuit, input_wires);

            for (i, &expected_val) in expected.iter().enumerate() {
                let expected_wire = circuit.add_constant(Word(expected_val));
                circuit.assert_eq(
                    format!("{}[{}]", description, i),
                    permuted[i],
                    expected_wire,
                );
            }

            let built_circuit = circuit.build();

            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in input_vals.iter().enumerate() {
                witness[input_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();

            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    // Tests from threefish_round.rs
    #[test]
    fn test_threefish_round_correctness() {
        // Test various round indices with different input states
        let test_cases = [
            // (round_idx, input_state, description)
            (
                0,
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Round 0 with sequential pattern",
            ),
            (
                1,
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Round 1 with sequential pattern",
            ),
            (
                7,
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Round 7 with sequential pattern",
            ),
            (
                8,
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Round 8 (same as round 0)",
            ),
        ];

        for (round_idx, input_state, description) in test_cases {
            let expected = reference::threefish_round(input_state, round_idx);

            let circuit = CircuitBuilder::new();

            let input_wires = std::array::from_fn(|_| circuit.add_witness());

            let v_out = threefish_round(&circuit, input_wires, round_idx);

            for i in 0..8 {
                let expected_wire = circuit.add_constant(Word(expected[i]));
                circuit.assert_eq(format!("{}[{}]", description, i), v_out[i], expected_wire);
            }

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in input_state.iter().enumerate() {
                witness[input_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();

            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    // Tests from threefish_subkey.rs
    #[test]
    fn test_threefish_subkey_correctness() {
        let test_cases = [
            // (s, description)
            (0, "First subkey (s=0)"),
            (1, "Second subkey (s=1)"),
            (5, "Middle subkey (s=5)"),
            (9, "Wrap-around subkey (s=9)"),
            (18, "Final subkey (s=18)"),
        ];

        for (s, description) in test_cases {
            let k_vals = [
                0x0123456789ABCDEFu64,
                0xFEDCBA9876543210u64,
                0x1111111111111111u64,
                0x2222222222222222u64,
                0x3333333333333333u64,
                0x4444444444444444u64,
                0x5555555555555555u64,
                0x6666666666666666u64,
                0x7777777777777777u64, // k8 (extended key)
            ];

            let t_vals = [
                0xAAAAAAAAAAAAAAAAu64,
                0xBBBBBBBBBBBBBBBBu64,
                0x1111111111111111u64, // t2 = t0 ^ t1
            ];

            let expected = reference::threefish_subkey(s, k_vals, t_vals);

            let circuit = CircuitBuilder::new();

            let k_wires = std::array::from_fn(|_| circuit.add_witness());
            let t_wires = std::array::from_fn(|_| circuit.add_witness());

            let subkey = threefish_subkey(&circuit, s, k_wires, t_wires);

            for (i, &expected_val) in expected.iter().enumerate() {
                let expected_wire = circuit.add_constant(Word(expected_val));
                circuit.assert_eq(format!("{}[{}]", description, i), subkey[i], expected_wire);
            }

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in k_vals.iter().enumerate() {
                witness[k_wires[i]] = Word(val);
            }
            for (i, &val) in t_vals.iter().enumerate() {
                witness[t_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();

            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    #[test]
    fn test_threefish_subkey_edge_cases() {
        let test_cases = [
            // (k_pattern, t_pattern, description)
            ([0u64; 9], [0u64; 3], "All zeros"),
            ([u64::MAX; 9], [u64::MAX; 3], "All ones"),
            (
                [0xDEADBEEFCAFEBABEu64; 9],
                [0x0123456789ABCDEFu64; 3],
                "Mixed pattern",
            ),
        ];

        for (k_vals, t_vals, description) in test_cases {
            let circuit = CircuitBuilder::new();

            let k_wires = std::array::from_fn(|_| circuit.add_witness());
            let t_wires = std::array::from_fn(|_| circuit.add_witness());

            let s = 9;
            let subkey = threefish_subkey(&circuit, s, k_wires, t_wires);
            let expected = reference::threefish_subkey(s, k_vals, t_vals);

            for i in 0..8 {
                circuit.assert_eq(
                    format!("{}[{}]", description, i),
                    subkey[i],
                    circuit.add_constant(Word(expected[i])),
                );
            }

            let built_circuit = circuit.build();
            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in k_vals.iter().enumerate() {
                witness[k_wires[i]] = Word(val);
            }
            for (i, &val) in t_vals.iter().enumerate() {
                witness[t_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();
            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    // Tests from tweak.rs
    #[test]
    fn test_tweak_correctness() {
        let test_cases = [
            // (position, type_code, is_first, is_final, description)
            (0u128, false, false, "CFG: neither first nor final"),
            (0u128, true, false, "CFG: first only"),
            (0u128, false, true, "CFG: final only"),
            (0u128, true, true, "CFG: both first and final"),
            (64u128, false, false, "MSG 64 bytes: neither"),
            (64u128, true, false, "MSG 64 bytes: first only"),
            (64u128, false, true, "MSG 64 bytes: final only"),
            (64u128, true, true, "MSG 64 bytes: both"),
            (128u128, false, false, "OUT 128 bytes: neither"),
            (128u128, true, false, "OUT 128 bytes: first only"),
            (128u128, false, true, "OUT 128 bytes: final only"),
            (128u128, true, true, "OUT 128 bytes: both"),
            // Test with larger position values
            (
                0x123456789ABCDEu128,
                true,
                false,
                "MSG large pos: first only",
            ),
            (
                0x123456789ABCDEu128,
                false,
                true,
                "MSG large pos: final only",
            ),
        ];

        fn test_tweak_inner(
            pos_bytes: u128,
            is_first: bool,
            is_final: bool,
            description: &str,
            cfg: u64,
        ) {
            let expected = reference::tweak(
                cfg,
                pos_bytes as u64,
                (pos_bytes >> 64) as u64,
                is_first,
                is_final,
            );

            let circuit = CircuitBuilder::new();
            let pos_t0 = circuit.add_witness();
            let pos_t1 = circuit.add_witness();

            let (t_low, t_high) = tweak(&circuit, pos_t0, pos_t1, is_first, is_final, cfg);

            let expected_t0 = circuit.add_constant(Word(expected[0]));
            let expected_t1 = circuit.add_constant(Word(expected[1]));

            circuit.assert_eq(format!("{}_t_low", description), t_low, expected_t0);
            circuit.assert_eq(format!("{}_t_high", description), t_high, expected_t1);

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            witness[pos_t0] = Word(pos_bytes as u64);
            witness[pos_t1] = Word((pos_bytes >> 64) as u64);

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();

            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }

        for (pos_bytes, is_first, is_final, description) in test_cases {
            test_tweak_inner(pos_bytes, is_first, is_final, description, TWEAK_TYPE_OUT);
        }
    }

    // Tests from ubi_block.rs
    #[test]
    fn test_ubi_block_correctness() {
        // Test various chaining value, tweak, and block combinations
        let test_cases = [
            // (chaining_value, tweak, block, description)
            (
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                [0xAAAAAAAAAAAAAAAAu64, 0xBBBBBBBBBBBBBBBBu64],
                [
                    0x0F0E0D0C0B0A0908u64,
                    0x0706050403020100u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Basic pattern",
            ),
            (
                [0u64; 8],
                [0u64; 2],
                [u64::MAX; 8],
                "All zeros chaining/tweak, all ones block",
            ),
            (
                [u64::MAX; 8],
                [u64::MAX; 2],
                [0u64; 8],
                "All ones chaining/tweak, all zeros block",
            ),
        ];

        for (chaining_value, tweak, block, description) in test_cases {
            let expected = reference::ubi_block(chaining_value, tweak, block);

            let circuit = CircuitBuilder::new();
            let chaining_wires = std::array::from_fn(|_| circuit.add_witness());
            let tweak_wires = std::array::from_fn(|_| circuit.add_witness());
            let block_wires = std::array::from_fn(|_| circuit.add_witness());

            let g_out = ubi_block(&circuit, chaining_wires, tweak_wires, block_wires);

            for i in 0..8 {
                let expected_wire = circuit.add_constant(Word(expected[i]));
                circuit.assert_eq(format!("{}[{}]", description, i), g_out[i], expected_wire);
            }

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in chaining_value.iter().enumerate() {
                witness[chaining_wires[i]] = Word(val);
            }
            for (i, &val) in tweak.iter().enumerate() {
                witness[tweak_wires[i]] = Word(val);
            }
            for (i, &val) in block.iter().enumerate() {
                witness[block_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();
            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    // Tests from threefish_4rounds_with_injection.rs
    #[test]
    fn test_threefish_4rounds_with_injection_correctness() {
        let test_cases = [
            // (group_idx, input_state, k, t, description)
            (
                0,
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                [
                    0x0F0E0D0C0B0A0908u64,
                    0x0706050403020100u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                    0x7777777777777777u64,
                ],
                [
                    0xAAAAAAAAAAAAAAAAu64,
                    0xBBBBBBBBBBBBBBBBu64,
                    0x1111111111111111u64,
                ],
                "Group 0, sequential pattern",
            ),
            (
                5,
                [
                    0xDEADBEEFCAFEBABEu64,
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                ],
                [0x1111111111111111u64; 9],
                [
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x1111111111111111u64,
                ],
                "Group 5, mixed pattern",
            ),
            (
                17,
                [0u64; 8],
                [u64::MAX; 9],
                [u64::MAX; 3],
                "Group 17, all zeros and all ones",
            ),
        ];

        for (group_idx, v_in, k, t, description) in test_cases {
            let expected = reference::threefish_4rounds_with_injection(v_in, k, t, group_idx);

            let circuit = CircuitBuilder::new();
            let v_in_wires = std::array::from_fn(|_| circuit.add_witness());
            let k_wires = std::array::from_fn(|_| circuit.add_witness());
            let t_wires = std::array::from_fn(|_| circuit.add_witness());

            let comp = Threefish4RoundsWithInjection::new(
                &circuit, v_in_wires, k_wires, t_wires, group_idx,
            );

            for i in 0..8 {
                let expected_wire = circuit.add_constant(Word(expected[i]));
                circuit.assert_eq(
                    format!("{}[{}]", description, i),
                    comp.v_out[i],
                    expected_wire,
                );
            }

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in v_in.iter().enumerate() {
                witness[v_in_wires[i]] = Word(val);
            }
            for (i, &val) in k.iter().enumerate() {
                witness[k_wires[i]] = Word(val);
            }
            for (i, &val) in t.iter().enumerate() {
                witness[t_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();
            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }

    // Tests from threefish512_block.rs
    #[test]
    fn test_threefish512_block_correctness() {
        // Test various key/tweak/block combinations
        let test_cases = [
            // (key, tweak, block, description)
            (
                [
                    0x0123456789ABCDEFu64,
                    0xFEDCBA9876543210u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                [0xAAAAAAAAAAAAAAAAu64, 0xBBBBBBBBBBBBBBBBu64],
                [
                    0x0F0E0D0C0B0A0908u64,
                    0x0706050403020100u64,
                    0x1111111111111111u64,
                    0x2222222222222222u64,
                    0x3333333333333333u64,
                    0x4444444444444444u64,
                    0x5555555555555555u64,
                    0x6666666666666666u64,
                ],
                "Basic pattern",
            ),
            (
                [0u64; 8],
                [0u64; 2],
                [u64::MAX; 8],
                "All zeros key/tweak, all ones block",
            ),
            (
                [u64::MAX; 8],
                [u64::MAX; 2],
                [0u64; 8],
                "All ones key/tweak, all zeros block",
            ),
        ];

        for (key, tweak, block, description) in test_cases {
            let expected = reference::threefish512_block(key, tweak, block);

            let circuit = CircuitBuilder::new();
            let key_wires = std::array::from_fn(|_| circuit.add_witness());
            let tweak_wires = std::array::from_fn(|_| circuit.add_witness());
            let block_wires = std::array::from_fn(|_| circuit.add_witness());

            let comp = Threefish512Block::new(&circuit, key_wires, tweak_wires, block_wires);

            for i in 0..8 {
                let expected_wire = circuit.add_constant(Word(expected[i]));
                circuit.assert_eq(
                    format!("{}[{}]", description, i),
                    comp.v_out[i],
                    expected_wire,
                );
            }

            let built_circuit = circuit.build();
            assert!(
                built_circuit.n_gates() > 0,
                "Circuit should have gates for {}",
                description
            );

            let mut witness = built_circuit.new_witness_filler();

            for (i, &val) in key.iter().enumerate() {
                witness[key_wires[i]] = Word(val);
            }
            for (i, &val) in tweak.iter().enumerate() {
                witness[tweak_wires[i]] = Word(val);
            }
            for (i, &val) in block.iter().enumerate() {
                witness[block_wires[i]] = Word(val);
            }

            built_circuit.populate_wire_witness(&mut witness).unwrap();

            let cs = built_circuit.constraint_system();
            verify_constraints(cs, &witness.into_value_vec())
                .unwrap_or_else(|_| panic!("Constraints verification failed for {}", description));
        }
    }
}
