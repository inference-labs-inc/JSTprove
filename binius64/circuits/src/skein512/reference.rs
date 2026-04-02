// Copyright 2025 Irreducible Inc.
/// This module contains a reference Skein-512 implementation that we use to
/// cross-check the different circuit builder implementations against.
/// It has been tested against known good vectors and should match the
/// reference output exactly.
/// It is also optimized for working with 64-bit words, so that circuit builders
/// can match exactly the word-level operations performed by the reference implementation.
use crate::skein512::constants::{C240, R512, TWEAK_TYPE_CFG, TWEAK_TYPE_MSG, TWEAK_TYPE_OUT};

#[inline(always)]
pub fn mix(a: u64, b: u64, r: u32) -> (u64, u64) {
    // Mix (Threefish): a = a + b; b = ROTL(b, r) ^ a
    let a2 = a.wrapping_add(b);
    let b2 = b.rotate_left(r) ^ a2;
    (a2, b2)
}

/// The 512-bit u64 permutation applied between rounds.
#[inline(always)]
pub fn permute_512(x: [u64; 8]) -> [u64; 8] {
    [x[2], x[1], x[4], x[7], x[6], x[5], x[0], x[3]]
}

#[inline(always)]
pub fn threefish_subkey(s: usize, k: [u64; 9], t: [u64; 3]) -> [u64; 8] {
    // Per Threefish: every 4 rounds we add a subkey.
    // v[i] += k[(s + i) mod 9], i=0..7
    // v[5] += t[s mod 3]
    // v[6] += t[(s + 1) mod 3]
    // v[7] += s
    let mut sk = [0u64; 8];
    for i in 0..8 {
        sk[i] = k[(s + i) % 9];
    }
    sk[5] = sk[5].wrapping_add(t[s % 3]);
    sk[6] = sk[6].wrapping_add(t[(s + 1) % 3]);
    sk[7] = sk[7].wrapping_add(s as u64);
    sk
}

#[inline(always)]
pub fn threefish_round(mut v: [u64; 8], round_idx: usize) -> [u64; 8] {
    // Pairing for 512 is (0,1), (2,3), (4,5), (6,7) with round-dependent rotations.
    let r = R512[round_idx % 8];

    // 4 independent MIXes (subcircuits can be parallelized)
    let (v0, v1) = mix(v[0], v[1], r[0]);
    let (v2, v3) = mix(v[2], v[3], r[1]);
    let (v4, v5) = mix(v[4], v[5], r[2]);
    let (v6, v7) = mix(v[6], v[7], r[3]);

    v = [v0, v1, v2, v3, v4, v5, v6, v7];

    // Permutation subcircuit (single shuffle on 8 lanes)
    permute_512(v)
}

#[inline(always)]
pub fn threefish_4rounds_with_injection(
    mut v: [u64; 8],
    k: [u64; 9],
    t: [u64; 3],
    group_idx: usize, // 0..=18 for 72 rounds
) -> [u64; 8] {
    // Inject subkey before the 4 rounds
    let sub = threefish_subkey(group_idx, k, t);
    for i in 0..8 {
        v[i] = v[i].wrapping_add(sub[i]);
    }
    // Do 4 rounds
    let base = group_idx * 4;
    v = threefish_round(v, base);
    v = threefish_round(v, base + 1);
    v = threefish_round(v, base + 2);
    v = threefish_round(v, base + 3);
    v
}

pub fn threefish512_block(key: [u64; 8], tweak: [u64; 2], block: [u64; 8]) -> [u64; 8] {
    // Expand key and tweak
    let mut k = [0u64; 9];
    let mut sum = 0u64;
    for i in 0..8 {
        k[i] = key[i];
        sum ^= key[i];
    }
    k[8] = C240 ^ sum;

    let t0 = tweak[0];
    let t1 = tweak[1];
    let t2 = t0 ^ t1;
    let t = [t0, t1, t2];

    // Initial state is the plaintext/message block
    let mut v = block;

    // 72 rounds = 18 groups of 4 rounds, with subkey injections
    for g in 0..18 {
        v = threefish_4rounds_with_injection(v, k, t, g);
    }

    // Final subkey injection (18th injection after the 72 rounds)
    let sub = threefish_subkey(18, k, t);
    for i in 0..8 {
        v[i] = v[i].wrapping_add(sub[i]);
    }

    v
}

// Tweak bit layout per Skein specification Table 5:
// - position: bits 0..95 = number of bytes processed so far (including this block)
// - tree level: 112..118 = 0 (non-tree)
// - bitpad: 119 = 0 (we only handle full bytes here)
// - type: 120..125 = typ
// - first: 126
// - final: 127
#[inline(always)]
pub fn tweak(
    type_code: u64,
    pos_bytes_0: u64,
    pos_bytes_1: u64,
    is_first: bool,
    is_final: bool,
) -> [u64; 2] {
    let pos_bytes_1 = pos_bytes_1 & ((1u64 << 32) - 1); // Ensure low 32 bits only

    let t_0 = pos_bytes_0;
    let mut t_1 = (type_code << 56) | pos_bytes_1;

    if is_first {
        t_1 |= 1u64 << 62;
    }
    if is_final {
        t_1 |= 1u64 << 63;
    }

    [t_0, t_1]
}

#[inline(always)]
pub fn ubi_block(chaining_value: [u64; 8], tweak: [u64; 2], block: [u64; 8]) -> [u64; 8] {
    // UBI compression:
    // G' = Threefish(K = CV, T = tweak, M = block) XOR M
    // Then CV_next = G'
    let out = threefish512_block(chaining_value, tweak, block);

    let mut g = [0u64; 8];
    for i in 0..8 {
        g[i] = out[i] ^ block[i];
    }
    g
}

#[inline(always)]
fn load_u64_le(x: &[u8]) -> u64 {
    u64::from_le_bytes(x.try_into().expect("slice with incorrect length"))
}

#[inline(always)]
fn store_u64_le(x: u64, out: &mut [u8]) {
    out.copy_from_slice(&x.to_le_bytes());
}

pub fn bytes_to_u64s_512(block: &[u8]) -> [u64; 8] {
    let mut w = [0u64; 8];
    for i in 0..8 {
        w[i] = load_u64_le(&block[8 * i..8 * (i + 1)]);
    }
    w
}

fn u64s_to_bytes_512(u64s: &[u64; 8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for i in 0..8 {
        store_u64_le(u64s[i], &mut out[8 * i..8 * (i + 1)]);
    }
    out
}

pub struct Skein512 {
    h: [u64; 8],           // current chaining value
    buf: [u8; 64],         // message buffer for streaming
    buf_len: usize,        // current buffer length
    processed: u128,       // how many message bytes have been processed so far
    msg_first_block: bool, // whether the next message block is the first in this UBI invocation
}

impl Skein512 {
    pub fn new() -> Self {
        // Build 32-byte configuration string C (Table 7 from Skein v1.3):
        // offset 0:4  = ASCII "SHA3" (0x53, 0x48, 0x41, 0x33)
        // offset 4:6  = version 1 (u16 LE)
        // offset 6:8  = reserved 0
        // offset 8:16 = output length in bits (u64 LE) => output_bits
        // offset 16   = Yl (0), 17: Yf (0), 18: Ym (0)
        // offset 19..31 = reserved 0
        let mut cfg = [0u8; 64];
        cfg[0..4].copy_from_slice(&0x3341_4853u32.to_le_bytes()); // "SHA3"
        cfg[4..6].copy_from_slice(&1u16.to_le_bytes());
        cfg[6..8].copy_from_slice(&0u16.to_le_bytes());
        cfg[8..16].copy_from_slice(&512u64.to_le_bytes());
        cfg[16] = 0; // Yl
        cfg[17] = 0; // Yf
        cfg[18] = 0; // Ym
        // rest already zero (bytes 19..31 and 32..63)

        let cfg_u64s = bytes_to_u64s_512(&cfg);
        let t_cfg = tweak(
            TWEAK_TYPE_CFG, // type code
            32,
            0,    // position = config length in bytes (32 bytes)
            true, // FIRST
            true, // FINAL (single config block)
        );
        // Initial CV is zero in Skein for config UBI
        let cv0 = [0u64; 8];
        let h0 = ubi_block(cv0, t_cfg, cfg_u64s);

        Self {
            h: h0,
            buf: [0u8; 64],
            buf_len: 0,
            processed: 0,
            msg_first_block: true,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        // Stream full 64-byte blocks immediately. The final (possibly partial) block is
        // handled in finalize().
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];

            if self.buf_len == 64 {
                // process this full block (not final)
                let block: [u8; 64] = self.buf;
                self.process_msg_block(&block, false, None);
                self.buf_len = 0;
            }
        }

        // process all full 64-byte chunks directly from input
        while data.len() >= 64 {
            let block: [u8; 64] = data[..64].try_into().unwrap();
            self.process_msg_block(&block, false, None);
            data = &data[64..];
        }

        // buffer the remainder
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn process_msg_block(
        &mut self,
        block: &[u8; 64],
        is_final: bool,
        pos_end_override: Option<u128>,
    ) {
        // Compute tweak for this block under Tmsg. Position is bytes processed so far INCLUDING
        // this block.
        let is_first = self.msg_first_block;
        let pos_end = pos_end_override.unwrap_or(self.processed + 64u128);
        let t = tweak(
            TWEAK_TYPE_MSG, // type code
            pos_end as u64,
            (pos_end >> 64) as u64,
            is_first,
            is_final,
        );

        // Process block and feedforward
        let u64s = bytes_to_u64s_512(block);
        self.h = ubi_block(self.h, t, u64s);

        self.processed = pos_end;
        self.msg_first_block = false;
    }

    pub fn finalize(mut self) -> [u8; 64] {
        // Handle the message UBI final block
        if self.processed == 0 && self.buf_len == 0 {
            // Zero-length message: per spec, process one all-zero block with pos = 0,
            // first=1, final=1, type=Tmsg.
            let zero_block = [0u8; 64];
            self.process_msg_block(&zero_block, true, Some(0));
        } else {
            // Final block = current buffer (padded with zeros)
            let mut block = [0u8; 64];
            block[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
            // Position field for the final block is total message length in bytes (without zero
            // padding)
            let pos_end = self.processed + (self.buf_len as u128);
            self.process_msg_block(&block, true, Some(pos_end));
        }

        // Output transform: UBI(G, ToBytes(counter,8), Tout) and take first 64 bytes.
        // For 512-bit output we only need counter=0 block.
        let mut out_block = [0u8; 64];
        // Counter=0 in first 8 bytes; rest zero.
        out_block[0..8].copy_from_slice(&0u64.to_le_bytes());
        let out_u64s = bytes_to_u64s_512(&out_block);
        let t_out = tweak(
            TWEAK_TYPE_OUT, // type code
            8,
            0,    /* position = 8 bytes counter */
            true, // FIRST
            true, // FINAL
        );
        let final_u64s = ubi_block(self.h, t_out, out_u64s);
        u64s_to_bytes_512(&final_u64s)
    }
}

/// Convenient one-shot function
pub fn skein512(input: &[u8]) -> [u8; 64] {
    let mut h = Skein512::new();
    h.update(input);
    h.finalize()
}

#[test]
fn smoke_build() {
    let _ = Skein512::new();
}

fn test_hash(data: &str, expected_hex: &str) {
    let digest = skein512(data.as_bytes());
    let digest_hex = digest
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();
    assert_eq!(digest_hex, expected_hex);
}

#[test]
fn hash_predefined() {
    for (data, expected_hex) in [
        (
            "",
            "bc5b4c50925519c290cc634277ae3d6257212395cba733bbad37a4af0fa06af41fca7903d06564fea7a2d3730dbdb80c1f85562dfcc070334ea4d1d9e72cba7a",
        ),
        (
            "The quick brown fox jumps over the lazy dog",
            "94c2ae036dba8783d0b3f7d6cc111ff810702f5c77707999be7e1c9486ff238a7044de734293147359b4ac7e1d09cd247c351d69826b78dcddd951f0ef912713",
        ),
        (
            "The quick brown fox jumps over the lazy cog",
            "7f81113575e4b4d3441940e87aca331e6d63d103fe5107f29cd877af0d0f5e0ea34164258c60da5190189d0872e63a96596d2ef25e709099842da71d64111e0f",
        ),
    ] {
        test_hash(data, expected_hex);
    }
}

#[test]
fn test_streaming_vs_oneshot() {
    // Test that streaming API produces same results as one-shot
    let test_message = b"The quick brown fox jumps over the lazy dog";

    // One-shot
    let digest_oneshot = skein512(test_message);

    // Streaming - process in chunks
    let mut hasher = Skein512::new();
    hasher.update(&test_message[..10]); // "The quick "
    hasher.update(&test_message[10..26]); // "brown fox jumps "
    hasher.update(&test_message[26..]); // "over the lazy dog"
    let digest_streaming = hasher.finalize();

    assert_eq!(digest_oneshot, digest_streaming);
}
