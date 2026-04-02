// Copyright 2025 Irreducible Inc.
//! BLAKE2b reference implementation
//!
//! This module provides a pure Rust implementation of BLAKE2b following RFC 7693.
//! It serves as a reference for the circuit implementation and testing.

use super::constants::{BLOCK_BYTES, IV, MAX_OUTPUT_BYTES, R1, R2, R3, R4, ROUNDS, SIGMA};

/// Rotate right for 64-bit words
#[inline(always)]
fn rotr64(x: u64, n: u32) -> u64 {
    (x >> n) | (x << (64 - n))
}

/// G mixing function - the core primitive of BLAKE2b
///
/// Performs 8 operations mixing two input words with the state
#[inline(always)]
pub fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = rotr64(v[d] ^ v[a], R1);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr64(v[b] ^ v[c], R2);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = rotr64(v[d] ^ v[a], R3);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr64(v[b] ^ v[c], R4);
}

/// Convert bytes to 64-bit words (little-endian)
pub fn bytes_to_words(bytes: &[u8]) -> [u64; 16] {
    let mut words = [0u64; 16];
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    words
}

/// BLAKE2b compression function F
///
/// Compresses a 128-byte block into the state using 12 rounds of mixing
pub fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    // Initialize working vector
    let mut v = [0u64; 16];

    // First half from state
    v[0..8].copy_from_slice(h);

    // Second half from IV
    v[8..16].copy_from_slice(&IV);

    // Mix in counter (128-bit counter split into two 64-bit words)
    v[12] ^= t as u64; // Low word
    v[13] ^= (t >> 64) as u64; // High word

    // Invert v[14] for last block flag
    if last {
        v[14] = !v[14];
    }

    // Convert block to 16 words
    let m = bytes_to_words(block);

    // 12 rounds of mixing
    for round in 0..ROUNDS {
        let s = &SIGMA[round];

        // Column step (mix columns of the 4x4 matrix)
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);

        // Diagonal step (mix diagonals of the 4x4 matrix)
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

    // Finalization: XOR the two halves back into state
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// BLAKE2b hash function with variable output length
///
/// Computes BLAKE2b hash of input data with specified output length (1-64 bytes)
pub fn blake2b(data: &[u8], outlen: usize) -> Vec<u8> {
    assert!(
        outlen > 0 && outlen <= MAX_OUTPUT_BYTES,
        "Output length must be 1-64 bytes"
    );

    // Initialize state with IV XORed with parameter block
    let mut h = IV;

    // Parameter block: Set output length in first byte, rest are zeros for basic version
    // Format: 0x0101kknn where nn=outlen, kk=keylen (0 for us), fanout=depth=1
    h[0] ^= 0x01010000 | (outlen as u64);

    // Process message blocks
    let mut t = 0u128; // Total bytes counter
    let mut offset = 0;

    // Process all complete blocks except the last one
    while offset + BLOCK_BYTES < data.len() {
        let mut block = [0u8; BLOCK_BYTES];
        block.copy_from_slice(&data[offset..offset + BLOCK_BYTES]);

        t += BLOCK_BYTES as u128;
        compress(&mut h, &block, t, false);

        offset += BLOCK_BYTES;
    }

    // Process final block (always exists, may be partial or full)
    let mut final_block = [0u8; BLOCK_BYTES];
    let remaining = data.len() - offset;
    if remaining > 0 {
        final_block[..remaining].copy_from_slice(&data[offset..]);
    }

    t += remaining as u128;
    compress(&mut h, &final_block, t, true); // Set last block flag

    // Convert state to bytes and return requested length
    let mut output = Vec::with_capacity(outlen);
    for word in h.iter() {
        let bytes = word.to_le_bytes();
        for byte in bytes {
            if output.len() < outlen {
                output.push(byte);
            }
        }
    }
    output.truncate(outlen);
    output
}

/// BLAKE2b-256: Fixed 256-bit (32-byte) output variant
///
/// This is a convenience function for the common 256-bit output case
pub fn blake2b_256(data: &[u8]) -> [u8; 32] {
    let hash = blake2b(data, 32);
    let mut result = [0u8; 32];
    result.copy_from_slice(&hash);
    result
}

/// BLAKE2b-512: Fixed 512-bit (64-byte) output variant
///
/// This is a convenience function for the maximum 512-bit output case
pub fn blake2b_512(data: &[u8]) -> [u8; 64] {
    let hash = blake2b(data, 64);
    let mut result = [0u8; 64];
    result.copy_from_slice(&hash);
    result
}

#[cfg(test)]
mod tests {
    use blake2::digest::{Update, VariableOutput};

    use super::*;

    /// Test variable output lengths
    #[test]
    fn test_variable_output_lengths() {
        let msg = b"test message for variable output lengths";

        // Test various output lengths from 1 to 64 bytes
        for outlen in 1..=64 {
            let mut hasher = blake2::Blake2bVar::new(outlen).unwrap();
            Update::update(&mut hasher, msg);
            let expected = hasher.finalize_boxed();

            let result = blake2b(msg, outlen);
            assert_eq!(
                result,
                expected.as_ref(),
                "Variable output test failed for length {}",
                outlen
            );
        }
    }

    /// Test incremental hashing verification
    #[test]
    fn test_incremental_hashing() {
        let data = vec![0x55u8; 300];

        // Hash all at once with our reference
        let expected = blake2b_256(&data);

        // Incremental hashing using standard crate
        let mut hasher = blake2::Blake2bVar::new(32).unwrap();
        Update::update(&mut hasher, &data[0..100]);
        Update::update(&mut hasher, &data[100..200]);
        Update::update(&mut hasher, &data[200..300]);
        let incremental = hasher.finalize_boxed();

        // Our implementation should match incrementally computed hash
        assert_eq!(
            &expected[..],
            &incremental[..],
            "Incremental hashing verification failed"
        );
    }
}
