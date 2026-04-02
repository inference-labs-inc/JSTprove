// Copyright 2025 Irreducible Inc.

//! Various utilities for circuit building.

use std::iter;

use binius_core::{Word, consts::WORD_SIZE_BITS};

use crate::compiler::{CircuitBuilder, Wire, circuit::WitnessFiller};

/// Populate the given wires from bytes using little-endian packed 64-bit words.
///
/// If `bytes` is not a multiple of 8, the last word is zero-padded.
///
/// If there are more wires than needed to hold all bytes, the remaining wires
/// are filled with `Word::ZERO`.
///
/// # Panics
/// * If bytes.len() exceeds wires.len() * 8
pub fn pack_bytes_into_wires_le(w: &mut WitnessFiller, wires: &[Wire], bytes: &[u8]) {
    let max_value_size = wires.len() * 8;
    assert!(
        bytes.len() <= max_value_size,
        "bytes length {} exceeds maximum {}",
        bytes.len(),
        max_value_size
    );

    // Pack bytes into words
    for (&wire, chunk) in iter::zip(wires, bytes.chunks(8)) {
        let mut chunk_arr = [0u8; 8];
        chunk_arr[..chunk.len()].copy_from_slice(chunk);
        w[wire] = Word(u64::from_le_bytes(chunk_arr));
    }

    // Zero out remaining words
    for &wire in &wires[bytes.len().div_ceil(8)..] {
        w[wire] = Word::ZERO;
    }
}

/// Pack bytes into constant wires using little-endian packed 64-bit words.
///
/// Creates and returns a vector of constant wires containing the packed byte data.
/// If `bytes` is not a multiple of 8, the last word is zero-padded.
///
/// # Example
/// ```ignore
/// let bytes = b"Hello";
/// let wires = pack_bytes_into_const_wires_le(&builder, bytes);
/// // wires.len() == 1 (5 bytes packed into 1 word, zero-padded)
/// ```
pub fn pack_bytes_into_const_wires_le(b: &CircuitBuilder, bytes: &[u8]) -> Vec<Wire> {
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut chunk_arr = [0u8; 8];
            chunk_arr[..chunk.len()].copy_from_slice(chunk);
            let word = Word(u64::from_le_bytes(chunk_arr));
            b.add_constant(word)
        })
        .collect()
}

/// Returns a BigUint from u64 limbs with little-endian ordering
pub fn num_biguint_from_u64_limbs<I>(limbs: I) -> num_bigint::BigUint
where
    I: IntoIterator,
    I::Item: std::borrow::Borrow<u64>,
    I::IntoIter: ExactSizeIterator,
{
    use std::borrow::Borrow;

    use num_bigint::BigUint;

    let iter = limbs.into_iter();
    // Each u64 becomes two u32s (low word first for little-endian)
    let mut digits = Vec::with_capacity(iter.len() * 2);
    for item in iter {
        let double_digit = *item.borrow();
        // push:
        // - low 32 bits
        // - high 32 bits
        digits.push(double_digit as u32);
        digits.push((double_digit >> 32) as u32);
    }
    BigUint::new(digits)
}

/// Check that all boolean wires in an iterable are simultaneously true.
pub fn all_true(b: &CircuitBuilder, booleans: impl IntoIterator<Item = Wire>) -> Wire {
    booleans
        .into_iter()
        .fold(b.add_constant(Word::ALL_ONE), |lhs, rhs| b.band(lhs, rhs))
}

/// Convert MSB-bool into an all-1/all-0 mask.
pub fn bool_to_mask(b: &CircuitBuilder, boolean: Wire) -> Wire {
    b.sar(boolean, (WORD_SIZE_BITS - 1) as u32)
}

/// Swap the byte order of the word.
///
/// Breaks the word down to bytes and reassembles in reversed order.
pub fn byteswap(b: &CircuitBuilder, word: Wire) -> Wire {
    let bytes = (0..8).map(|j| {
        let byte = b.extract_byte(word, j as u32);
        b.shl(byte, (56 - 8 * j) as u32)
    });
    bytes
        .reduce(|lhs, rhs| b.bxor(lhs, rhs))
        .expect("WORD_SIZE_BITS > 0")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to test pack_bytes_into_const_wires_le with expected word values
    fn test_pack_bytes(bytes: &[u8], expected_words: &[u64]) {
        let b = CircuitBuilder::new();
        let wires = pack_bytes_into_const_wires_le(&b, bytes);

        assert_eq!(wires.len(), expected_words.len());

        let circuit = b.build();
        let mut filler = circuit.new_witness_filler();
        circuit.populate_wire_witness(&mut filler).unwrap();

        for (wire, &expected) in wires.iter().zip(expected_words) {
            assert_eq!(filler[*wire], Word(expected));
        }
    }

    #[test]
    fn test_pack_bytes_into_const_wires_le_aligned() {
        // Test with exactly 8 bytes (1 word)
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        test_pack_bytes(&bytes, &[0x0807060504030201]);
    }

    #[test]
    fn test_pack_bytes_into_const_wires_le_partial() {
        // Test with 5 bytes (partial word)
        // "Hello" in little-endian with zero padding
        // H=0x48, e=0x65, l=0x6c, l=0x6c, o=0x6f
        test_pack_bytes(b"Hello", &[0x6f6c6c6548]);
    }

    #[test]
    fn test_pack_bytes_into_const_wires_le_multiple_words() {
        // Test with 16 bytes (2 words)
        let bytes: Vec<u8> = (0..16).collect();
        test_pack_bytes(&bytes, &[0x0706050403020100, 0x0f0e0d0c0b0a0908]);
    }

    #[test]
    fn test_pack_bytes_into_const_wires_le_empty() {
        // Test with empty bytes
        test_pack_bytes(&[], &[]);
    }

    #[test]
    fn test_pack_bytes_into_const_wires_le_unaligned_multi() {
        // Test with 11 bytes (1 full word + partial second word, zero-padded)
        let bytes: Vec<u8> = (0..11).collect();
        test_pack_bytes(&bytes, &[0x0706050403020100, 0x0a0908]);
    }
}
