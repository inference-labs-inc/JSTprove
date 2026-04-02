// Copyright 2025 Irreducible Inc.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::{
    bignum::BigUint,
    bytes::swap_bytes_32,
    ecdsa::scalar_mul::scalar_mul_naive,
    ripemd::ripemd160_fixed,
    secp256k1::{Secp256k1, Secp256k1Affine},
    sha256::sha256_fixed,
};

/// Convert 20-byte address payload into five little-endian u32 words (circuit witness layout).
pub fn addr_bytes_to_le_words(addr: &[u8; 20]) -> [u32; 5] {
    [
        u32::from_le_bytes([addr[0], addr[1], addr[2], addr[3]]),
        u32::from_le_bytes([addr[4], addr[5], addr[6], addr[7]]),
        u32::from_le_bytes([addr[8], addr[9], addr[10], addr[11]]),
        u32::from_le_bytes([addr[12], addr[13], addr[14], addr[15]]),
        u32::from_le_bytes([addr[16], addr[17], addr[18], addr[19]]),
    ]
}

/// Builds a circuit that proves knowledge of a Bitcoin private key corresponding to a P2PKH
/// address.
///
/// This circuit implements the complete Bitcoin P2PKH address derivation:
/// Private Key → \[scalar_mul\] → Curve Point → \[compress\] → Compressed PubKey
/// → \[sha256\] → SHA256 Digest → \[swap_bytes_32\] → LE Format → \[ripemd160\] → Address
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `private_key` - Private key as BigUint (4 limbs, 256 bits)
/// * `expected_address` - Expected Bitcoin address as RIPEMD160 output (5 × 32-bit words)
///
/// # Circuit Flow
/// 1. Multiply private key by secp256k1 generator point
/// 2. Compress the resulting public key to 33-byte format
/// 3. Compute SHA256 hash of compressed public key
/// 4. Convert SHA256 output from big-endian to little-endian format
/// 5. Compute RIPEMD160 hash of the SHA256 digest
/// 6. Assert computed address equals expected address
///
/// # Panics
/// * If private_key doesn't have exactly 4 limbs (256 bits)
pub fn build_p2pkh_circuit(
    builder: &CircuitBuilder,
    private_key: &BigUint,
    expected_address: [Wire; 5],
) {
    assert_eq!(
        private_key.limbs.len(),
        4,
        "private_key must be exactly 4 limbs (256 bits)"
    );

    // Step 1: Scalar multiplication - private_key × generator → public key point
    let curve = Secp256k1::new(builder);
    let generator = Secp256k1Affine::generator(builder);

    let public_key_point = scalar_mul_naive(builder, &curve, 256, private_key, generator);

    // Step 2: Compress public key - (x, y) → 33-byte compressed format
    let compressed_pubkey = compress_pubkey(builder, &public_key_point.x, &public_key_point.y);

    // Step 3: SHA256 hash - 33 bytes → 32-byte digest
    let sha256_digest = sha256_fixed(builder, &compressed_pubkey, 33);

    // Step 4: Convert SHA256 output from BE to LE format for RIPEMD160
    // sha256_fixed returns [Wire; 8] (8 × 32-bit words in big-endian)
    // We need to swap bytes in each word to get little-endian format
    let mut swapped_digest = Vec::with_capacity(8);
    for &word in &sha256_digest {
        swapped_digest.push(swap_bytes_32(builder, word));
    }

    // Step 5: RIPEMD160 hash - 32 bytes → 20-byte address
    let computed_address = ripemd160_fixed(builder, &swapped_digest, 32);

    // Step 6: Verify computed address matches expected address
    for i in 0..5 {
        builder.assert_eq(
            format!("p2pkh_address[{}]", i),
            computed_address[i],
            expected_address[i],
        );
    }
}

/// Compresses a secp256k1 public key from uncompressed (x, y) format to compressed format.
///
/// Bitcoin uses compressed public keys which are 33 bytes instead of 65 bytes:
/// - Uncompressed: \[0x04\] || x (32 bytes) || y (32 bytes) = 65 bytes
/// - Compressed: \[0x02 or 0x03\] || x (32 bytes) = 33 bytes
///
/// The prefix byte indicates the parity of the y-coordinate:
/// - 0x02 if y is even (LSB = 0)
/// - 0x03 if y is odd (LSB = 1)
///
/// # Arguments
/// * `builder` - Circuit builder for constructing constraints
/// * `x` - x-coordinate of the public key (32 bytes, 4 limbs)
/// * `y` - y-coordinate of the public key (32 bytes, 4 limbs)
///
/// # Returns
/// * `Vec<Wire>` - Compressed public key as 33 bytes suitable for sha256_fixed input. Each wire
///   contains 4 bytes (32-bit word) with high 32 bits zeroed.
///
/// # Panics
/// * If x or y don't have exactly 4 limbs (256 bits)
pub fn compress_pubkey(builder: &CircuitBuilder, x: &BigUint, y: &BigUint) -> Vec<Wire> {
    assert_eq!(
        x.limbs.len(),
        4,
        "x-coordinate must be exactly 4 limbs (256 bits)"
    );
    assert_eq!(
        y.limbs.len(),
        4,
        "y-coordinate must be exactly 4 limbs (256 bits)"
    );

    let y_is_odd = builder.shl(y.limbs[0], 63);

    // Create prefix: 0x02 if y is even, 0x03 if y is odd
    let prefix_even = builder.add_constant(Word::from_u64(0x02 << 24));
    let prefix_odd = builder.add_constant(Word::from_u64(0x03 << 24));
    let prefix_byte = builder.select(y_is_odd, prefix_odd, prefix_even);

    // We need to produce 9 words (33 bytes) for sha256_fixed
    // Each word represents 4 bytes packed in big-endian format
    let lower_32_bits = |limb| builder.band(limb, builder.add_constant_64(0xFFFFFFFF));
    vec![
        builder.bor(builder.shr(x.limbs[3], 40), prefix_byte),
        lower_32_bits(builder.shr(x.limbs[3], 8)),
        lower_32_bits(builder.bor(builder.shl(x.limbs[3], 24), builder.shr(x.limbs[2], 40))),
        lower_32_bits(builder.shr(x.limbs[2], 8)),
        lower_32_bits(builder.bor(builder.shl(x.limbs[2], 24), builder.shr(x.limbs[1], 40))),
        lower_32_bits(builder.shr(x.limbs[1], 8)),
        lower_32_bits(builder.bor(builder.shl(x.limbs[1], 24), builder.shr(x.limbs[0], 40))),
        lower_32_bits(builder.shr(x.limbs[0], 8)),
        lower_32_bits(builder.shl(x.limbs[0], 24)),
    ]
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use bitcoin::PublicKey;
    use bitcoin_hashes::Hash;
    use proptest::prelude::RngCore;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    /// This is a reference function reused by tests and examples
    /// Compute compressed SEC1 public key from a 32-byte big-endian private key.
    /// Panics if the scalar is zero.
    pub fn compressed_pubkey_from_be_sk(sk_be: [u8; 32]) -> [u8; 33] {
        use k256::{
            ProjectivePoint, Scalar,
            elliptic_curve::{ops::MulByGenerator, sec1::ToEncodedPoint},
        };

        // Interpret big-endian bytes as scalar mod curve order via Horner's method
        let mut s = Scalar::from(0u64);
        let base = Scalar::from(256u64);
        for &b in &sk_be {
            s = s * base + Scalar::from(b as u64);
        }
        assert!(!bool::from(s.is_zero()), "private key must be non-zero");

        let affine = ProjectivePoint::mul_by_generator(&s).to_affine();
        let ep = affine.to_encoded_point(true);
        let mut out = [0u8; 33];
        out.copy_from_slice(ep.as_bytes());
        out
    }

    /// Convenience to produce a random pair (privkey_be, compressed_pubkey)
    pub fn gen_priv_pub_pair_from_rng(mut rng: impl RngCore) -> ([u8; 32], [u8; 33]) {
        let mut sk_be = [0u8; 32];
        rng.fill_bytes(&mut sk_be);
        let pk_comp = compressed_pubkey_from_be_sk(sk_be);
        (sk_be, pk_comp)
    }

    // Utility: validate both compressed pubkey and full P2PKH circuit against a priv/pub pair
    fn validate_priv_pub_pair(private_key_be: [u8; 32], compressed_pubkey: [u8; 33]) {
        // 1) Check scalar_mul + compression matches expected compressed pubkey
        assert_circuit_matches_priv_pub_pair(private_key_be, compressed_pubkey);

        // 2) Check the full P2PKH circuit (SHA256 then RIPEMD160 on compressed pubkey)
        let builder = CircuitBuilder::new();
        let private_key = BigUint::new_witness(&builder, 4);
        let expected_address: [Wire; 5] = std::array::from_fn(|_| builder.add_witness());

        build_p2pkh_circuit(&builder, &private_key, expected_address);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        // Populate private key limbs (LE u64s)
        let mut sk_le = [0u8; 32];
        for i in 0..32 {
            sk_le[i] = private_key_be[31 - i];
        }
        let limbs: [u64; 4] = [
            u64::from_le_bytes(sk_le[0..8].try_into().unwrap()),
            u64::from_le_bytes(sk_le[8..16].try_into().unwrap()),
            u64::from_le_bytes(sk_le[16..24].try_into().unwrap()),
            u64::from_le_bytes(sk_le[24..32].try_into().unwrap()),
        ];
        private_key.populate_limbs(&mut w, &limbs);

        // Compute expected P2PKH address using bitcoin crate and populate expected wires
        let public_key =
            PublicKey::from_slice(&compressed_pubkey).expect("Invalid compressed public key");
        let pubkey_hash = public_key.pubkey_hash();
        let addr_bytes: [u8; 20] = pubkey_hash.to_byte_array();
        for i in 0..5 {
            let start = i * 4;
            let v = u32::from_le_bytes([
                addr_bytes[start],
                addr_bytes[start + 1],
                addr_bytes[start + 2],
                addr_bytes[start + 3],
            ]);
            w[expected_address[i]] = Word::from_u64(v as u64);
        }

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
    }

    // Utility: verify the circuit's scalar_mul + compression against a given private/public pair
    // private_key_be: 32-byte big-endian private key
    // expected_compressed_sec1: 33-byte SEC1 compressed public key (prefix + 32-byte x)
    fn assert_circuit_matches_priv_pub_pair(
        private_key_be: [u8; 32],
        expected_compressed_sec1: [u8; 33],
    ) {
        let builder = CircuitBuilder::new();

        // Inputs and expected outputs
        let private_key = BigUint::new_witness(&builder, 4);
        let expected_words: Vec<Wire> = (0..9).map(|_| builder.add_witness()).collect();

        // Compute pubkey in the circuit
        let curve = Secp256k1::new(&builder);
        let generator = Secp256k1Affine::generator(&builder);
        let pub_point = scalar_mul_naive(&builder, &curve, 256, &private_key, generator);
        let compressed = compress_pubkey(&builder, &pub_point.x, &pub_point.y);

        for i in 0..9 {
            builder.assert_eq(format!("compressed[{i}]"), compressed[i], expected_words[i]);
        }

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        // Populate private key limbs (circuit expects 4x u64 little-endian limbs)
        let mut sk_le = [0u8; 32];
        for i in 0..32 {
            sk_le[i] = private_key_be[31 - i];
        }
        let limbs: [u64; 4] = [
            u64::from_le_bytes(sk_le[0..8].try_into().unwrap()),
            u64::from_le_bytes(sk_le[8..16].try_into().unwrap()),
            u64::from_le_bytes(sk_le[16..24].try_into().unwrap()),
            u64::from_le_bytes(sk_le[24..32].try_into().unwrap()),
        ];
        private_key.populate_limbs(&mut w, &limbs);

        // Pack expected compressed bytes into 9 big-endian u32 words (low 32 bits used)
        let mut expected_word_values = [0u32; 9];
        for i in 0..9 {
            let start = i * 4;
            let mut word = 0u32;
            for j in 0..4 {
                if start + j < 33 {
                    word |= (expected_compressed_sec1[start + j] as u32) << (24 - j * 8);
                }
            }
            expected_word_values[i] = word;
        }
        for i in 0..9 {
            w[expected_words[i]] = Word::from_u64(expected_word_values[i] as u64);
        }

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
    }

    fn test_compress_helper(x_bytes: [u8; 32], y_bytes: [u8; 32], prefix: u8) {
        let builder = CircuitBuilder::new();

        // Convert byte arrays to BigUint limbs (little-endian)
        let x = BigUint::new_witness(&builder, 4);
        let y = BigUint::new_witness(&builder, 4);

        // Expected compressed output wires for verification
        let expected_words: Vec<Wire> = (0..9).map(|_| builder.add_witness()).collect();

        // Call compress function
        let compressed = compress_pubkey(&builder, &x, &y);

        // Assert equality with expected result
        for i in 0..9 {
            builder.assert_eq(
                format!("compressed[{}]", i),
                compressed[i],
                expected_words[i],
            );
        }

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        // Populate x and y coordinates
        let x_limbs: [u64; 4] = [
            u64::from_le_bytes([
                x_bytes[0], x_bytes[1], x_bytes[2], x_bytes[3], x_bytes[4], x_bytes[5], x_bytes[6],
                x_bytes[7],
            ]),
            u64::from_le_bytes([
                x_bytes[8],
                x_bytes[9],
                x_bytes[10],
                x_bytes[11],
                x_bytes[12],
                x_bytes[13],
                x_bytes[14],
                x_bytes[15],
            ]),
            u64::from_le_bytes([
                x_bytes[16],
                x_bytes[17],
                x_bytes[18],
                x_bytes[19],
                x_bytes[20],
                x_bytes[21],
                x_bytes[22],
                x_bytes[23],
            ]),
            u64::from_le_bytes([
                x_bytes[24],
                x_bytes[25],
                x_bytes[26],
                x_bytes[27],
                x_bytes[28],
                x_bytes[29],
                x_bytes[30],
                x_bytes[31],
            ]),
        ];

        let y_limbs: [u64; 4] = [
            u64::from_le_bytes([
                y_bytes[0], y_bytes[1], y_bytes[2], y_bytes[3], y_bytes[4], y_bytes[5], y_bytes[6],
                y_bytes[7],
            ]),
            u64::from_le_bytes([
                y_bytes[8],
                y_bytes[9],
                y_bytes[10],
                y_bytes[11],
                y_bytes[12],
                y_bytes[13],
                y_bytes[14],
                y_bytes[15],
            ]),
            u64::from_le_bytes([
                y_bytes[16],
                y_bytes[17],
                y_bytes[18],
                y_bytes[19],
                y_bytes[20],
                y_bytes[21],
                y_bytes[22],
                y_bytes[23],
            ]),
            u64::from_le_bytes([
                y_bytes[24],
                y_bytes[25],
                y_bytes[26],
                y_bytes[27],
                y_bytes[28],
                y_bytes[29],
                y_bytes[30],
                y_bytes[31],
            ]),
        ];

        x.populate_limbs(&mut w, &x_limbs);
        y.populate_limbs(&mut w, &y_limbs);

        // Create expected compressed public key format (embedded logic)
        let mut expected_compressed = [0u8; 33];
        expected_compressed[0] = prefix;
        for i in 0..32 {
            expected_compressed[1 + i] = x_bytes[31 - i];
        }

        // Pack expected compressed bytes into 32-bit words for comparison
        let mut expected_word_values = [0u32; 9];
        for i in 0..9 {
            let word_start = i * 4;
            if word_start < 33 {
                let bytes_in_word = std::cmp::min(4, 33 - word_start);
                let mut word = 0u32;
                for j in 0..bytes_in_word {
                    word |= (expected_compressed[word_start + j] as u32) << (24 - j * 8);
                }
                expected_word_values[i] = word;
            }
        }

        for i in 0..9 {
            w[expected_words[i]] = Word::from_u64(expected_word_values[i] as u64);
        }

        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_p2pkh_circuit_privkey_external() {
        let (sk_be, pk_comp) = gen_priv_pub_pair_from_rng(StdRng::seed_from_u64(0));
        validate_priv_pub_pair(sk_be, pk_comp);
    }

    #[test]
    fn test_compress_simple() {
        // Simple test with known values to debug byte ordering
        let x_bytes = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C,
            0x1D, 0x1E, 0x1F, 0x20,
        ];

        let y_bytes = [
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, // Even y (LSB = 0x02 which is even)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        // Expected compressed: prefix 0x02 + x bytes in big-endian (reverse of LE storage)
        test_compress_helper(x_bytes, y_bytes, 0x02);
    }

    #[test]
    fn test_compress_odd_y() {
        // Point with odd y coordinate
        let x_bytes = [
            0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
            0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
            0x11, 0x11, 0x11, 0x11,
        ];

        let y_bytes = [
            0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, // LSB is odd
            0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
            0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
        ];

        // Expected compressed format: 0x03 prefix + x coordinate big-endian (reverse of LE storage)
        test_compress_helper(x_bytes, y_bytes, 0x03);
    }

    #[test]
    fn test_compress_even_y() {
        // Point with even y coordinate
        let x_bytes = [
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
            0xAA, 0xAA, 0xAA, 0xAA,
        ];

        let y_bytes = [
            0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, // LSB is even
            0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
            0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        ];

        // Expected compressed format: 0x02 prefix + x coordinate big-endian (reverse of LE storage)
        test_compress_helper(x_bytes, y_bytes, 0x02);
    }
}
