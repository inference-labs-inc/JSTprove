// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};
use num_integer::Integer;

use super::fixed_byte_vec::ByteVec;
use crate::{
    bignum::{BigUint, ModReduce, assert_eq, optimal_mul, optimal_sqr},
    sha256::Sha256,
};

/// Convert a FixedByteVec with little-endian wire packing to a BigUint.
///
/// This function converts a FixedByteVec with little-endian packed wires
/// to a BigUint representing a big-endian number.
fn fixedbytevec_le_to_biguint(builder: &mut CircuitBuilder, byte_vec: &ByteVec) -> BigUint {
    // With LE packing, each wire contains 8 bytes as: byte0 | byte1<<8 | ... | byte7<<56
    // For a BE number, we need to both reverse wire order AND byte-swap within each wire
    let mut limbs = Vec::new();
    // Process wires in reverse order (big-endian to little-endian conversion)
    for packed_wire in byte_vec.data.clone().into_iter().rev() {
        // Extract bytes from LE-packed wire and repack in reverse order
        let mut bytes = Vec::with_capacity(8);
        for i in 0..8 {
            let shift_amount = i * 8;
            let byte = builder.shr(packed_wire, shift_amount as u32);
            let byte_masked = builder.band(byte, builder.add_constant_64(0xFF));
            bytes.push(byte_masked);
        }
        // Repack bytes in reverse order (byte-swap)
        let mut swapped_limb = bytes[7];
        for i in 1..8 {
            let shifted = builder.shl(bytes[7 - i], (i * 8) as u32);
            swapped_limb = builder.bor(swapped_limb, shifted);
        }
        limbs.push(swapped_limb);
    }
    BigUint { limbs }
}

/// RS256 signature verification circuit (internal implementation)
///
/// This circuit verifies a `signature` for a given `message` according to the
/// signature verification algorithm RSASSA-PKCS1-v1_5, using SHA-256 as a
/// hash.
///
/// This signature verification algorithm is used in JWT signatures which have
/// the "alg" header set to "RS256".
/// <https://datatracker.ietf.org/doc/html/rfc7518#section-3.1>
pub struct Rs256Verify {
    /// The message to verify (packed as 64-bit words, 8 bytes per wire)
    pub message: ByteVec,
    /// The RSA signature as a FixedByteVec (the primary input interface)
    pub signature: ByteVec,
    /// The RSA modulus as a FixedByteVec (the primary input interface)
    pub modulus: ByteVec,
    /// Wires associated with intermediate RSA computations
    pub rsa_intermediates: RsaIntermediates,
    /// SHA256 circuit for hashing the message
    pub sha256: Sha256,
}

impl Rs256Verify {
    /// Create a new RS256 verification circuit
    ///
    /// This constructor accepts inputs with little-endian wire packing, which
    /// is convenient when composing with other circuits (e.g base64, concat)
    /// which use the same wire packing.
    ///
    /// RS256 uses the public exponent 2^16 + 1 (65537). The circuit verifies
    /// that the encoded message (EM) has the following properties:
    ///
    /// - `EM = signature^65537 mod modulus`
    /// - `EM` has a valid PKCS#1 v1.5 prefix
    /// - The hash stored in `EM` is equal to the SHA-256 hash of the provided message.
    ///
    /// # Arguments
    /// * `builder` - Circuit builder
    /// * `message` - A FixedByteVec containing the plaintext message
    /// * `signature` - The RSA signature with little-endian wire packing (256 bytes for 2048-bit
    ///   RSA)
    /// * `modulus` - The RSA modulus with little-endian wire packing (256 bytes for 2048-bit RSA)
    ///
    /// # Panics
    /// * If signature or modulus don't have at least 256 bytes
    pub fn new(
        builder: &mut CircuitBuilder,
        message: ByteVec,
        signature: ByteVec,
        modulus: ByteVec,
    ) -> Self {
        assert!(
            signature.data.len() >= 32,
            "signature must have at least 256 bytes for 2048-bit RSA"
        );
        assert!(
            modulus.data.len() >= 32,
            "modulus must have at least 256 bytes for 2048-bit RSA"
        );

        // Truncate signature to exactly 256 bytes if it's larger
        // This is needed because:
        // 1. RSA signatures are always exactly 256 bytes for 2048-bit RSA
        // 2. The input might be padded for other circuit requirements (e.g., Base64 alignment to
        //    264 bytes)
        let signature = if signature.data.len() > 32 {
            signature.truncate(builder, 32)
        } else {
            signature.clone()
        };

        let signature_bignum = fixedbytevec_le_to_biguint(builder, &signature);
        builder.assert_eq(
            "signature_bytes_len",
            signature.len_bytes,
            builder.add_constant_64(256),
        );

        let modulus_bignum = fixedbytevec_le_to_biguint(builder, &modulus);
        builder.assert_eq(
            "modulus_bytes_len",
            modulus.len_bytes,
            builder.add_constant_64(256),
        );

        let sha256_builder = builder.subcircuit("sha256");
        let expected_hash_wires: [Wire; 4] = std::array::from_fn(|_| sha256_builder.add_witness());
        let sha256 = Sha256::new(
            &sha256_builder,
            message.len_bytes,
            expected_hash_wires,
            message.data.clone(),
        );
        let expected_hash = BigUint {
            limbs: expected_hash_wires.to_vec(),
        };

        let rsa_intermediates = RsaIntermediates::new_witness(builder);

        modexp_65537_verify(
            builder,
            &signature_bignum,
            &modulus_bignum,
            &rsa_intermediates.square_quotients,
            &rsa_intermediates.square_remainders,
            &rsa_intermediates.mul_quotient,
            &rsa_intermediates.mul_remainder,
        );

        // Validate PKCS#1 v1.5 prefix structure
        // The EM (Encoded Message) has the following format (in big-endian):
        // Bytes 0-1: 0x00 0x01
        // Bytes 2-203: 0xFF padding (202 bytes)
        // Byte 204: 0x00 separator
        // Bytes 205-223: SHA-256 DigestInfo (19 bytes)
        // Bytes 224-255: SHA-256 hash (32 bytes)

        // When converted to little-endian limbs (as used in BigNum):
        // - Limbs 0-3: SHA-256 hash (bytes 224-255 in big-endian)
        // - Limbs 4-31: PKCS#1 v1.5 prefix (bytes 0-223 in big-endian)

        // Pre-computed expected limbs for PKCS#1 v1.5 prefix with SHA-256
        // These values represent the PKCS#1 v1.5 structure when converted from
        // big-endian bytes to little-endian u64 limbs as a 256-byte BigUint
        const EXPECTED_PREFIX_LIMBS: [u64; 28] = [
            // Limb 4-6: DigestInfo bytes
            0x0304020105000420,
            0x0d06096086480165,
            0xffffffff00303130,
            // Limbs 7-30: All padding (0xFF)
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            // Limb 31: Header bytes (0x00, 0x01) and padding
            0x0001ffffffffffff,
        ];

        // Create expected EM (Encoded Message) by combining hash limbs and prefix constants
        let prefix_wires = EXPECTED_PREFIX_LIMBS.map(|l| builder.add_constant_64(l));
        let expected_em = BigUint {
            limbs: expected_hash
                .limbs
                .iter()
                .copied()
                .rev()
                .chain(prefix_wires)
                .collect(),
        };

        assert_eq(
            builder,
            "mul_remainder_expected_em",
            &rsa_intermediates.mul_remainder,
            &expected_em,
        );

        Self {
            message,
            signature,
            modulus,
            rsa_intermediates,
            sha256,
        }
    }

    /// Populate the message length
    pub fn populate_len_bytes(&self, w: &mut WitnessFiller, len_bytes: usize) {
        self.sha256.populate_len_bytes(w, len_bytes);
    }

    /// Populate the RSA signature, modulus and intermediate computations
    pub fn populate_rsa(&self, w: &mut WitnessFiller, signature: &[u8], modulus: &[u8]) {
        self.populate_signature(w, signature);
        self.populate_modulus(w, modulus);
        self.rsa_intermediates
            .populate_witness(w, signature, modulus);
    }

    pub fn populate_intermediates(&self, w: &mut WitnessFiller, signature: &[u8], modulus: &[u8]) {
        self.rsa_intermediates
            .populate_witness(w, signature, modulus);
    }

    /// Populate the message
    ///
    /// # Panics
    /// Panics if message.len() > self.message.len() * 8
    pub fn populate_message(&self, w: &mut WitnessFiller, message: &[u8]) {
        self.sha256.populate_message(w, message);
    }

    /// Populate the RSA modulus
    ///
    /// # Panics
    /// Panics if modulus_bytes.len() != 256
    pub fn populate_modulus(&self, w: &mut WitnessFiller, modulus_bytes: &[u8]) {
        assert_eq!(
            modulus_bytes.len(),
            256,
            "modulus must be exactly 256 bytes"
        );
        self.modulus.populate_bytes_le(w, modulus_bytes);
    }

    /// Populate the RSA signature
    ///
    /// # Arguments
    /// * `w` - Witness filler
    /// * `signature_bytes` - The bytes of an RSA signature
    ///
    /// # Panics
    /// Panics if signature_bytes.len() != 256
    pub fn populate_signature(&self, w: &mut WitnessFiller, signature_bytes: &[u8]) {
        assert_eq!(
            signature_bytes.len(),
            256,
            "signature must be exactly 256 bytes"
        );
        self.signature.populate_bytes_le(w, signature_bytes);
    }
}

/// Verify base^65537 mod modulus using provided intermediate values
fn modexp_65537_verify(
    builder: &CircuitBuilder,
    base: &BigUint,
    modulus: &BigUint,
    square_quotients: &[BigUint],
    square_remainders: &[BigUint],
    mul_quotient: &BigUint,
    mul_remainder: &BigUint,
) {
    let mut result = base.clone();

    for i in 0..16 {
        let builder = builder.subcircuit(format!("square[{i}]"));
        let squared = optimal_sqr(&builder, &result);
        let circuit = ModReduce::new(
            &builder,
            squared,
            modulus.clone(),
            square_quotients[i].clone(),
            square_remainders[i].clone(),
        );
        result = circuit.remainder;
    }

    let builder = builder.subcircuit("final_multiply");
    let multiplied = optimal_mul(&builder, &result, base);
    let _mod_reduce_multiplied = ModReduce::new(
        &builder,
        multiplied,
        modulus.clone(),
        mul_quotient.clone(),
        mul_remainder.clone(),
    );
}

/// Wires associated with intermediate RSA computations
pub struct RsaIntermediates {
    /// Quotients for each of the 16 squaring operations
    square_quotients: Vec<BigUint>,
    /// Remainders for each of the 16 squaring operations
    square_remainders: Vec<BigUint>,
    /// Quotient for the final multiplication
    mul_quotient: BigUint,
    /// Remainder for the final multiplication (the EM - Encoded Message)
    mul_remainder: BigUint,
}

impl RsaIntermediates {
    fn new_witness(builder: &CircuitBuilder) -> Self {
        let mut square_quotients = Vec::new();
        let mut square_remainders = Vec::new();
        for _ in 0..16 {
            square_quotients.push(BigUint::new_witness(builder, 32));
            square_remainders.push(BigUint::new_witness(builder, 32));
        }
        let mul_quotient = BigUint::new_witness(builder, 32);
        let mul_remainder = BigUint::new_witness(builder, 32);

        RsaIntermediates {
            square_quotients,
            square_remainders,
            mul_quotient,
            mul_remainder,
        }
    }

    /// Populate RSA intermediate values for RS256 verification
    ///
    /// This function populates the quotients and remainders needed for
    /// verifying RSA signatures with public exponent 65537 (2^16 + 1).
    ///
    /// # Arguments
    /// * `signature` - The bytes of a RSA signature
    /// * `modulus_limbs` - The bytes of a RSA modulus
    pub fn populate_witness(&self, w: &mut WitnessFiller, signature: &[u8], modulus: &[u8]) {
        assert_eq!(signature.len(), 256, "signature must be exactly 256 bytes");
        assert_eq!(modulus.len(), 256, "modulus must be exactly 256 bytes");

        let signature_value = num_bigint::BigUint::from_bytes_be(signature);
        let modulus_value = num_bigint::BigUint::from_bytes_be(modulus);

        let mut square_quotients = Vec::new();
        let mut square_remainders = Vec::new();

        let mut result = signature_value.clone();
        for _ in 0..16 {
            let squared = &result * &result;
            let (q, r) = squared.div_rem(&modulus_value);

            let mut q_limbs = q.to_u64_digits();
            q_limbs.resize(32, 0u64);
            square_quotients.push(q_limbs);

            let mut r_limbs = r.to_u64_digits();
            r_limbs.resize(32, 0u64);
            square_remainders.push(r_limbs);

            result = r;
        }

        // Final multiplication
        let multiplied = &result * &signature_value;
        let (mul_q, mul_r) = multiplied.div_rem(&modulus_value);

        let mut mul_quotient = mul_q.to_u64_digits();
        mul_quotient.resize(32, 0u64);

        let mut mul_remainder = mul_r.to_u64_digits();
        mul_remainder.resize(32, 0u64);

        self.populate_square_quotients(w, &square_quotients);
        self.populate_square_remainders(w, &square_remainders);
        self.populate_mul_quotient(w, &mul_quotient);
        self.populate_mul_remainder(w, &mul_remainder);
    }

    /// Populate the square quotients for the 16 squaring operations
    ///
    /// # Panics
    /// Panics if square_quotient_limbs.len() != 16 or if any quotient doesn't have 32 limbs.
    fn populate_square_quotients(&self, w: &mut WitnessFiller, square_quotient_limbs: &[Vec<u64>]) {
        assert_eq!(
            square_quotient_limbs.len(),
            16,
            "must provide 16 square quotients"
        );
        for (i, q_limbs) in square_quotient_limbs.iter().enumerate() {
            assert_eq!(
                q_limbs.len(),
                self.square_quotients[i].limbs.len(),
                "square_quotient[{i}] must have {} limbs",
                self.square_quotients[i].limbs.len()
            );
            self.square_quotients[i].populate_limbs(w, q_limbs);
        }
    }

    /// Populate the square remainders for the 16 squaring operations
    ///
    /// # Panics
    /// Panics if square_remainder_limbs.len() != 16 or if any remainder doesn't have 32 limbs
    fn populate_square_remainders(
        &self,
        w: &mut WitnessFiller,
        square_remainder_limbs: &[Vec<u64>],
    ) {
        assert_eq!(
            square_remainder_limbs.len(),
            16,
            "must provide 16 square remainders"
        );
        for (i, r_limbs) in square_remainder_limbs.iter().enumerate() {
            assert_eq!(
                r_limbs.len(),
                32,
                "square_remainder[{i}] must have 32 limbs"
            );
            self.square_remainders[i].populate_limbs(w, r_limbs);
        }
    }

    /// Populate the multiplication quotient
    ///
    /// # Panics
    /// Panics if mul_quotient_limbs.len() != 32
    fn populate_mul_quotient(&self, w: &mut WitnessFiller, mul_quotient_limbs: &[u64]) {
        assert_eq!(
            mul_quotient_limbs.len(),
            self.mul_quotient.limbs.len(),
            "mul_quotient must have {} limbs",
            self.mul_quotient.limbs.len()
        );
        self.mul_quotient.populate_limbs(w, mul_quotient_limbs);
    }

    /// Populate the multiplication remainder (the EM - Encoded Message)
    ///
    /// # Panics
    /// Panics if mul_remainder_limbs.len() != 32
    fn populate_mul_remainder(&self, w: &mut WitnessFiller, mul_remainder_limbs: &[u64]) {
        assert_eq!(
            mul_remainder_limbs.len(),
            32,
            "mul_remainder must have 32 limbs"
        );
        self.mul_remainder.populate_limbs(w, mul_remainder_limbs);
    }
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use hex_literal::hex;
    use num_bigint::BigUint;
    use rand::{SeedableRng, TryRngCore, rngs::StdRng};
    use rsa::{
        BigUint as RsaBigUint, RsaPrivateKey, RsaPublicKey,
        sha2::{Digest, Sha256},
        traits::{PrivateKeyParts, PublicKeyParts},
    };

    use super::*;

    /// Create a deterministic test RSA key
    /// This is a 2048-bit RSA key generated with ChaCha8Rng seed 42
    fn test_rsa_key() -> RsaPrivateKey {
        let p = RsaBigUint::from_bytes_be(&hex!(
            "c8b4e97508c3d0fad0062e8ee475909d5315bc9433e9b8a174a52b8f024e7d6b"
            "ea80a56901555021b2d44f727aa287b84de8bac5ceef88d03b259f8ac91bda42"
            "e653e27596d8090e08e9dac47dcd288e1c0e95ac74d7428cd0479c8514bc3538"
            "7380a480873c7f519ece6f5ea4356c81bd7ec31c126c1f097b84bb33c8acd565"
        ));
        let q = RsaBigUint::from_bytes_be(&hex!(
            "efffcc7f550f977db26971fb6a0f036d61cccde351c394fe177cd36a0a7dde60"
            "8cd263d8ca382031fc0f16bef5ebb2125ab1b8e837c71c006a8639c090a7ebac"
            "530de579bca2ea7ad175c8a31d45078130e0ad15cf23139d230f30c106259c7a"
            "55024f4e51a97b1b38b7ed4dfe05a0706bf53a067e7f0ee18dc685b53300708b"
        ));
        let e = RsaBigUint::from(65537u32);
        RsaPrivateKey::from_p_q(p, q, e).expect("valid key")
    }

    fn populate_circuit(
        circuit: &Rs256Verify,
        w: &mut WitnessFiller,
        signature_bytes: &[u8],
        message_bytes: &[u8],
        modulus_bytes: &[u8],
    ) {
        let hash = Sha256::digest(message_bytes);
        circuit.populate_rsa(w, signature_bytes, modulus_bytes);
        circuit.populate_len_bytes(w, message_bytes.len());
        circuit.populate_message(w, message_bytes);
        circuit.sha256.populate_digest(w, hash.into());
    }

    fn setup_circuit(builder: &mut CircuitBuilder, max_len: usize) -> Rs256Verify {
        // max_len is now denominated in wires, not bytes.
        // this is a consistent change made throughout the codebase.
        let signature_bytes = ByteVec::new_inout(builder, 32);
        let modulus_bytes = ByteVec::new_inout(builder, 32);
        let message = ByteVec::new_witness(builder, max_len);

        Rs256Verify::new(builder, message, signature_bytes, modulus_bytes)
    }

    #[test]
    fn test_real_rsa_signature_verification_with_message() {
        let mut builder = CircuitBuilder::new();
        let circuit = setup_circuit(&mut builder, 32);
        let cs = builder.build();

        let private_key = test_rsa_key();
        let public_key = RsaPublicKey::from(&private_key);
        let mut rng = StdRng::seed_from_u64(42);
        let mut message_bytes = [0u8; 256];
        rng.try_fill_bytes(&mut message_bytes).unwrap();

        // Sign with PKCS1v15 padding scheme
        let digest = Sha256::digest(message_bytes);
        let signature_bytes = private_key
            .sign(rsa::Pkcs1v15Sign::new::<Sha256>(), &digest)
            .expect("failed to sign");
        let modulus_bytes = public_key.n().to_bytes_be();

        let mut w = cs.new_witness_filler();
        populate_circuit(
            &circuit,
            &mut w,
            &signature_bytes,
            &message_bytes,
            &modulus_bytes,
        );

        cs.populate_wire_witness(&mut w).unwrap();
        verify_constraints(cs.constraint_system(), &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_real_rsa_signature_with_invalid_prefix() {
        let mut builder = CircuitBuilder::new();
        let max_message_len = 256;
        let circuit = setup_circuit(&mut builder, max_message_len);
        let cs = builder.build();

        let private_key = test_rsa_key();
        let public_key = RsaPublicKey::from(&private_key);

        let message = b"Test message for RS256 verification with invalid prefix";

        // Compute signature that would produce a corrupted EM
        // signature = EM^d mod n
        let corrupted_em = BigUint::ZERO;
        let d_bytes = private_key.d().to_bytes_le();
        let n_bytes = private_key.n().to_bytes_le();
        let d = BigUint::from_bytes_le(&d_bytes);
        let n = BigUint::from_bytes_le(&n_bytes);
        let corrupted_signature = corrupted_em.modpow(&d, &n);

        let mut signature_bytes = corrupted_signature.to_bytes_be();
        signature_bytes.resize(256, 0u8);
        let modulus_bytes = public_key.n().to_bytes_be();

        let mut w = cs.new_witness_filler();
        populate_circuit(&circuit, &mut w, &signature_bytes, message, &modulus_bytes);

        let result = cs.populate_wire_witness(&mut w);
        assert!(
            result.is_err(),
            "Circuit should fail when PKCS#1 v1.5 prefix is corrupted"
        );
    }

    #[test]
    fn test_real_rsa_signature_verification_with_wrong_message() {
        let mut builder = CircuitBuilder::new();
        let max_message_len = 256;
        let circuit = setup_circuit(&mut builder, max_message_len);
        let cs = builder.build();

        let private_key = test_rsa_key();
        let public_key = RsaPublicKey::from(&private_key);

        let message = b"Test message for RS256 verification with wrong message";
        let digest = Sha256::digest(message);
        let signature_bytes = private_key
            .sign(rsa::Pkcs1v15Sign::new::<Sha256>(), &digest)
            .expect("failed to sign");

        let signature_bytes = BigUint::from_bytes_be(&signature_bytes).to_bytes_be();
        let modulus_bytes = public_key.n().to_bytes_be();

        // Use a WRONG message
        let wrong_message = b"This is a completely different message!";

        let mut w = cs.new_witness_filler();
        populate_circuit(
            &circuit,
            &mut w,
            &signature_bytes,
            wrong_message,
            &modulus_bytes,
        );

        let result = cs.populate_wire_witness(&mut w);
        assert!(
            result.is_err(),
            "Circuit should fail when message doesn't match signature"
        );
    }
}
