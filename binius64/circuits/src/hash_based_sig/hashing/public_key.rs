// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::base::circuit_tweaked_keccak;
// Note: PublicKeyTweak reuses TREE_TWEAK (0x01) for consistency with XMSS spec
pub use super::tree::TREE_TWEAK as PUBLIC_KEY_TWEAK;
use crate::{fixed_byte_vec::ByteVec, keccak::Keccak256};

/// A circuit that verifies a public key hash for XMSS.
///
/// This circuit verifies Keccak-256 of a message that's been tweaked with
/// multiple public key hashes: `Keccak256(domain_param || 0x01 || pk_hash_0 || pk_hash_1 || ... ||
/// pk_hash_n)`
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param_wires` - The cryptographic domain parameter wires, where each wire holds 8 bytes
///   as a 64-bit LE-packed value
/// * `domain_param_len` - The actual domain parameter length in bytes
/// * `pk_hashes` - The public key end hashes (32 bytes each as 4x64-bit LE-packed wires)
/// * `digest` - Output: The computed Keccak-256 digest (32 bytes as 4x64-bit LE-packed wires)
///
/// # Returns
///
/// A `Keccak` circuit that needs to be populated with the tweaked message and digest
pub fn circuit_public_key_hash(
    builder: &CircuitBuilder,
    domain_param_wires: Vec<Wire>,
    domain_param_len: usize,
    pk_hashes: &[[Wire; 4]],
    digest: [Wire; 4],
) -> Keccak256 {
    let num_hashes = pk_hashes.len();
    let message_len = domain_param_len + 1 + (num_hashes * 32); // +1 for tweak byte
    assert_eq!(domain_param_wires.len(), domain_param_len.div_ceil(8));

    // Build additional terms for all public key hashes
    let mut additional_terms = Vec::new();

    // Add all public key hashes
    for pk_hash in pk_hashes {
        let hash_term = ByteVec {
            len_bytes: builder.add_constant_64(32),
            data: pk_hash.to_vec(),
        };
        additional_terms.push(hash_term);
    }

    circuit_tweaked_keccak(
        builder,
        domain_param_wires,
        domain_param_len,
        PUBLIC_KEY_TWEAK,
        additional_terms,
        message_len,
        digest,
    )
}

/// Build the tweaked message for public key hashing.
///
/// Constructs the complete message for Keccak-256 hashing by concatenating:
/// `domain_param || 0x01 || pk_hash_0 || pk_hash_1 || ... || pk_hash_n`
///
/// This function is typically used when populating witness data for the
/// `circuit_public_key_hash` circuit.
///
/// # Arguments
///
/// * `domain_param_bytes` - The cryptographic domain parameter bytes
/// * `pk_hashes` - Array of 32-byte public key hashes
///
/// # Returns
///
/// A vector containing the complete tweaked message ready for hashing
pub fn build_public_key_hash(domain_param_bytes: &[u8], pk_hashes: &[[u8; 32]]) -> Vec<u8> {
    let mut message = Vec::new();
    message.extend_from_slice(domain_param_bytes);
    message.push(PUBLIC_KEY_TWEAK);
    for pk_hash in pk_hashes {
        message.extend_from_slice(pk_hash);
    }
    message
}

/// Computes the public key hash from Winternitz public keys.
///
/// # Arguments
/// * `domain_param` - Cryptographic domain parameter
/// * `pk_hashes` - Array of 32-byte public key hashes
pub fn hash_public_key_keccak(domain_param: &[u8], pk_hashes: &[[u8; 32]]) -> [u8; 32] {
    use sha3::Digest;
    let tweaked_public_key = build_public_key_hash(domain_param, pk_hashes);
    sha3::Keccak256::digest(tweaked_public_key).into()
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::{Circuit, CircuitBuilder, util::pack_bytes_into_wires_le};
    use proptest::prelude::*;
    use sha3::Digest;

    use super::*;

    /// Helper struct for PublicKeyHash testing
    struct PublicKeyTestCircuit {
        circuit: Circuit,
        keccak: Keccak256,
        domain_param_wires: Vec<Wire>,
        domain_param_len: usize,
        pk_hashes: Vec<[Wire; 4]>,
    }

    impl PublicKeyTestCircuit {
        fn new(domain_param_len: usize, num_hashes: usize) -> Self {
            let builder = CircuitBuilder::new();

            let digest: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

            let num_domain_param_wires = domain_param_len.div_ceil(8);
            let domain_param_wires: Vec<Wire> = (0..num_domain_param_wires)
                .map(|_| builder.add_inout())
                .collect();

            let pk_hashes: Vec<[Wire; 4]> = (0..num_hashes)
                .map(|_| std::array::from_fn(|_| builder.add_inout()))
                .collect();

            let keccak = circuit_public_key_hash(
                &builder,
                domain_param_wires.clone(),
                domain_param_len,
                &pk_hashes,
                digest,
            );

            let circuit = builder.build();

            Self {
                circuit,
                keccak,
                domain_param_wires,
                domain_param_len,
                pk_hashes,
            }
        }

        /// Populate witness and verify constraints with given test data
        fn populate_and_verify(
            &self,
            domain_param_bytes: &[u8],
            pk_hashes_bytes: &[[u8; 32]],
            message: &[u8],
            digest: [u8; 32],
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut w = self.circuit.new_witness_filler();

            // Populate domain param
            assert_eq!(domain_param_bytes.len(), self.domain_param_len);
            pack_bytes_into_wires_le(&mut w, &self.domain_param_wires, domain_param_bytes);

            // Populate public key hashes
            assert_eq!(pk_hashes_bytes.len(), self.pk_hashes.len());
            for (wires, bytes) in self.pk_hashes.iter().zip(pk_hashes_bytes.iter()) {
                pack_bytes_into_wires_le(&mut w, wires, bytes);
            }

            // Populate message for Keccak
            let expected_len = self.domain_param_len + 1 + (self.pk_hashes.len() * 32);
            assert_eq!(
                message.len(),
                expected_len,
                "Message length {} doesn't match expected length {}",
                message.len(),
                expected_len
            );
            self.keccak.populate_message(&mut w, message);

            // Populate digest
            self.keccak.populate_digest(&mut w, digest);

            self.circuit.populate_wire_witness(&mut w)?;
            let cs = self.circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec())?;
            Ok(())
        }
    }

    #[test]
    fn test_public_key_hash_basic() {
        let test_circuit = PublicKeyTestCircuit::new(32, 3);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let pk_hashes = [
            *b"first_public_key_hash_32_bytes!!",
            *b"second_public_key_hash_32_bytes!",
            *b"third_public_key_hash_32_bytes!!",
        ];

        let message = build_public_key_hash(domain_param_bytes, &pk_hashes);

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                &pk_hashes,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_public_key_hash_with_18_byte_domain_param() {
        // Test with 18-byte domain param as per XMSS specifications
        let test_circuit = PublicKeyTestCircuit::new(18, 2);

        let domain_param_bytes: &[u8; 18] = b"test_param_18bytes";
        let pk_hashes = [
            *b"first_public_key_hash_32_bytes!!",
            *b"second_public_key_hash_32_bytes!",
        ];

        let message = build_public_key_hash(domain_param_bytes, &pk_hashes);

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                &pk_hashes,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_public_key_hash_single_hash() {
        let test_circuit = PublicKeyTestCircuit::new(32, 1);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let pk_hashes = [*b"single_public_key_hash_32_bytes!"];

        let message = build_public_key_hash(domain_param_bytes, &pk_hashes);

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                &pk_hashes,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_public_key_hash_many_hashes() {
        // Test with many hashes (e.g., Winternitz with 72 chains)
        let num_hashes = 72;
        let test_circuit = PublicKeyTestCircuit::new(18, num_hashes);

        let domain_param_bytes: &[u8; 18] = b"test_param_18bytes";

        // Generate deterministic hashes for testing
        let mut pk_hashes = Vec::new();
        for i in 0..num_hashes {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            hash[1] = (i >> 8) as u8;
            for j in 2..32 {
                hash[j] = ((i * j) % 256) as u8;
            }
            pk_hashes.push(hash);
        }

        let message = build_public_key_hash(domain_param_bytes, &pk_hashes);

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                &pk_hashes,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_public_key_hash_wrong_digest() {
        let test_circuit = PublicKeyTestCircuit::new(32, 2);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let pk_hashes = [
            *b"first_public_key_hash_32_bytes!!",
            *b"second_public_key_hash_32_bytes!",
        ];

        let message = build_public_key_hash(domain_param_bytes, &pk_hashes);

        // Populate with WRONG digest - this should cause verification to fail
        let wrong_digest = [0u8; 32];

        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            &pk_hashes,
            &message,
            wrong_digest,
        );

        assert!(result.is_err(), "Expected error for wrong digest");
    }

    #[test]
    fn test_public_key_hash_ensures_tweak_byte() {
        // This test verifies that the PUBLIC_KEY_TWEAK byte (0x01) is correctly inserted
        let test_circuit = PublicKeyTestCircuit::new(16, 1);

        let domain_param_bytes = b"param_16_bytes!!";
        let pk_hashes = [*b"single_public_key_hash_32_bytes!"];

        let message = build_public_key_hash(domain_param_bytes, &pk_hashes);

        // Verify the tweak byte is at the correct position
        assert_eq!(message[16], PUBLIC_KEY_TWEAK);
        assert_eq!(message.len(), 16 + 1 + 32); // domain_param + tweak + one hash

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                &pk_hashes,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    proptest! {
        #[test]
        fn test_public_key_hash_property_based(
            domain_param_len in 1usize..=100,
            num_hashes in 1usize..=10,
        ) {
            use rand::SeedableRng;
            use rand::prelude::StdRng;

            let mut rng = StdRng::seed_from_u64(0);

            // Generate random domain param bytes
            let mut domain_param_bytes = vec![0u8; domain_param_len];
            rng.fill_bytes(&mut domain_param_bytes);

            // Generate random public key hashes
            let mut pk_hashes = Vec::new();
            for _ in 0..num_hashes {
                let mut hash = [0u8; 32];
                rng.fill_bytes(&mut hash);
                pk_hashes.push(hash);
            }

            // Create circuit
            let test_circuit = PublicKeyTestCircuit::new(domain_param_len, num_hashes);

            // Build message and compute digest
            let message = build_public_key_hash(&domain_param_bytes, &pk_hashes);

            // Verify message structure
            prop_assert_eq!(message.len(), domain_param_len + 1 + (num_hashes * 32));
            prop_assert_eq!(message[domain_param_len], PUBLIC_KEY_TWEAK);

            let expected_digest: [u8; 32] = sha3::Keccak256::digest(&message).into();

            // Verify circuit
            test_circuit
                .populate_and_verify(&domain_param_bytes, &pk_hashes, &message, expected_digest)
                .unwrap();
        }
    }
}
