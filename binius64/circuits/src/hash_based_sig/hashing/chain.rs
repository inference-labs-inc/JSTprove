// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::base::circuit_tweaked_keccak;
use crate::{fixed_byte_vec::ByteVec, keccak::Keccak256};

pub const CHAIN_TWEAK: u8 = 0x00;

/// Fixed overhead in the message beyond the parameter length:
/// - 1 byte: tweak_byte
/// - 32 bytes: hash value
/// - 8 bytes: chain_index
/// - 8 bytes: position
pub const FIXED_MESSAGE_OVERHEAD: usize = 1 + 32 + 8 + 8;

/// A circuit that verifies a chain-tweaked Keccak-256 computation.
///
/// This circuit verifies Keccak-256 of a message that's been tweaked with
/// chain-specific parameters: `Keccak256(domain_param || 0x00 || hash || chain_index || position)`
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param_wires` - The cryptographic domain parameter wires, where each wire holds 8 bytes
///   as a 64-bit LE-packed value
/// * `domain_param_len` - The actual domain parameter length in bytes
/// * `hash` - The hash value to be tweaked (32 bytes as 4x64-bit LE-packed wires)
/// * `chain_index` - Index of this chain (as 64-bit LE-packed value in wire)
/// * `position` - Position within the chain (as 64-bit LE-packed value in wire)
/// * `digest` - Output: The computed Keccak-256 digest (32 bytes as 4x64-bit LE-packed wires)
///
/// # Returns
///
/// A `Keccak` circuit that needs to be populated with the tweaked message and digest
pub fn circuit_chain_hash(
    builder: &CircuitBuilder,
    domain_param_wires: Vec<Wire>,
    domain_param_len: usize,
    hash: [Wire; 4],
    chain_index: Wire,
    position: Wire,
    digest: [Wire; 4],
) -> Keccak256 {
    let message_len = domain_param_len + FIXED_MESSAGE_OVERHEAD;
    assert_eq!(domain_param_wires.len(), domain_param_len.div_ceil(8));

    // Build additional terms for hash, chain_index, and position
    let mut additional_terms = Vec::new();

    let hash_term = ByteVec {
        len_bytes: builder.add_constant_64(32),
        data: hash.to_vec(),
    };
    additional_terms.push(hash_term);

    let chain_index_term = ByteVec {
        len_bytes: builder.add_constant_64(8),
        data: vec![chain_index],
    };
    additional_terms.push(chain_index_term);

    let position_term = ByteVec {
        len_bytes: builder.add_constant_64(8),
        data: vec![position],
    };
    additional_terms.push(position_term);

    circuit_tweaked_keccak(
        builder,
        domain_param_wires,
        domain_param_len,
        CHAIN_TWEAK,
        additional_terms,
        message_len,
        digest,
    )
}

/// Build the tweaked message from components.
///
/// Constructs the complete message for Keccak-256 hashing by concatenating:
/// `domain_param || 0x00 || hash || chain_index || position`
///
/// This function is typically used when populating witness data for the
/// `circuit_chain_hash` circuit.
///
/// # Arguments
///
/// * `domain_param_bytes` - The cryptographic domain parameter bytes
/// * `hash_bytes` - The 32-byte hash value to be tweaked
/// * `chain_index_value` - The chain index as a u64 (will be encoded as little-endian)
/// * `position_value` - The position within the chain as a u64 (will be encoded as little-endian)
///
/// # Returns
///
/// A vector containing the complete tweaked message ready for hashing
pub fn build_chain_hash(
    domain_param_bytes: &[u8],
    hash_bytes: &[u8; 32],
    chain_index_value: u64,
    position_value: u64,
) -> Vec<u8> {
    let mut message = Vec::new();
    message.extend_from_slice(domain_param_bytes);
    message.push(CHAIN_TWEAK);
    message.extend_from_slice(hash_bytes);
    message.extend_from_slice(&chain_index_value.to_le_bytes());
    message.extend_from_slice(&position_value.to_le_bytes());
    message
}

/// Computes a hash chain for Winternitz OTS signature verification.
///
/// # Arguments
/// * `param` - Cryptographic parameter
/// * `chain_index` - Index of the chain being computed
/// * `start_hash` - Starting hash value
/// * `start_pos` - Starting position in the chain
/// * `num_hashes` - Number of hash iterations to perform
pub fn hash_chain_keccak(
    param: &[u8],
    chain_index: usize,
    start_hash: &[u8; 32],
    start_pos: usize,
    num_hashes: usize,
) -> [u8; 32] {
    use sha3::Digest;

    let mut current = *start_hash;
    for i in 0..num_hashes {
        let position = start_pos + i + 1;
        let tweaked_message =
            build_chain_hash(param, &current, chain_index as u64, position as u64);
        current = sha3::Keccak256::digest(tweaked_message).into();
    }
    current
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::{Circuit, CircuitBuilder, util::pack_bytes_into_wires_le};
    use proptest::prelude::*;
    use sha3::Digest;

    use super::*;

    /// Helper struct for ChainHash testing
    struct ChainTestCircuit {
        circuit: Circuit,
        keccak: Keccak256,
        domain_param_wires: Vec<Wire>,
        domain_param_len: usize,
        hash: [Wire; 4],
        chain_index: Wire,
        position: Wire,
    }

    impl ChainTestCircuit {
        fn new(domain_param_len: usize) -> Self {
            let builder = CircuitBuilder::new();

            let hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
            let chain_index = builder.add_inout();
            let position = builder.add_inout();
            let digest: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

            let num_domain_param_wires = domain_param_len.div_ceil(8);
            let domain_param_wires: Vec<Wire> = (0..num_domain_param_wires)
                .map(|_| builder.add_inout())
                .collect();

            let keccak = circuit_chain_hash(
                &builder,
                domain_param_wires.clone(),
                domain_param_len,
                hash,
                chain_index,
                position,
                digest,
            );

            let circuit = builder.build();

            Self {
                circuit,
                keccak,
                domain_param_wires,
                domain_param_len,
                hash,
                chain_index,
                position,
            }
        }

        /// Populate witness and verify constraints with given test data
        fn populate_and_verify(
            &self,
            domain_param_bytes: &[u8],
            hash_bytes: &[u8; 32],
            chain_index_val: u64,
            position_val: u64,
            message: &[u8],
            digest: [u8; 32],
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut w = self.circuit.new_witness_filler();

            // Populate domain param
            assert_eq!(domain_param_bytes.len(), self.domain_param_len);
            pack_bytes_into_wires_le(&mut w, &self.domain_param_wires, domain_param_bytes);

            // Populate hash, chain_index, position
            pack_bytes_into_wires_le(&mut w, &self.hash, hash_bytes);
            w[self.chain_index] = Word::from_u64(chain_index_val);
            w[self.position] = Word::from_u64(position_val);

            // Populate message for Keccak
            let expected_len = self.domain_param_len + FIXED_MESSAGE_OVERHEAD;
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
    fn test_chain_hash_basic() {
        let test_circuit = ChainTestCircuit::new(32);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let hash_bytes = b"hash_value_32_bytes_long!!!!!!!!";
        let chain_index_val = 123u64;
        let position_val = 456u64;

        let message = build_chain_hash(
            domain_param_bytes,
            hash_bytes,
            chain_index_val,
            position_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                hash_bytes,
                chain_index_val,
                position_val,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_chain_hash_with_18_byte_domain_param() {
        // Test with 18-byte domain param as per SPEC_1 and SPEC_2
        let test_circuit = ChainTestCircuit::new(18);

        let domain_param_bytes: &[u8; 18] = b"test_param_18bytes";
        let hash_bytes = b"hash_value_32_bytes_long!!!!!!!!";
        let chain_index_val = 123u64;
        let position_val = 456u64;

        let message = build_chain_hash(
            domain_param_bytes,
            hash_bytes,
            chain_index_val,
            position_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                hash_bytes,
                chain_index_val,
                position_val,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_chain_hash_wrong_digest() {
        let test_circuit = ChainTestCircuit::new(32);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let hash_bytes = b"hash_value_32_bytes_long!!!!!!!!";
        let chain_index_val = 123u64;
        let position_val = 456u64;

        let message = build_chain_hash(
            domain_param_bytes,
            hash_bytes,
            chain_index_val,
            position_val,
        );

        // Populate with WRONG digest - this should cause verification to fail
        let wrong_digest = [0u8; 32];

        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            hash_bytes,
            chain_index_val,
            position_val,
            &message,
            wrong_digest,
        );

        assert!(result.is_err(), "Expected error for wrong digest");
    }

    #[test]
    fn test_chain_hash_wrong_domain_param() {
        let test_circuit = ChainTestCircuit::new(32);

        let correct_domain_param_bytes = b"correct_parameter_32_bytes!!!!!!";
        let wrong_domain_param_bytes = b"wrong___parameter_32_bytes!!!!!!";
        let hash_bytes = b"hash_value_32_bytes_long!!!!!!!!";
        let chain_index_val = 123u64;
        let position_val = 456u64;

        // Message built with correct domain param
        let message = build_chain_hash(
            correct_domain_param_bytes,
            hash_bytes,
            chain_index_val,
            position_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        // Populate with WRONG domain param but correct digest
        let result = test_circuit.populate_and_verify(
            wrong_domain_param_bytes,
            hash_bytes,
            chain_index_val,
            position_val,
            &message,
            expected_digest.into(),
        );

        assert!(
            result.is_err(),
            "Expected error for mismatched domain param"
        );
    }

    #[test]
    fn test_chain_hash_wrong_chain_index() {
        let test_circuit = ChainTestCircuit::new(32);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let hash_bytes = b"hash_value_32_bytes_long!!!!!!!!";
        let correct_chain_index = 123u64;
        let wrong_chain_index = 999u64;
        let position_val = 456u64;

        // Message built with correct chain_index
        let message = build_chain_hash(
            domain_param_bytes,
            hash_bytes,
            correct_chain_index,
            position_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        // Populate with WRONG chain_index but correct digest
        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            hash_bytes,
            wrong_chain_index,
            position_val,
            &message,
            expected_digest.into(),
        );

        assert!(result.is_err(), "Expected error for mismatched chain_index");
    }

    proptest! {
        #[test]
        fn test_chain_hash_property_based(
            domain_param_len in 1usize..=100,
            chain_index in 0u64..=1000,
            position in 0u64..=1000,
        ) {
            use rand::SeedableRng;
            use rand::prelude::StdRng;

            let mut rng = StdRng::seed_from_u64(0);

            // Generate random domain param bytes
            let mut domain_param_bytes = vec![0u8; domain_param_len];
            rng.fill_bytes(&mut domain_param_bytes);

            // Generate random hash
            let mut hash_bytes = [0u8; 32];
            rng.fill_bytes(&mut hash_bytes);

            // Create circuit
            let test_circuit = ChainTestCircuit::new(domain_param_len);

            // Build message and compute digest
            let message = build_chain_hash(
                &domain_param_bytes,
                &hash_bytes,
                chain_index,
                position,
            );

            // Verify message structure
            prop_assert_eq!(message.len(), domain_param_len + FIXED_MESSAGE_OVERHEAD);
            prop_assert_eq!(message[domain_param_len], CHAIN_TWEAK);

            let expected_digest: [u8; 32] = sha3::Keccak256::digest(&message).into();

            // Verify circuit
            test_circuit
                .populate_and_verify(
                    &domain_param_bytes,
                    &hash_bytes,
                    chain_index,
                    position,
                    &message,
                    expected_digest,
                )
                .unwrap();
        }
    }
}
