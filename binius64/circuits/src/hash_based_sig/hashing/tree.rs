// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::base::circuit_tweaked_keccak;
use crate::{fixed_byte_vec::ByteVec, keccak::Keccak256};

pub const TREE_TWEAK: u8 = 0x01;

/// Fixed overhead in the tree node message beyond the parameter length:
/// - 1 byte: tweak_byte (TREE_TWEAK)
/// - 4 bytes: level
/// - 4 bytes: index
/// - 32 bytes: left hash
/// - 32 bytes: right hash
pub const TREE_MESSAGE_OVERHEAD: usize = 1 + 4 + 4 + 32 + 32;

/// A circuit that verifies a tree node hashing for Merkle trees.
///
/// This circuit verifies Keccak-256 of a tree node that's been tweaked with
/// tree-specific parameters: `Keccak256(domain_param || 0x01 || level || index || left || right)`
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param_wires` - The cryptographic domain parameter wires, where each wire holds 8 bytes
///   as a 64-bit LE-packed value
/// * `domain_param_len` - The actual domain parameter length in bytes
/// * `left` - The left child hash (32 bytes as 4x64-bit LE-packed wires)
/// * `right` - The right child hash (32 bytes as 4x64-bit LE-packed wires)
/// * `level` - The level in the tree (as 64-bit value in wire, only lower 4 bytes used)
/// * `index` - The index at this level (as 64-bit value in wire, only lower 4 bytes used)
/// * `digest` - Output: The computed Keccak-256 digest (32 bytes as 4x64-bit LE-packed wires)
///
/// # Returns
///
/// A `Keccak` circuit that needs to be populated with the tweaked message and digest
#[allow(clippy::too_many_arguments)]
pub fn circuit_tree_hash(
    builder: &CircuitBuilder,
    domain_param_wires: Vec<Wire>,
    domain_param_len: usize,
    left: [Wire; 4],
    right: [Wire; 4],
    level: Wire,
    index: Wire,
    digest: [Wire; 4],
) -> Keccak256 {
    let message_len = domain_param_len + TREE_MESSAGE_OVERHEAD;
    assert_eq!(domain_param_wires.len(), domain_param_len.div_ceil(8));

    let mut additional_terms = Vec::new();

    // Add level (4 bytes, truncated from 8-byte wire)
    let level_term = ByteVec {
        len_bytes: builder.add_constant_64(4),
        data: vec![level],
    };
    additional_terms.push(level_term);

    // Add index (4 bytes, truncated from 8-byte wire)
    let index_term = ByteVec {
        len_bytes: builder.add_constant_64(4),
        data: vec![index],
    };
    additional_terms.push(index_term);

    // Add left hash
    let left_term = ByteVec {
        len_bytes: builder.add_constant_64(32),
        data: left.to_vec(),
    };
    additional_terms.push(left_term);

    // Add right hash
    let right_term = ByteVec {
        len_bytes: builder.add_constant_64(32),
        data: right.to_vec(),
    };
    additional_terms.push(right_term);

    circuit_tweaked_keccak(
        builder,
        domain_param_wires,
        domain_param_len,
        TREE_TWEAK,
        additional_terms,
        message_len,
        digest,
    )
}

/// Build the tweaked message for tree node hashing.
///
/// Constructs the complete message for Keccak-256 hashing by concatenating:
/// `domain_param || 0x01 || level || index || left || right`
///
/// This function is typically used when populating witness data for the
/// `circuit_tree_hash` circuit.
///
/// # Arguments
///
/// * `domain_param_bytes` - The cryptographic domain parameter bytes
/// * `left_bytes` - The 32-byte left child hash
/// * `right_bytes` - The 32-byte right child hash
/// * `level` - The level in the tree as a u32 (will be encoded as little-endian)
/// * `index` - The index at this level as a u32 (will be encoded as little-endian)
///
/// # Returns
///
/// A vector containing the complete tweaked message ready for hashing
pub fn build_tree_hash(
    domain_param_bytes: &[u8],
    left_bytes: &[u8; 32],
    right_bytes: &[u8; 32],
    level: u32,
    index: u32,
) -> Vec<u8> {
    let mut message = Vec::new();
    message.extend_from_slice(domain_param_bytes);
    message.push(TREE_TWEAK);
    message.extend_from_slice(&level.to_le_bytes());
    message.extend_from_slice(&index.to_le_bytes());
    message.extend_from_slice(left_bytes);
    message.extend_from_slice(right_bytes);
    message
}

/// Computes a Merkle tree node hash.
///
/// # Arguments
/// * `param` - Cryptographic parameter
/// * `left` - Left child hash
/// * `right` - Right child hash
/// * `level` - Tree level (0 = leaf level)
/// * `index` - Node index at this level
pub fn hash_tree_node_keccak(
    param: &[u8],
    left: &[u8; 32],
    right: &[u8; 32],
    level: u32,
    index: u32,
) -> [u8; 32] {
    use sha3::Digest;
    let tweaked_tree_node = build_tree_hash(param, left, right, level, index);
    sha3::Keccak256::digest(tweaked_tree_node).into()
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::{Circuit, CircuitBuilder, util::pack_bytes_into_wires_le};
    use proptest::prelude::*;
    use sha3::Digest;

    use super::*;

    /// Helper struct for TreeHash testing
    struct TreeTestCircuit {
        circuit: Circuit,
        keccak: Keccak256,
        domain_param_wires: Vec<Wire>,
        domain_param_len: usize,
        left: [Wire; 4],
        right: [Wire; 4],
        level: Wire,
        index: Wire,
    }

    impl TreeTestCircuit {
        fn new(domain_param_len: usize) -> Self {
            let builder = CircuitBuilder::new();

            let left: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
            let right: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
            let level = builder.add_inout();
            let index = builder.add_inout();
            let digest: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

            let num_domain_param_wires = domain_param_len.div_ceil(8);
            let domain_param_wires: Vec<Wire> = (0..num_domain_param_wires)
                .map(|_| builder.add_inout())
                .collect();

            let keccak = circuit_tree_hash(
                &builder,
                domain_param_wires.clone(),
                domain_param_len,
                left,
                right,
                level,
                index,
                digest,
            );

            let circuit = builder.build();

            Self {
                circuit,
                keccak,
                domain_param_wires,
                domain_param_len,
                left,
                right,
                level,
                index,
            }
        }

        /// Populate witness and verify constraints with given test data
        #[allow(clippy::too_many_arguments)]
        fn populate_and_verify(
            &self,
            domain_param_bytes: &[u8],
            left_bytes: &[u8; 32],
            right_bytes: &[u8; 32],
            level_val: u32,
            index_val: u32,
            message: &[u8],
            digest: [u8; 32],
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut w = self.circuit.new_witness_filler();

            // Populate domain param
            assert_eq!(domain_param_bytes.len(), self.domain_param_len);
            pack_bytes_into_wires_le(&mut w, &self.domain_param_wires, domain_param_bytes);

            // Populate left, right, level, index
            pack_bytes_into_wires_le(&mut w, &self.left, left_bytes);
            pack_bytes_into_wires_le(&mut w, &self.right, right_bytes);
            w[self.level] = Word::from_u64(level_val as u64);
            w[self.index] = Word::from_u64(index_val as u64);

            // Populate message for Keccak
            let expected_len = self.domain_param_len + TREE_MESSAGE_OVERHEAD;
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
    fn test_tree_hash_basic() {
        let test_circuit = TreeTestCircuit::new(32);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let left_bytes = b"left_child_hash_32_bytes_long!!!";
        let right_bytes = b"right_child_hash_32_bytes_long!!";
        let level_val = 5u32;
        let index_val = 123u32;

        let message = build_tree_hash(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            level_val,
            index_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                left_bytes,
                right_bytes,
                level_val,
                index_val,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_tree_hash_with_18_byte_domain_param() {
        // Test with 18-byte domain param as per XMSS specifications
        let test_circuit = TreeTestCircuit::new(18);

        let domain_param_bytes: &[u8; 18] = b"test_param_18bytes";
        let left_bytes = b"left_child_hash_32_bytes_long!!!";
        let right_bytes = b"right_child_hash_32_bytes_long!!";
        let level_val = 10u32;
        let index_val = 456u32;

        let message = build_tree_hash(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            level_val,
            index_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                left_bytes,
                right_bytes,
                level_val,
                index_val,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_tree_hash_wrong_digest() {
        let test_circuit = TreeTestCircuit::new(32);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let left_bytes = b"left_child_hash_32_bytes_long!!!";
        let right_bytes = b"right_child_hash_32_bytes_long!!";
        let level_val = 5u32;
        let index_val = 123u32;

        let message = build_tree_hash(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            level_val,
            index_val,
        );

        // Populate with WRONG digest - this should cause verification to fail
        let wrong_digest = [0u8; 32];

        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            level_val,
            index_val,
            &message,
            wrong_digest,
        );

        assert!(result.is_err(), "Expected error for wrong digest");
    }

    #[test]
    fn test_tree_hash_wrong_level() {
        let test_circuit = TreeTestCircuit::new(32);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let left_bytes = b"left_child_hash_32_bytes_long!!!";
        let right_bytes = b"right_child_hash_32_bytes_long!!";
        let correct_level = 5u32;
        let wrong_level = 10u32;
        let index_val = 123u32;

        // Message built with correct level
        let message = build_tree_hash(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            correct_level,
            index_val,
        );

        let expected_digest = sha3::Keccak256::digest(&message);

        // Populate with WRONG level but correct digest
        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            wrong_level,
            index_val,
            &message,
            expected_digest.into(),
        );

        assert!(result.is_err(), "Expected error for mismatched level");
    }

    #[test]
    fn test_tree_hash_ensures_tweak_byte() {
        // This test verifies that the TREE_TWEAK byte (0x01) is correctly inserted
        let test_circuit = TreeTestCircuit::new(16);

        let domain_param_bytes = b"param_16_bytes!!";
        let left_bytes = b"left_child_hash_32_bytes_long!!!";
        let right_bytes = b"right_child_hash_32_bytes_long!!";
        let level_val = 3u32;
        let index_val = 7u32;

        let message = build_tree_hash(
            domain_param_bytes,
            left_bytes,
            right_bytes,
            level_val,
            index_val,
        );

        // Verify the tweak byte is at the correct position
        assert_eq!(message[16], TREE_TWEAK);
        assert_eq!(message.len(), 16 + TREE_MESSAGE_OVERHEAD);

        let expected_digest = sha3::Keccak256::digest(&message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                left_bytes,
                right_bytes,
                level_val,
                index_val,
                &message,
                expected_digest.into(),
            )
            .unwrap();
    }

    proptest! {
        #[test]
        fn test_tree_hash_property_based(
            domain_param_len in 1usize..=100,
            level in 0u32..=20,
            index in 0u32..=1000,
        ) {
            use rand::SeedableRng;
            use rand::prelude::StdRng;

            let mut rng = StdRng::seed_from_u64(0);

            // Generate random domain param bytes
            let mut domain_param_bytes = vec![0u8; domain_param_len];
            rng.fill_bytes(&mut domain_param_bytes);

            // Generate random left and right hashes
            let mut left_bytes = [0u8; 32];
            rng.fill_bytes(&mut left_bytes);
            let mut right_bytes = [0u8; 32];
            rng.fill_bytes(&mut right_bytes);

            // Create circuit
            let test_circuit = TreeTestCircuit::new(domain_param_len);

            // Build message and compute digest
            let message = build_tree_hash(
                &domain_param_bytes,
                &left_bytes,
                &right_bytes,
                level,
                index,
            );

            // Verify message structure
            prop_assert_eq!(message.len(), domain_param_len + TREE_MESSAGE_OVERHEAD);
            prop_assert_eq!(message[domain_param_len], TREE_TWEAK);

            let expected_digest: [u8; 32] = sha3::Keccak256::digest(&message).into();

            // Verify circuit
            test_circuit
                .populate_and_verify(
                    &domain_param_bytes,
                    &left_bytes,
                    &right_bytes,
                    level,
                    index,
                    &message,
                    expected_digest,
                )
                .unwrap();
        }
    }
}
