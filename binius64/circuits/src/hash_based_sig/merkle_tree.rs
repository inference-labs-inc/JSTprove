// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::hashing::circuit_tree_hash;
use crate::{keccak::Keccak256, multiplexer::multi_wire_multiplex};

/// Verifies a Merkle tree authentication path.
///
/// This circuit verifies that a given leaf hash is part of a Merkle tree
/// by reconstructing the path from leaf to root using the provided
/// authentication path (sibling hashes).
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param` - Cryptographic domain parameter (32 bytes as 4x64-bit LE wires)
/// * `domain_param_len` - Actual byte length of the parameter (must be less than or equal to
///   domain_param.len() * 8)
/// * `leaf_hash` - The leaf hash to verify (32 bytes as 4x64-bit LE wires)
/// * `leaf_index` - Index of the leaf in the tree (as a wire)
/// * `auth_path` - Authentication path: sibling hashes from leaf to root
/// * `root_hash` - Expected root hash (32 bytes as 4x64-bit LE wires)
///
/// # Returns
///
/// A vector of Keccak hashers that need witness population
pub fn circuit_merkle_path(
    builder: &CircuitBuilder,
    domain_param: &[Wire],
    domain_param_len: usize,
    leaf_hash: &[Wire; 4],
    leaf_index: Wire,
    auth_path: &[[Wire; 4]],
    root_hash: &[Wire; 4],
) -> Vec<Keccak256> {
    assert!(
        domain_param_len <= domain_param.len() * 8,
        "domain_param_len {} exceeds maximum capacity {} of domain_param wires",
        domain_param_len,
        domain_param.len() * 8
    );

    let tree_height = auth_path.len();
    let mut hashers = Vec::with_capacity(tree_height);
    let mut current_hash = *leaf_hash;
    let mut current_index = leaf_index;
    let one = builder.add_constant_64(1);

    // Process each level of the tree
    for level in 0..tree_height {
        let sibling_hash = auth_path[level];

        // Determine if current node is left or right child
        // If current_index is even (LSB = 0), current is left child
        // If current_index is odd (LSB = 1), current is right child
        let is_left = builder.bnot(builder.band(current_index, one));

        // Select left and right hashes based on position
        // If is_left: left = current, right = sibling
        // If !is_left: left = sibling, right = current
        let left_hash = multi_wire_multiplex(builder, &[&sibling_hash, &current_hash], is_left)
            .try_into()
            .expect("multi_wire_multiplex should return 4 wires");
        let right_hash = multi_wire_multiplex(builder, &[&current_hash, &sibling_hash], is_left)
            .try_into()
            .expect("multi_wire_multiplex should return 4 wires");

        // Compute parent index: parent_index = current_index / 2
        let parent_index = builder.shr(current_index, 1);

        // Create output hash wire for this level
        let parent_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());

        // Create tree node hasher
        let level_wire = builder.add_constant_64(level as u64);
        let hasher = circuit_tree_hash(
            builder,
            domain_param.to_vec(),
            domain_param_len,
            left_hash,
            right_hash,
            level_wire,
            parent_index,
            parent_hash,
        );

        hashers.push(hasher);

        // Move up the tree
        current_hash = parent_hash;
        current_index = parent_index;
    }

    // Assert that the final hash matches the expected root
    builder.assert_eq_v("merkle_root_check", current_hash, *root_hash);

    hashers
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::util::pack_bytes_into_wires_le;

    use super::*;
    use crate::hash_based_sig::hashing::{build_tree_hash, hash_tree_node_keccak};

    #[test]
    fn test_circuit_merkle_path_verification() {
        // Build a simple 4-leaf tree for testing
        // Tree structure:
        //        root
        //       /    \
        //      n2     n3
        //     / \    / \
        //    l0 l1  l2 l3

        let builder = CircuitBuilder::new();

        // Create input wires
        let domain_param: Vec<Wire> = (0..4).map(|_| builder.add_inout()).collect();
        let leaf_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
        let leaf_index = builder.add_inout();
        let root_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

        // Authentication path (2 levels for 4-leaf tree)
        let auth_path: Vec<[Wire; 4]> = (0..2)
            .map(|_| std::array::from_fn(|_| builder.add_inout()))
            .collect();

        // Create the verification circuit
        let hashers = circuit_merkle_path(
            &builder,
            &domain_param,
            domain_param.len() * 8,
            &leaf_hash,
            leaf_index,
            &auth_path,
            &root_hash,
        );

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        // Set up test data
        let param_bytes = b"test_merkle_tree_parameter!!!!!!";
        pack_bytes_into_wires_le(&mut w, &domain_param, param_bytes);

        // Create leaf hashes
        let leaf0 = b"leaf_0_hash_value_32_bytes!!!!!!";
        let leaf1 = b"leaf_1_hash_value_32_bytes!!!!!!";
        let leaf2 = b"leaf_2_hash_value_32_bytes!!!!!!";
        let leaf3 = b"leaf_3_hash_value_32_bytes!!!!!!";

        // Compute internal nodes
        let node2 = hash_tree_node_keccak(param_bytes, leaf0, leaf1, 0, 0);
        let node3 = hash_tree_node_keccak(param_bytes, leaf2, leaf3, 0, 1);
        let root = hash_tree_node_keccak(param_bytes, &node2, &node3, 1, 0);

        // Test verification for leaf 1 (index 1)
        // Path: [leaf0 (sibling at level 0), node3 (sibling at level 1)]
        pack_bytes_into_wires_le(&mut w, &leaf_hash, leaf1);
        w[leaf_index] = Word::from_u64(1);
        pack_bytes_into_wires_le(&mut w, &auth_path[0], leaf0);
        pack_bytes_into_wires_le(&mut w, &auth_path[1], &node3);
        pack_bytes_into_wires_le(&mut w, &root_hash, &root);

        // Populate hashers
        // Level 0: hash(leaf0, leaf1) with index 0
        let hasher0 = &hashers[0];
        let message0 = build_tree_hash(param_bytes, leaf0, leaf1, 0, 0);
        hasher0.populate_message(&mut w, &message0);
        hasher0.populate_digest(&mut w, node2);

        // Level 1: hash(node2, node3) with index 0
        let hasher1 = &hashers[1];
        let message1 = build_tree_hash(param_bytes, &node2, &node3, 1, 0);
        hasher1.populate_message(&mut w, &message1);
        hasher1.populate_digest(&mut w, root);

        // Populate witness and verify
        circuit.populate_wire_witness(&mut w).unwrap();

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_negative_circuit_merkle_invalid_auth_path() {
        // Circuit does not verify invalid auth path
        let builder = CircuitBuilder::new();

        let param: Vec<Wire> = (0..4).map(|_| builder.add_inout()).collect();
        let leaf_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
        let leaf_index = builder.add_inout();
        let root_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
        let auth_path: Vec<[Wire; 4]> = (0..2)
            .map(|_| std::array::from_fn(|_| builder.add_inout()))
            .collect();

        let hashers = circuit_merkle_path(
            &builder,
            &param,
            param.len() * 8,
            &leaf_hash,
            leaf_index,
            &auth_path,
            &root_hash,
        );

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        let param_bytes = b"test_merkle_tree_parameter!!!!!!";
        pack_bytes_into_wires_le(&mut w, &param, param_bytes);

        let leaf0 = b"leaf_0_hash_value_32_bytes!!!!!!";
        let leaf1 = b"leaf_1_hash_value_32_bytes!!!!!!";
        let leaf2 = b"leaf_2_hash_value_32_bytes!!!!!!";
        let leaf3 = b"leaf_3_hash_value_32_bytes!!!!!!";

        let node2 = hash_tree_node_keccak(param_bytes, leaf0, leaf1, 0, 0);
        let node3 = hash_tree_node_keccak(param_bytes, leaf2, leaf3, 0, 1);
        let root = hash_tree_node_keccak(param_bytes, &node2, &node3, 1, 0);

        // Test verification for leaf 2 (index 2)
        // Path: [leaf3 (sibling at level 0), node2 incorrect!]
        pack_bytes_into_wires_le(&mut w, &leaf_hash, leaf2);
        w[leaf_index] = Word::from_u64(2);
        pack_bytes_into_wires_le(&mut w, &auth_path[0], leaf3);
        pack_bytes_into_wires_le(&mut w, &auth_path[1], &node3);
        pack_bytes_into_wires_le(&mut w, &root_hash, &root);

        // Populate hashers
        // Level 0: hash(leaf2, leaf3) with index 1
        let hasher0 = &hashers[0];
        let message0 = build_tree_hash(param_bytes, leaf2, leaf3, 0, 1);
        hasher0.populate_message(&mut w, &message0);
        hasher0.populate_digest(&mut w, node3);

        // Level 1: hash(node2, node3) with index 0
        let hasher1 = &hashers[1];
        let message1 = build_tree_hash(param_bytes, &node2, &node3, 1, 0);
        hasher1.populate_message(&mut w, &message1);
        hasher1.populate_digest(&mut w, root);

        assert!(circuit.populate_wire_witness(&mut w).is_err());
    }

    #[test]
    fn test_negative_circuit_merkle_invalid_hash_population() {
        // The circuit does not verify an invalid hash population
        let builder = CircuitBuilder::new();

        let param: Vec<Wire> = (0..4).map(|_| builder.add_inout()).collect();
        let leaf_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
        let leaf_index = builder.add_inout();
        let root_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
        let auth_path: Vec<[Wire; 4]> = (0..2)
            .map(|_| std::array::from_fn(|_| builder.add_inout()))
            .collect();

        let hashers = circuit_merkle_path(
            &builder,
            &param,
            param.len() * 8,
            &leaf_hash,
            leaf_index,
            &auth_path,
            &root_hash,
        );

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        let param_bytes = b"test_merkle_tree_parameter!!!!!!";
        pack_bytes_into_wires_le(&mut w, &param, param_bytes);

        let leaf0 = b"leaf_0_hash_value_32_bytes!!!!!!";
        let leaf1 = b"leaf_1_hash_value_32_bytes!!!!!!";
        let leaf2 = b"leaf_2_hash_value_32_bytes!!!!!!";
        let leaf3 = b"leaf_3_hash_value_32_bytes!!!!!!";

        let node2 = hash_tree_node_keccak(param_bytes, leaf0, leaf1, 0, 0);
        let node3 = hash_tree_node_keccak(param_bytes, leaf2, leaf3, 0, 1);
        let root = hash_tree_node_keccak(param_bytes, &node2, &node3, 1, 0);

        // Test verification for leaf 2 (index 2)
        // Path: [leaf3 (sibling at level 0), node2 (sibling at level 1)]
        pack_bytes_into_wires_le(&mut w, &leaf_hash, leaf2);
        w[leaf_index] = Word::from_u64(2);
        pack_bytes_into_wires_le(&mut w, &auth_path[0], leaf3);
        pack_bytes_into_wires_le(&mut w, &auth_path[1], &node2);
        pack_bytes_into_wires_le(&mut w, &root_hash, &root);

        // Populate hashers
        // Level 0: hash(leaf2, leaf3) with index 1
        let hasher0 = &hashers[0];
        let message0 = build_tree_hash(param_bytes, leaf2, leaf3, 0, 1);
        hasher0.populate_message(&mut w, &message0);
        hasher0.populate_digest(&mut w, node3);

        // Level 1: hash(node2, leaf3) with index 0 - incorrect!
        let hasher1 = &hashers[1];
        let message1 = build_tree_hash(param_bytes, &node2, leaf3, 1, 0);
        hasher1.populate_message(&mut w, &message1);
        hasher1.populate_digest(&mut w, root);

        assert!(circuit.populate_wire_witness(&mut w).is_err());
    }
}
