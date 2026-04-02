// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::{
    hashing::circuit_public_key_hash,
    merkle_tree::circuit_merkle_path,
    winternitz_ots::{WinternitzOtsHashers, WinternitzSpec, circuit_winternitz_ots},
};
use crate::keccak::Keccak256;

/// An XMSS signature.
///
/// This structure contains all the witness data for an XMSS signature to be
/// verified.
#[derive(Clone)]
pub struct XmssSignature {
    /// Nonce (23 bytes in 3 wires)
    pub nonce: Vec<Wire>,
    /// The epoch is the index of key-pair used in the signature
    pub epoch: Wire,
    /// Winternitz signature hash values
    pub signature_hashes: Vec<[Wire; 4]>,
    /// Winternitz public key hashes
    pub public_key_hashes: Vec<[Wire; 4]>,
    /// Merkle authentication path
    pub auth_path: Vec<[Wire; 4]>,
}

/// The collection of Keccak hashers used in XMSS verification.
pub struct XmssHashers {
    /// Winternitz OTS hashers containing the message hasher and chain verification hashers.
    /// See `WinternitzOtsHashers` documentation for details on populating these.
    pub winternitz_ots: WinternitzOtsHashers,

    /// Keccak hasher for computing the OTS public key hash from individual Winternitz public keys.
    /// Computes: `hash(param || TWEAK_PUBLIC_KEY || pk_hash[0] || pk_hash[1] || ... ||
    /// pk_hash[D-1])` Must be populated with:
    /// - Message: The concatenated public key data (use `hashing::build_public_key_hash`)
    /// - Digest: The resulting public key hash (which becomes a leaf in the Merkle tree)
    pub public_key_hasher: Keccak256,

    /// Vector of Keccak hashers for verifying the Merkle tree authentication path.
    /// Contains one hasher per level of the tree that needs to be computed.
    /// Each hasher computes: `hash(param || TWEAK_TREE || level || index || left_child ||
    /// right_child)` Must be populated with:
    /// - Message: The tree node hash message (use `hashing::build_tree_hash`)
    /// - Digest: The parent node hash at that level
    ///
    /// The hashers are ordered from leaf level upward to the root.
    pub merkle_path_hashers: Vec<Keccak256>,
}

/// Verifies an XMSS (eXtended Merkle Signature Scheme) signature.
///
/// This circuit combines:
/// 1. Winternitz OTS verification for the one-time signature
/// 2. Computation of public key hash from Winternitz public key
/// 3. Merkle tree path verification to prove the public key is in the tree
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `spec` - Winternitz specification parameters (including domain_param_len)
/// * `domain_param` - Cryptographic domain parameter as 64-bit LE-packed wires. The actual byte
///   length is specified by `spec.domain_param_len`, and the wires must have sufficient capacity
///   (i.e., `domain_param.len() * 8 >= spec.domain_param_len`)
/// * `message` - Message to verify (32 bytes as 4x64-bit LE wires)
/// * `signature` - The XMSS signature containing all witness data
/// * `root_hash` - Expected Merkle tree root hash (32 bytes as 4x64-bit LE wires)
///
/// # Returns
///
/// An `XmssHashers` struct containing all hashers that need witness population
pub fn circuit_xmss(
    builder: &CircuitBuilder,
    spec: &WinternitzSpec,
    domain_param: &[Wire],
    message: &[Wire],
    signature: &XmssSignature,
    root_hash: &[Wire; 4],
) -> XmssHashers {
    // Step 1: Verify the Winternitz OTS signature
    let winternitz_ots = circuit_winternitz_ots(
        builder,
        domain_param,
        message,
        &signature.nonce,
        &signature.signature_hashes,
        &signature.public_key_hashes,
        spec,
    );

    // Step 2: Compute the public key hash from the Winternitz public key
    let pk_hash_output: [Wire; 4] = std::array::from_fn(|_| builder.add_witness());
    let public_key_hasher = circuit_public_key_hash(
        builder,
        domain_param.to_vec(),
        spec.domain_param_len,
        &signature.public_key_hashes,
        pk_hash_output,
    );

    // Step 3: Verify the Merkle tree path
    let merkle_path_hashers = circuit_merkle_path(
        builder,
        domain_param,
        spec.domain_param_len,
        &pk_hash_output,
        signature.epoch,
        &signature.auth_path,
        root_hash,
    );

    XmssHashers {
        winternitz_ots,
        public_key_hasher,
        merkle_path_hashers,
    }
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::util::pack_bytes_into_wires_le;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    use super::*;
    use crate::hash_based_sig::{
        hashing::{hash_chain_keccak, hash_public_key_keccak},
        winternitz_ots::grind_nonce,
        witness_utils::{
            XmssHasherData, build_merkle_tree, extract_auth_path, populate_xmss_hashers,
        },
    };

    /// Helper struct containing all test data for XMSS verification
    struct XmssTestData {
        param_bytes: Vec<u8>,
        message_bytes: [u8; 32],
        nonce_bytes: Vec<u8>,
        epoch: u64,
        coords: Vec<u8>,
        sig_hashes: Vec<[u8; 32]>,
        pk_hashes: Vec<[u8; 32]>,
        auth_path: Vec<[u8; 32]>,
        root_hash: [u8; 32],
        tree_depth: usize,
    }

    impl XmssTestData {
        /// Generate test data for XMSS verification
        fn generate(
            spec: &WinternitzSpec,
            tree_size: usize,
            signing_epoch: u64,
            rng: &mut StdRng,
        ) -> Self {
            // Generate random parameters based on spec
            let mut param_bytes = vec![0u8; spec.domain_param_len];
            rng.fill_bytes(&mut param_bytes);

            let mut message_bytes = [0u8; 32];
            rng.fill_bytes(&mut message_bytes);

            // Find valid nonce
            let grind_result = grind_nonce(spec, rng, &param_bytes, &message_bytes)
                .expect("Failed to find valid nonce");

            // Generate Winternitz signature and public key
            let mut sig_hashes = Vec::new();
            let mut pk_hashes = Vec::new();

            for (chain_idx, &coord) in grind_result.coords.iter().enumerate() {
                let mut sig_hash = [0u8; 32];
                rng.fill_bytes(&mut sig_hash);
                sig_hashes.push(sig_hash);

                let pk_hash = hash_chain_keccak(
                    &param_bytes,
                    chain_idx,
                    &sig_hash,
                    coord as usize,
                    spec.chain_len() - 1 - coord as usize,
                );
                pk_hashes.push(pk_hash);
            }

            // Build Merkle tree
            let mut leaves = Vec::new();
            for i in 0..tree_size {
                if i as u64 == signing_epoch {
                    leaves.push(hash_public_key_keccak(&param_bytes, &pk_hashes));
                } else {
                    // Fill other epochs with random values - these represent other public keys
                    // in the tree that we're not using for this signature verification
                    let mut leaf = [0u8; 32];
                    rng.fill_bytes(&mut leaf);
                    leaves.push(leaf);
                }
            }

            let (tree_levels, root_hash) = build_merkle_tree(&param_bytes, &leaves);
            let auth_path = extract_auth_path(&tree_levels, signing_epoch as usize);

            XmssTestData {
                param_bytes,
                message_bytes,
                nonce_bytes: grind_result.nonce,
                epoch: signing_epoch,
                coords: grind_result.coords,
                sig_hashes,
                pk_hashes,
                auth_path,
                root_hash,
                tree_depth: tree_levels.len() - 1,
            }
        }

        /// Run verification test with this test data
        fn run(&self, spec: &WinternitzSpec) -> Result<(), String> {
            let builder = CircuitBuilder::new();

            // Create input wires based on spec
            let param_wire_count = spec.domain_param_len.div_ceil(8);
            let param: Vec<Wire> = (0..param_wire_count).map(|_| builder.add_inout()).collect();
            let message: Vec<Wire> = (0..4).map(|_| builder.add_inout()).collect();
            let nonce: Vec<Wire> = (0..3).map(|_| builder.add_inout()).collect();
            let epoch = builder.add_inout();
            let root_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

            let signature_hashes: Vec<[Wire; 4]> = (0..spec.dimension())
                .map(|_| std::array::from_fn(|_| builder.add_inout()))
                .collect();

            let public_key_hashes: Vec<[Wire; 4]> = (0..spec.dimension())
                .map(|_| std::array::from_fn(|_| builder.add_inout()))
                .collect();

            let auth_path: Vec<[Wire; 4]> = (0..self.tree_depth)
                .map(|_| std::array::from_fn(|_| builder.add_inout()))
                .collect();

            // Create the verification circuit
            let signature = XmssSignature {
                nonce: nonce.clone(),
                epoch,
                signature_hashes: signature_hashes.clone(),
                public_key_hashes: public_key_hashes.clone(),
                auth_path: auth_path.clone(),
            };

            let hashers = circuit_xmss(&builder, spec, &param, &message, &signature, &root_hash);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();

            // Pack inputs into wires (pad param_bytes to match wire count)
            let mut padded_param = vec![0u8; param.len() * 8];
            padded_param[..self.param_bytes.len()].copy_from_slice(&self.param_bytes);
            pack_bytes_into_wires_le(&mut w, &param, &padded_param);
            pack_bytes_into_wires_le(&mut w, &message, &self.message_bytes);

            let mut nonce_padded = vec![0u8; 24];
            nonce_padded[..self.nonce_bytes.len()].copy_from_slice(&self.nonce_bytes);
            pack_bytes_into_wires_le(&mut w, &nonce, &nonce_padded);

            w[epoch] = Word::from_u64(self.epoch);
            pack_bytes_into_wires_le(&mut w, &root_hash, &self.root_hash);

            for (i, sig_hash) in self.sig_hashes.iter().enumerate() {
                pack_bytes_into_wires_le(&mut w, &signature_hashes[i], sig_hash);
            }

            for (i, pk_hash) in self.pk_hashes.iter().enumerate() {
                pack_bytes_into_wires_le(&mut w, &public_key_hashes[i], pk_hash);
            }

            for (i, auth_node) in self.auth_path.iter().enumerate() {
                pack_bytes_into_wires_le(&mut w, &auth_path[i], auth_node);
            }

            let hasher_data = XmssHasherData {
                param_bytes: self.param_bytes.clone(),
                message_bytes: self.message_bytes,
                nonce_bytes: self.nonce_bytes.clone(),
                epoch: self.epoch,
                coords: self.coords.clone(),
                sig_hashes: self.sig_hashes.clone(),
                pk_hashes: self.pk_hashes.clone(),
                auth_path: self.auth_path.clone(),
            };
            populate_xmss_hashers(&mut w, &hashers, spec, &hasher_data);

            circuit
                .populate_wire_witness(&mut w)
                .map_err(|e| format!("Wire population failed: {:?}", e))?;

            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec())
                .map_err(|e| format!("Constraint verification failed: {:?}", e))?;

            Ok(())
        }
    }

    /// Test case configuration for parameterized testing
    enum TestCase {
        Valid {
            tree_size: usize,
            signing_epoch: u64,
        },
        Invalid {
            tree_size: usize,
            signing_epoch: u64,
            corrupt_fn: fn(&mut XmssTestData),
        },
    }

    impl TestCase {
        fn run(&self, spec: WinternitzSpec) {
            let mut rng = StdRng::seed_from_u64(42);

            match self {
                TestCase::Valid {
                    tree_size,
                    signing_epoch,
                } => {
                    // Generate test data
                    let test_data =
                        XmssTestData::generate(&spec, *tree_size, *signing_epoch, &mut rng);

                    let result = test_data.run(&spec);
                    result.unwrap_or_else(|e| {
                        panic!("Test expected to pass but failed: {}", e);
                    });
                }
                TestCase::Invalid {
                    tree_size,
                    signing_epoch,
                    corrupt_fn,
                } => {
                    // Generate test data
                    let mut test_data =
                        XmssTestData::generate(&spec, *tree_size, *signing_epoch, &mut rng);

                    // Apply corruption
                    corrupt_fn(&mut test_data);

                    let result = test_data.run(&spec);
                    assert!(result.is_err(), "Test expected to fail but passed");
                }
            }
        }
    }

    fn corrupt_signature(test_data: &mut XmssTestData) {
        // Corrupt the first signature hash
        if !test_data.sig_hashes.is_empty() {
            test_data.sig_hashes[0][0] ^= 0xFF;
        }
    }

    fn corrupt_public_key(test_data: &mut XmssTestData) {
        // Corrupt the first public key hash
        if !test_data.pk_hashes.is_empty() {
            test_data.pk_hashes[0][0] ^= 0xFF;
        }
    }

    fn corrupt_auth_path(test_data: &mut XmssTestData) {
        // Corrupt a node in the authentication path
        if !test_data.auth_path.is_empty() {
            test_data.auth_path[0][0] ^= 0xFF;
        }
    }

    fn corrupt_root_hash(test_data: &mut XmssTestData) {
        // Corrupt the root hash
        test_data.root_hash[0] ^= 0xFF;
    }

    fn corrupt_message(test_data: &mut XmssTestData) {
        // Change the message after signing
        test_data.message_bytes[0] ^= 0xFF;
    }

    fn corrupt_epoch(test_data: &mut XmssTestData) {
        // Use wrong epoch
        test_data.epoch = (test_data.epoch + 1) % 4;
    }

    // ==================== Test Specs ====================

    fn test_spec_small() -> WinternitzSpec {
        WinternitzSpec {
            message_hash_len: 4,
            coordinate_resolution_bits: 2,
            target_sum: 24,
            domain_param_len: 32,
        }
    }

    /// Valid test cases with different configurations
    #[rstest]
    #[case::small_tree_4(test_spec_small(), 4, 1)]
    #[case::small_tree_8(test_spec_small(), 8, 3)]
    #[case::medium_tree_16(test_spec_small(), 16, 7)]
    #[case::spec1(WinternitzSpec::spec_1(), 4, 0)]
    #[case::spec2(WinternitzSpec::spec_2(), 4, 2)]
    fn test_xmss_valid(
        #[case] spec: WinternitzSpec,
        #[case] tree_size: usize,
        #[case] signing_epoch: u64,
    ) {
        TestCase::Valid {
            tree_size,
            signing_epoch,
        }
        .run(spec);
    }

    /// Invalid test cases with various corruption scenarios
    #[rstest]
    #[case::corrupt_signature(corrupt_signature)]
    #[case::corrupt_public_key(corrupt_public_key)]
    #[case::corrupt_auth_path(corrupt_auth_path)]
    #[case::corrupt_root(corrupt_root_hash)]
    #[case::corrupt_message(corrupt_message)]
    #[case::corrupt_epoch(corrupt_epoch)]
    fn test_xmss_invalid(#[case] corrupt_fn: fn(&mut XmssTestData)) {
        TestCase::Invalid {
            tree_size: 4,
            signing_epoch: 1,
            corrupt_fn,
        }
        .run(test_spec_small());
    }
}
