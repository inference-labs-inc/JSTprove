// Copyright 2025 Irreducible Inc.
//! XMSS multi-signature aggregation for multiple validators.
//!
//! This module implements aggregation of XMSS signatures where each validator
//! has their own independent XMSS tree and signs at their designated epoch.
//! The aggregation creates a single proof that all signatures are valid.

use binius_frontend::{CircuitBuilder, Wire};

use super::{
    winternitz_ots::WinternitzSpec,
    xmss::{XmssHashers, XmssSignature, circuit_xmss},
};

/// The collection of XMSS hashers for multi-signature verification.
/// Contains one `XmssHashers` struct per validator.
pub struct XmssMultisigHashers {
    /// Vector of XmssHashers, one for each validator
    pub validator_hashers: Vec<XmssHashers>,
}

/// Verifies multiple XMSS signatures on the same message from different validators at a common
/// epoch.
///
/// This function implements multi-signature aggregation where:
/// - Each validator has their own independent XMSS tree (different roots)
/// - All validators sign the same message
/// - All validators sign at the same epoch (leaf index)
/// - The proof aggregates all individual signature verifications
///
/// # Public Inputs (inout wires)
/// - `domain_param`: Cryptographic domain parameter shared by all validators as 64-bit LE-packed
///   wires. The actual byte length is specified by `spec.domain_param_len`
/// - `message`: The common message being signed by all validators
/// - `epoch`: The common epoch (leaf index) at which all validators sign
/// - `validator_roots`: Each validator's XMSS tree root
///
/// # Private Inputs (witness wires)
/// - `validator_signatures`: Each validator's signature data (witness)
///
/// # Returns
///
/// An `XmssMultisigHashers` struct containing all hashers that need witness population
pub fn circuit_xmss_multisig(
    builder: &CircuitBuilder,
    spec: &WinternitzSpec,
    domain_param: &[Wire],
    message: &[Wire],
    epoch: Wire,
    validator_roots: &[[Wire; 4]],
    validator_signatures: &[XmssSignature],
) -> XmssMultisigHashers {
    assert_eq!(
        validator_roots.len(),
        validator_signatures.len(),
        "Number of validator roots must match number of signatures"
    );

    let mut validator_hashers = Vec::new();
    for (root, sig) in validator_roots.iter().zip(validator_signatures.iter()) {
        builder.assert_eq("epoch_equality", sig.epoch, epoch);
        let hashers = circuit_xmss(builder, spec, domain_param, message, sig, root);
        validator_hashers.push(hashers);
    }

    XmssMultisigHashers { validator_hashers }
}

/// Convenience structure for building multi-signature circuits.
///
/// This helps organize the wire allocation for multiple validators.
pub struct MultiSigBuilder<'a> {
    builder: &'a CircuitBuilder,
    spec: &'a WinternitzSpec,
}

impl<'a> MultiSigBuilder<'a> {
    pub fn new(builder: &'a CircuitBuilder, spec: &'a WinternitzSpec) -> Self {
        Self { builder, spec }
    }

    /// Creates public input wires for parameters, message, and epoch.
    pub fn create_public_inputs(&self) -> (Vec<Wire>, Vec<Wire>, Wire) {
        let param_wire_count = self.spec.domain_param_len.div_ceil(8);
        let param: Vec<Wire> = (0..param_wire_count)
            .map(|_| self.builder.add_inout())
            .collect();
        let message: Vec<Wire> = (0..4).map(|_| self.builder.add_inout()).collect();
        let epoch = self.builder.add_inout();
        (param, message, epoch)
    }

    /// Creates public input wires for validator roots.
    pub fn create_validator_roots(&self, num_validators: usize) -> Vec<[Wire; 4]> {
        (0..num_validators)
            .map(|_| std::array::from_fn(|_| self.builder.add_inout()))
            .collect()
    }

    /// Creates private witness wires for a single validator's signature using the shared epoch.
    pub fn create_validator_signature(&self, tree_height: usize, epoch: Wire) -> XmssSignature {
        XmssSignature {
            nonce: (0..3).map(|_| self.builder.add_witness()).collect(),
            epoch, // Use the shared epoch wire
            signature_hashes: (0..self.spec.dimension())
                .map(|_| std::array::from_fn(|_| self.builder.add_witness()))
                .collect(),
            public_key_hashes: (0..self.spec.dimension())
                .map(|_| std::array::from_fn(|_| self.builder.add_witness()))
                .collect(),
            auth_path: (0..tree_height)
                .map(|_| std::array::from_fn(|_| self.builder.add_witness()))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::util::pack_bytes_into_wires_le;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    use super::*;
    use crate::hash_based_sig::witness_utils::{
        ValidatorSignatureData, XmssHasherData, populate_xmss_hashers,
    };

    fn test_spec_small() -> WinternitzSpec {
        WinternitzSpec {
            message_hash_len: 4,
            coordinate_resolution_bits: 2,
            target_sum: 24,
            domain_param_len: 32,
        }
    }

    enum MultisigTestCase {
        Valid {
            num_validators: usize,
            tree_height: usize,
            epoch: u32,
        },
        Invalid {
            num_validators: usize,
            tree_height: usize,
            epoch: u32,
            corrupt_fn: fn(&mut MultisigTestData),
        },
    }

    impl MultisigTestCase {
        fn run(&self, spec: WinternitzSpec) {
            let mut rng = StdRng::seed_from_u64(42);

            match self {
                MultisigTestCase::Valid {
                    num_validators,
                    tree_height,
                    epoch,
                } => {
                    let test_data = MultisigTestData::generate(
                        *num_validators,
                        *tree_height,
                        *epoch,
                        &spec,
                        &mut rng,
                    );
                    test_data.run(&spec, *tree_height).unwrap();
                }
                MultisigTestCase::Invalid {
                    num_validators,
                    tree_height,
                    epoch,
                    corrupt_fn,
                } => {
                    let mut test_data = MultisigTestData::generate(
                        *num_validators,
                        *tree_height,
                        *epoch,
                        &spec,
                        &mut rng,
                    );
                    corrupt_fn(&mut test_data);
                    let result = test_data.run(&spec, *tree_height);
                    assert!(result.is_err(), "Test expected to fail but passed");
                }
            }
        }
    }

    // These functions corrupt specific aspects of multisig test data
    struct MultisigTestData {
        param_bytes: Vec<u8>,
        message_bytes: [u8; 32],
        epoch: u32, // Single shared epoch for all validators
        validators: Vec<ValidatorSignatureData>,
    }

    impl MultisigTestData {
        /// Generate test data for multi-signature verification
        fn generate(
            num_validators: usize,
            tree_height: usize,
            epoch: u32,
            spec: &WinternitzSpec,
            rng: &mut StdRng,
        ) -> Self {
            let mut param_bytes = vec![0u8; spec.domain_param_len];
            rng.fill_bytes(&mut param_bytes);

            let mut message_bytes = [0u8; 32];
            rng.fill_bytes(&mut message_bytes);

            let mut validators = Vec::new();
            for _ in 0..num_validators {
                validators.push(ValidatorSignatureData::generate(
                    rng,
                    &param_bytes,
                    &message_bytes,
                    epoch, // All validators sign at the same epoch
                    spec,
                    tree_height,
                ));
            }

            MultisigTestData {
                param_bytes,
                message_bytes,
                epoch,
                validators,
            }
        }

        /// Run the multi-signature verification test
        fn run(&self, spec: &WinternitzSpec, tree_height: usize) -> Result<(), Box<dyn Error>> {
            let builder = CircuitBuilder::new();
            let multisig_builder = MultiSigBuilder::new(&builder, spec);

            let (param, message, epoch_wire) = multisig_builder.create_public_inputs();
            let num_validators = self.validators.len();
            let validator_roots = multisig_builder.create_validator_roots(num_validators);

            let mut validator_signatures = Vec::new();
            for _ in 0..num_validators {
                validator_signatures
                    .push(multisig_builder.create_validator_signature(tree_height, epoch_wire));
            }

            let hashers = circuit_xmss_multisig(
                &builder,
                spec,
                &param,
                &message,
                epoch_wire,
                &validator_roots,
                &validator_signatures,
            );

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();

            // Pack param_bytes (pad to match wire count)
            let mut padded_param = vec![0u8; param.len() * 8];
            padded_param[..self.param_bytes.len()].copy_from_slice(&self.param_bytes);
            pack_bytes_into_wires_le(&mut w, &param, &padded_param);
            pack_bytes_into_wires_le(&mut w, &message, &self.message_bytes);
            w[epoch_wire] = Word::from_u64(self.epoch as u64);

            for (i, validator) in self.validators.iter().enumerate() {
                pack_bytes_into_wires_le(&mut w, &validator_roots[i], &validator.root);

                let mut nonce_padded = [0u8; 24];
                nonce_padded[..23].copy_from_slice(&validator.nonce);
                pack_bytes_into_wires_le(&mut w, &validator_signatures[i].nonce, &nonce_padded);

                for (j, sig_hash) in validator.signature_hashes.iter().enumerate() {
                    pack_bytes_into_wires_le(
                        &mut w,
                        &validator_signatures[i].signature_hashes[j],
                        sig_hash,
                    );
                }

                for (j, pk_hash) in validator.public_key_hashes.iter().enumerate() {
                    pack_bytes_into_wires_le(
                        &mut w,
                        &validator_signatures[i].public_key_hashes[j],
                        pk_hash,
                    );
                }

                for (j, auth_node) in validator.auth_path.iter().enumerate() {
                    pack_bytes_into_wires_le(
                        &mut w,
                        &validator_signatures[i].auth_path[j],
                        auth_node,
                    );
                }
            }

            for (val_idx, validator) in self.validators.iter().enumerate() {
                let validator_hasher = &hashers.validator_hashers[val_idx];

                let hasher_data = XmssHasherData {
                    param_bytes: self.param_bytes.to_vec(),
                    message_bytes: self.message_bytes,
                    nonce_bytes: validator.nonce.to_vec(),
                    epoch: self.epoch as u64, // Use shared epoch
                    coords: validator.coords.clone(),
                    sig_hashes: validator.signature_hashes.clone(),
                    pk_hashes: validator.public_key_hashes.clone(),
                    auth_path: validator.auth_path.clone(),
                };

                populate_xmss_hashers(&mut w, validator_hasher, spec, &hasher_data);
            }

            circuit.populate_wire_witness(&mut w)?;

            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec())?;

            Ok(())
        }
    }

    // ==================== Parameterized Tests ====================

    /// Valid test cases with different configurations
    #[rstest]
    #[case::three_validators_epoch_1(3, 3, 1, test_spec_small())]
    #[case::single_validator_epoch_2(1, 3, 2, test_spec_small())]
    #[case::five_validators_epoch_0(5, 3, 0, test_spec_small())]
    #[case::two_validators_spec1_epoch_0(2, 2, 0, WinternitzSpec::spec_1())]
    #[case::four_validators_spec2_epoch_1(4, 3, 1, WinternitzSpec::spec_2())]
    #[case::two_validators_small_tree_epoch_1(2, 2, 1, test_spec_small())]
    #[case::three_validators_large_tree_epoch_2(3, 4, 2, test_spec_small())]
    #[case::many_validators_same_epoch(6, 3, 2, test_spec_small())]
    fn test_xmss_multisig_valid(
        #[case] num_validators: usize,
        #[case] tree_height: usize,
        #[case] epoch: u32,
        #[case] spec: WinternitzSpec,
    ) {
        MultisigTestCase::Valid {
            num_validators,
            tree_height,
            epoch,
        }
        .run(spec);
    }

    fn corrupt_one_validator_signature(test_data: &mut MultisigTestData) {
        // Corrupt the second validator's first signature hash
        if test_data.validators.len() > 1 {
            test_data.validators[1].signature_hashes[0][0] ^= 0xFF;
        }
    }

    fn corrupt_shared_epoch(test_data: &mut MultisigTestData) {
        // Change the shared epoch to an incorrect value
        test_data.epoch = (test_data.epoch + 1) % 8;
    }

    fn corrupt_one_validator_message(test_data: &mut MultisigTestData) {
        // Make second validator sign a different message
        if test_data.validators.len() > 1 {
            let mut rng = StdRng::seed_from_u64(99999);
            let mut wrong_message = [0u8; 32];
            rng.fill_bytes(&mut wrong_message);

            // Regenerate second validator's signature with wrong message
            let spec = test_spec_small();
            test_data.validators[1] = ValidatorSignatureData::generate(
                &mut rng,
                &test_data.param_bytes,
                &wrong_message,
                test_data.epoch,
                &spec,
                3,
            );
        }
    }

    fn corrupt_one_validator_root(test_data: &mut MultisigTestData) {
        // Corrupt the first validator's root
        if !test_data.validators.is_empty() {
            test_data.validators[0].root[0] ^= 0xFF;
        }
    }

    fn corrupt_one_validator_auth_path(test_data: &mut MultisigTestData) {
        // Corrupt the last validator's first auth path node
        if let Some(validator) = test_data.validators.last_mut()
            && !validator.auth_path.is_empty()
        {
            validator.auth_path[0][0] ^= 0xFF;
        }
    }

    fn corrupt_validator_epochs(test_data: &mut MultisigTestData) {
        // Make validators sign at different epochs
        if test_data.validators.len() > 1 {
            let mut rng = StdRng::seed_from_u64(88888);
            let spec = test_spec_small();

            // Regenerate second validator with a different epoch
            let different_epoch = (test_data.epoch + 1) % 8;
            test_data.validators[1] = ValidatorSignatureData::generate(
                &mut rng,
                &test_data.param_bytes,
                &test_data.message_bytes,
                different_epoch,
                &spec,
                3,
            );
        }
    }

    /// Test that mismatched number of roots and signatures causes panic
    #[test]
    #[should_panic(expected = "Number of validator roots must match number of signatures")]
    fn test_multisig_mismatched_validators() {
        let builder = CircuitBuilder::new();
        let spec = test_spec_small();
        let multisig_builder = MultiSigBuilder::new(&builder, &spec);

        let (param, message, epoch) = multisig_builder.create_public_inputs();

        // Create 3 roots but only 2 signatures
        let validator_roots = multisig_builder.create_validator_roots(3);
        let validator_signatures = vec![
            multisig_builder.create_validator_signature(3, epoch),
            multisig_builder.create_validator_signature(3, epoch),
        ];

        // This should panic
        circuit_xmss_multisig(
            &builder,
            &spec,
            &param,
            &message,
            epoch,
            &validator_roots,
            &validator_signatures,
        );
    }

    /// Invalid test cases for multisig with various corruption scenarios
    #[rstest]
    #[case::corrupt_one_signature(corrupt_one_validator_signature)]
    #[case::corrupt_epoch(corrupt_shared_epoch)]
    #[case::corrupt_different_message(corrupt_one_validator_message)]
    #[case::corrupt_root(corrupt_one_validator_root)]
    #[case::corrupt_auth_path(corrupt_one_validator_auth_path)]
    #[case::corrupt_validator_epochs(corrupt_validator_epochs)]
    fn test_xmss_multisig_invalid(#[case] corrupt_fn: fn(&mut MultisigTestData)) {
        MultisigTestCase::Invalid {
            num_validators: 3,
            tree_height: 3,
            epoch: 2, // All validators sign at epoch 2
            corrupt_fn,
        }
        .run(test_spec_small());
    }
}
