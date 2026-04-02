// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use super::base::circuit_tweaked_keccak;
use crate::{fixed_byte_vec::ByteVec, keccak::Keccak256};

pub const MESSAGE_TWEAK: u8 = 0x02;

/// A circuit that verifies a message-tweaked Keccak-256 computation.
///
/// This circuit verifies Keccak-256 of a message that's been tweaked with
/// message-specific parameters: `Keccak256(domain_param || 0x02 || nonce || message)`
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param_wires` - The cryptographic domain parameter wires (typically public key
///   material), where each wire holds 8 bytes as a 64-bit LE-packed value
/// * `domain_param_len` - The actual domain parameter length in bytes
/// * `nonce_wires` - Random nonce wires to ensure uniqueness, where each wire holds 8 bytes as a
///   64-bit LE-packed value
/// * `nonce_len` - The actual nonce length in bytes
/// * `message_wires` - The message content wires, where each wire holds 8 bytes as a 64-bit
///   LE-packed value
/// * `message_len` - The actual message length in bytes
/// * `digest` - Output: The computed Keccak-256 digest (32 bytes as 4x64-bit LE-packed wires)
///
/// # Returns
///
/// A `Keccak` circuit that needs to be populated with the tweaked message and digest
#[allow(clippy::too_many_arguments)]
pub fn circuit_message_hash(
    builder: &CircuitBuilder,
    domain_param_wires: Vec<Wire>,
    domain_param_len: usize,
    nonce_wires: Vec<Wire>,
    nonce_len: usize,
    message_wires: Vec<Wire>,
    message_len: usize,
    digest: [Wire; 4],
) -> Keccak256 {
    let total_message_len = domain_param_len + 1 + nonce_len + message_len; // +1 for tweak byte

    let mut additional_terms = Vec::new();

    let nonce_term = ByteVec {
        len_bytes: builder.add_constant_64(nonce_len as u64),
        data: nonce_wires.clone(),
    };
    additional_terms.push(nonce_term);

    let message_term = ByteVec {
        len_bytes: builder.add_constant_64(message_len as u64),
        data: message_wires.clone(),
    };
    additional_terms.push(message_term);

    circuit_tweaked_keccak(
        builder,
        domain_param_wires,
        domain_param_len,
        MESSAGE_TWEAK,
        additional_terms,
        total_message_len,
        digest,
    )
}

/// Build the tweaked message from components.
///
/// Constructs the complete message for Keccak-256 hashing by concatenating:
/// `domain_param || 0x02 || nonce || message`
///
/// This function is typically used when populating witness data for the
/// `circuit_message_hash` circuit.
///
/// # Arguments
///
/// * `domain_param_bytes` - The cryptographic domain parameter bytes
/// * `nonce_bytes` - The random nonce bytes
/// * `message_bytes` - The message content bytes
///
/// # Returns
///
/// A vector containing the complete tweaked message ready for hashing
pub fn build_message_hash(
    domain_param_bytes: &[u8],
    nonce_bytes: &[u8],
    message_bytes: &[u8],
) -> Vec<u8> {
    let mut message = Vec::new();
    message.extend_from_slice(domain_param_bytes);
    message.push(MESSAGE_TWEAK); // TWEAK_MESSAGE
    message.extend_from_slice(nonce_bytes);
    message.extend_from_slice(message_bytes);
    message
}

/// Compute the tweaked hash of a message from components.
///
/// Constructs the complete message for Keccak-256 hashing by concatenating:
/// `param || 0x02 || nonce || message`
///
/// # Arguments
///
/// * `param_bytes` - The cryptographic parameter bytes
/// * `nonce_bytes` - The random nonce bytes
/// * `message_bytes` - The message content bytes
///
/// # Returns
///
/// The tweaked message hash
pub fn hash_message(param: &[u8], nonce: &[u8], message: &[u8]) -> [u8; 32] {
    use sha3::Digest;
    let tweaked_message = build_message_hash(param, nonce, message);
    sha3::Keccak256::digest(tweaked_message).into()
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::{Circuit, CircuitBuilder, util::pack_bytes_into_wires_le};
    use proptest::prelude::*;
    use sha3::Digest;

    use super::*;

    /// Helper struct for MessageHash testing
    struct MessageTestCircuit {
        circuit: Circuit,
        keccak: Keccak256,
        domain_param_wires: Vec<Wire>,
        domain_param_len: usize,
        nonce_wires: Vec<Wire>,
        nonce_len: usize,
        message_wires: Vec<Wire>,
        message_len: usize,
    }

    impl MessageTestCircuit {
        fn new(domain_param_len: usize, nonce_len: usize, message_len: usize) -> Self {
            let builder = CircuitBuilder::new();

            let num_domain_param_wires = domain_param_len.div_ceil(8);
            let domain_param_wires: Vec<Wire> = (0..num_domain_param_wires)
                .map(|_| builder.add_inout())
                .collect();

            let num_nonce_wires = nonce_len.div_ceil(8);
            let nonce_wires: Vec<Wire> = (0..num_nonce_wires)
                .map(|_| builder.add_witness())
                .collect();

            let num_message_wires = message_len.div_ceil(8);
            let message_wires: Vec<Wire> = (0..num_message_wires)
                .map(|_| builder.add_witness())
                .collect();

            let digest: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

            let keccak = circuit_message_hash(
                &builder,
                domain_param_wires.clone(),
                domain_param_len,
                nonce_wires.clone(),
                nonce_len,
                message_wires.clone(),
                message_len,
                digest,
            );

            let circuit = builder.build();

            Self {
                circuit,
                keccak,
                domain_param_wires,
                domain_param_len,
                nonce_wires,
                nonce_len,
                message_wires,
                message_len,
            }
        }

        /// Populate witness and verify constraints with given test data
        fn populate_and_verify(
            &self,
            domain_param_bytes: &[u8],
            nonce_bytes: &[u8],
            message_bytes: &[u8],
            full_message: &[u8],
            digest: [u8; 32],
        ) -> Result<(), Box<dyn std::error::Error>> {
            let mut w = self.circuit.new_witness_filler();

            assert_eq!(domain_param_bytes.len(), self.domain_param_len);
            pack_bytes_into_wires_le(&mut w, &self.domain_param_wires, domain_param_bytes);

            assert_eq!(nonce_bytes.len(), self.nonce_len);
            pack_bytes_into_wires_le(&mut w, &self.nonce_wires, nonce_bytes);

            assert_eq!(message_bytes.len(), self.message_len);
            pack_bytes_into_wires_le(&mut w, &self.message_wires, message_bytes);

            self.keccak.populate_message(&mut w, full_message);
            self.keccak.populate_digest(&mut w, digest);

            self.circuit.populate_wire_witness(&mut w)?;
            let cs = self.circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec())?;
            Ok(())
        }
    }

    #[test]
    fn test_message_hash_basic() {
        let test_circuit = MessageTestCircuit::new(32, 16, 64);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let nonce_bytes = b"random_nonce_16b";
        let message_bytes = b"This is a test message that is exactly 64 bytes long!!!!!!!!!!!."; // 64 bytes

        let full_message = build_message_hash(domain_param_bytes, nonce_bytes, message_bytes);

        let expected_digest = sha3::Keccak256::digest(&full_message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                nonce_bytes,
                message_bytes,
                &full_message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_message_hash_with_18_byte_domain_param() {
        // Test with 18-byte domain param as commonly used in XMSS
        let test_circuit = MessageTestCircuit::new(18, 8, 32);

        let domain_param_bytes: &[u8; 18] = b"test_param_18bytes";
        let nonce_bytes = b"nonce_8b";
        let message_bytes = b"message_that_is_32_bytes_long!!!";

        let full_message = build_message_hash(domain_param_bytes, nonce_bytes, message_bytes);

        let expected_digest = sha3::Keccak256::digest(&full_message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                nonce_bytes,
                message_bytes,
                &full_message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_message_hash_variable_lengths() {
        // Test with various non-aligned lengths
        let test_circuit = MessageTestCircuit::new(13, 7, 29);

        let domain_param_bytes = b"param_13bytes";
        let nonce_bytes = b"nonce7b";
        let message_bytes = b"msg_that_is_29_bytes_long!!!!";

        let full_message = build_message_hash(domain_param_bytes, nonce_bytes, message_bytes);

        let expected_digest = sha3::Keccak256::digest(&full_message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                nonce_bytes,
                message_bytes,
                &full_message,
                expected_digest.into(),
            )
            .unwrap();
    }

    #[test]
    fn test_message_hash_wrong_digest() {
        let test_circuit = MessageTestCircuit::new(32, 16, 64);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let nonce_bytes = b"random_nonce_16b";
        let message_bytes = b"This is a test message that is exactly 64 bytes long!!!!!!!!!!!."; // 64 bytes

        let full_message = build_message_hash(domain_param_bytes, nonce_bytes, message_bytes);

        // Populate with WRONG digest - this should cause verification to fail
        let wrong_digest = [0u8; 32];

        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            nonce_bytes,
            message_bytes,
            &full_message,
            wrong_digest,
        );

        assert!(result.is_err(), "Expected error for wrong digest");
    }

    #[test]
    fn test_message_hash_wrong_domain_param() {
        let test_circuit = MessageTestCircuit::new(32, 16, 64);

        let correct_domain_param_bytes = b"correct_parameter_32_bytes!!!!!!";
        let wrong_domain_param_bytes = b"wrong___parameter_32_bytes!!!!!!";
        let nonce_bytes = b"random_nonce_16b";
        let message_bytes = b"This is a test message that is exactly 64 bytes long!!!!!!!!!!!."; // 64 bytes

        // Build message with correct domain param
        let full_message =
            build_message_hash(correct_domain_param_bytes, nonce_bytes, message_bytes);

        let expected_digest = sha3::Keccak256::digest(&full_message);

        // Populate with WRONG domain param but correct digest
        let result = test_circuit.populate_and_verify(
            wrong_domain_param_bytes,
            nonce_bytes,
            message_bytes,
            &full_message,
            expected_digest.into(),
        );

        assert!(
            result.is_err(),
            "Expected error for mismatched domain param"
        );
    }

    #[test]
    fn test_message_hash_wrong_nonce() {
        let test_circuit = MessageTestCircuit::new(32, 16, 64);

        let domain_param_bytes = b"test_parameter_32_bytes_long!!!!";
        let correct_nonce = b"correct_nonce16b";
        let wrong_nonce = b"wrong___nonce16b";
        let message_bytes = b"This is a test message that is exactly 64 bytes long!!!!!!!!!!!."; // 64 bytes

        // Build message with correct nonce
        let full_message = build_message_hash(domain_param_bytes, correct_nonce, message_bytes);

        let expected_digest = sha3::Keccak256::digest(&full_message);

        // Populate with WRONG nonce but correct digest
        let result = test_circuit.populate_and_verify(
            domain_param_bytes,
            wrong_nonce,
            message_bytes,
            &full_message,
            expected_digest.into(),
        );

        assert!(result.is_err(), "Expected error for mismatched nonce");
    }

    #[test]
    fn test_message_hash_ensures_tweak_byte() {
        // This test verifies that the MESSAGE_TWEAK byte (0x02) is correctly inserted
        let test_circuit = MessageTestCircuit::new(8, 8, 16);

        let domain_param_bytes = b"param_8b";
        let nonce_bytes = b"nonce_8b";
        let message_bytes = b"message_16_bytes";

        let full_message = build_message_hash(domain_param_bytes, nonce_bytes, message_bytes);

        // Verify the tweak byte is at the correct position
        assert_eq!(full_message[8], MESSAGE_TWEAK);
        assert_eq!(full_message.len(), 8 + 1 + 8 + 16); // domain_param + tweak + nonce + message

        let expected_digest = sha3::Keccak256::digest(&full_message);

        test_circuit
            .populate_and_verify(
                domain_param_bytes,
                nonce_bytes,
                message_bytes,
                &full_message,
                expected_digest.into(),
            )
            .unwrap();
    }

    proptest! {
        #[test]
        fn test_message_hash_property_based(
            domain_param_len in 1usize..=100,
            nonce_len in 1usize..=50,
            message_len in 1usize..=200,
        ) {
            use rand::SeedableRng;
            use rand::prelude::StdRng;

            let mut rng = StdRng::seed_from_u64(0);

            // Generate random data of specified lengths
            let mut domain_param_bytes = vec![0u8; domain_param_len];
            rng.fill_bytes(&mut domain_param_bytes);

            let mut nonce_bytes = vec![0u8; nonce_len];
            rng.fill_bytes(&mut nonce_bytes);

            let mut message_bytes = vec![0u8; message_len];
            rng.fill_bytes(&mut message_bytes);

            // Create circuit
            let test_circuit = MessageTestCircuit::new(domain_param_len, nonce_len, message_len);

            // Build full message and compute digest
            let full_message =
                build_message_hash(&domain_param_bytes, &nonce_bytes, &message_bytes);

            // Verify message structure
            prop_assert_eq!(full_message.len(), domain_param_len + 1 + nonce_len + message_len);
            prop_assert_eq!(full_message[domain_param_len], MESSAGE_TWEAK);

            let expected_digest: [u8; 32] = sha3::Keccak256::digest(&full_message).into();

            // Verify circuit
            test_circuit
                .populate_and_verify(
                    &domain_param_bytes,
                    &nonce_bytes,
                    &message_bytes,
                    &full_message,
                    expected_digest,
                )
                .unwrap();
        }
    }
}
