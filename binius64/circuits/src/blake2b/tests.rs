// Copyright 2025 Irreducible Inc.
//! Comprehensive BLAKE2b tests for both implementations
//!
//! This module contains tests that validate both the reference implementation and
//! circuit implementation against the standard blake2 crate, ensuring they produce
//! identical results across all test cases.

#[cfg(test)]
mod blake2b_tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::CircuitBuilder;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::blake2b::{
        BLOCK_BYTES,
        circuit::Blake2bCircuit,
        reference,
        test_constants::{ground_truth_blake2b, hex_string},
    };

    /// All test vectors for comprehensive testing
    fn get_all_test_vectors() -> Vec<(Vec<u8>, String)> {
        let mut vectors = vec![
            // Basic test vectors
            (b"".to_vec(), "empty".to_string()),
            (b"\x00".to_vec(), "single_null_byte".to_string()),
            (b"abc".to_vec(), "abc".to_string()),
            (b"a".to_vec(), "single_a".to_string()),
            (b"message digest".to_vec(), "message_digest".to_string()),
            (
                b"abcdefghijklmnopqrstuvwxyz".to_vec(),
                "alphabet".to_string(),
            ),
            (
                b"The quick brown fox jumps over the lazy dog".to_vec(),
                "quick_brown_fox".to_string(),
            ),
        ];

        // Block boundary tests
        vectors.push((
            vec![0x42u8; BLOCK_BYTES - 1],
            "just_under_one_block".to_string(),
        ));
        vectors.push((vec![0x42u8; BLOCK_BYTES], "exactly_one_block".to_string()));
        vectors.push((
            vec![0x42u8; BLOCK_BYTES + 1],
            "just_over_one_block".to_string(),
        ));
        vectors.push((
            vec![0xAAu8; 2 * BLOCK_BYTES - 1],
            "just_under_two_blocks".to_string(),
        ));
        vectors.push((
            vec![0xAAu8; 2 * BLOCK_BYTES],
            "exactly_two_blocks".to_string(),
        ));
        vectors.push((
            vec![0xAAu8; 2 * BLOCK_BYTES + 1],
            "just_over_two_blocks".to_string(),
        ));

        // Pattern tests
        vectors.push((vec![0x00u8; 64], "all_zero_64".to_string()));
        vectors.push((vec![0xFFu8; 64], "all_one_64".to_string()));
        vectors.push(((0..64u8).collect(), "sequential_64".to_string()));

        // Various sizes
        for size in [1, 3, 32, 55, 64, 128, 129, 200, 256, 300, 1024] {
            vectors.push((vec![0x55u8; size], format!("size_{}_0x55", size)));
        }

        // Some specific patterns for different sizes
        vectors.push((
            (0..128u8).cycle().take(200).collect(),
            "cyclic_200".to_string(),
        ));
        vectors.push((
            [0xDEu8, 0xADu8, 0xBEu8, 0xEFu8].repeat(75),
            "deadbeef_300".to_string(),
        ));

        vectors
    }

    /// Validate both implementations against ground truth for a single message
    fn validate_implementations(message: &[u8], test_name: &str) {
        let ground_truth = ground_truth_blake2b(message);

        // Test reference implementation
        let reference_result = reference::blake2b(message, 64);
        assert_eq!(
            reference_result,
            ground_truth,
            "Reference mismatch for {}: ref={}, ground_truth={}",
            test_name,
            hex_string(&reference_result),
            hex_string(&ground_truth)
        );

        // Test circuit implementation
        let circuit_result = test_circuit_with_message(message);
        assert_eq!(
            circuit_result,
            ground_truth,
            "Circuit mismatch for {}: circuit={}, ground_truth={}",
            test_name,
            hex_string(&circuit_result),
            hex_string(&ground_truth)
        );

        // Ensure both implementations match each other
        assert_eq!(
            &circuit_result[..],
            &reference_result[..],
            "Circuit vs Reference mismatch for {}: circuit={}, ref={}",
            test_name,
            hex_string(&circuit_result),
            hex_string(&reference_result)
        );
    }

    /// Test all predefined test vectors
    #[test]
    fn test_all_predefined_vectors() {
        let vectors = get_all_test_vectors();
        for (message, name) in vectors {
            validate_implementations(&message, &name);
        }
    }

    /// Test random messages to ensure robustness
    #[test]
    fn test_random_messages() {
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..100 {
            let size = rng.random_range(0..1000);
            let mut message = vec![0u8; size];
            rng.fill(&mut message[..]);

            let test_name = format!("random_{}_size_{}", i, size);
            validate_implementations(&message, &test_name);
        }
    }

    /// Helper function to test circuit with any message
    fn test_circuit_with_message(input: &[u8]) -> [u8; 64] {
        let builder = CircuitBuilder::new();
        // Create circuit with exact size for the input
        let circuit_impl = Blake2bCircuit::new_with_length(&builder, input.len());
        let circuit = builder.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        // Compute expected digest using reference implementation
        let expected_digest_vec = reference::blake2b(input, 64);
        let mut expected_digest = [0u8; 64];
        expected_digest.copy_from_slice(&expected_digest_vec);

        // Use the new populate methods
        circuit_impl.populate_message(&mut w, input);
        circuit_impl.populate_digest(&mut w, &expected_digest);

        // Populate wire witness
        circuit.populate_wire_witness(&mut w).unwrap();

        // Extract circuit output (which should match the expected digest)
        let circuit_hash: [u8; 64] = core::array::from_fn(|i| {
            let word_idx = i / 8;
            let byte_idx = i % 8;
            let word_val = w[circuit_impl.digest[word_idx]].0;
            ((word_val >> (byte_idx * 8)) & 0xFF) as u8
        });

        // Verify constraints
        verify_constraints(cs, &w.into_value_vec()).unwrap();

        circuit_hash
    }
}
