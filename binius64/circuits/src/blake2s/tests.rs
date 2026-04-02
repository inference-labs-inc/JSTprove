// Copyright 2025 Irreducible Inc.

use binius_core::verify::verify_constraints;
use binius_frontend::CircuitBuilder;
use blake2::{Blake2s256, Digest};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use rstest::rstest;

use super::Blake2s;

#[rstest]
#[case(0, 1)] // Empty message - edge case
#[case(1, 1)] // Single byte - minimal non-empty
#[case(3, 2)] // 3 bytes - partial word (less than 4 bytes)
#[case(64, 16)] // 64 bytes - exactly one block
#[case(65, 17)] // 65 bytes - crosses block boundary
fn test_blake2s_circuit(#[case] message_len_bytes: usize, #[case] max_message_len_words: usize) {
    // Create test message with deterministic random bytes seeded by the length inputs
    let seed = ((message_len_bytes as u64) << 32) | (max_message_len_words as u64);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut message = vec![0u8; message_len_bytes];
    rng.fill_bytes(&mut message);

    // Compute expected digest using reference implementation
    let mut hasher = Blake2s256::new();
    hasher.update(&message);
    let expected_digest = hasher.finalize();

    // Build circuit with specified max message length
    // Note: Blake2s expects length in bytes, not words
    let max_bytes = max_message_len_words * 4;
    assert!(
        message_len_bytes <= max_bytes,
        "Message length {} exceeds max capacity {} bytes",
        message_len_bytes,
        max_bytes
    );

    let mut builder = CircuitBuilder::new();
    let blake2s = Blake2s::new_witness(&mut builder, message_len_bytes);
    let circuit = builder.build();

    // Create and populate witness
    let mut witness = circuit.new_witness_filler();
    blake2s.populate_message(&mut witness, &message);
    blake2s.populate_digest(&mut witness, expected_digest.as_slice().try_into().unwrap());

    // Verify circuit accepts the witness
    circuit
        .populate_wire_witness(&mut witness)
        .expect("Circuit should accept valid witness");

    // Verify all constraints are satisfied
    let cs = circuit.constraint_system();
    verify_constraints(cs, &witness.into_value_vec()).expect("All constraints should be satisfied");
}
