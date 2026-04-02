// Copyright 2025 Irreducible Inc.
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::hashing::circuit_chain_hash;
use crate::{keccak::Keccak256, multiplexer::multi_wire_multiplex};

/// Verifies a hash chain for hash-based signature schemes using Keccak-256.
///
/// This function iteratively hashes a signature hash from a starting position
/// and verifies that the final result matches an expected public key element.
///
/// # Hash Chain Structure
///
/// A hash chain is a sequence of values where each value is computed by hashing the previous one:
/// ```text
/// start → H(start) → H(H(start)) → ... → end
/// ```
///
/// # Circuit Operation
///
/// The circuit starts at position `starting_position` and performs iterations up to
/// `max_chain_len`. The actual number of hash iterations performed depends on the
/// `starting_position`:
/// - If `starting_position = 0`, it performs `max_chain_len` iterations
/// - If `starting_position = n`, it performs `max_chain_len - n` iterations
///
/// Each iteration:
/// 1. Takes the current hash value
/// 2. Applies Keccak-256 with appropriate tweaking parameters
/// 3. Uses the result as input for the next iteration
///
/// After all iterations, it verifies the final hash equals `public_key_element`.
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param` - Cryptographic domain parameter as 64-bit packed wires (LE format)
/// * `param_len` - Actual byte length of the parameter (must be less than or equal to
///   domain_param.len() * 8)
/// * `chain_index` - Index of this chain in the signature structure
/// * `signature_hash` - Starting hash value (32 bytes as 4x64-bit LE wires)
/// * `starting_position` - Starting position in the hash chain (from codeword). The actual number
///   of iterations performed is `max_chain_len - starting_position`
/// * `max_chain_len` - Maximum chain length
/// * `public_key_element` - Expected final hash value (32 bytes as 4x64-bit LE wires)
///
/// # Returns
///
/// A vector of `Keccak` hashers that need to be populated with witness values.
/// The number of hashers equals the maximum chain length supported.
#[allow(clippy::too_many_arguments)]
pub fn circuit_chain(
    builder: &CircuitBuilder,
    domain_param: &[Wire],
    param_len: usize,
    chain_index: Wire,
    signature_hash: [Wire; 4],
    starting_position: Wire,
    max_chain_len: u64,
    public_key_element: [Wire; 4],
) -> Vec<Keccak256> {
    assert!(
        param_len <= domain_param.len() * 8,
        "param_len {} exceeds maximum capacity {} of domain_param wires",
        param_len,
        domain_param.len() * 8
    );
    let mut hashers = Vec::with_capacity(max_chain_len as usize);
    let mut current_hash = signature_hash;

    let one = builder.add_constant(Word::ONE);
    let zero = builder.add_constant(Word::ZERO);
    let max_chain_len_wire = builder.add_constant_64(max_chain_len);

    // Build the hash chain
    for step in 0..max_chain_len {
        let step_wire = builder.add_constant_64(step);
        let (position, _) = builder.iadd_cin_cout(step_wire, starting_position, zero);
        let (position_plus_one, _) = builder.iadd_cin_cout(position, one, zero);

        let next_hash = std::array::from_fn(|_| builder.add_witness());
        let keccak = circuit_chain_hash(
            builder,
            domain_param.to_vec(),
            param_len,
            current_hash,
            chain_index,
            position_plus_one,
            next_hash,
        );

        hashers.push(keccak);

        // Conditionally select the hash based on whether position + 1 < max_chain_len
        // If position + 1 < max_chain_len, use next_hash, otherwise keep current_hash
        // icmp_ult returns an MSB-bool, but multiplexer needs the result in LSB
        let position_lt_max_chain_len_msb = builder.icmp_ult(position_plus_one, max_chain_len_wire);
        let position_lt_max_chain_len = builder.shr(position_lt_max_chain_len_msb, 63);
        current_hash = multi_wire_multiplex(
            builder,
            &[&current_hash, &next_hash],
            position_lt_max_chain_len,
        )
        .try_into()
        .expect("multi_wire_multiplex should return 4 wires");
    }

    // Assert that the final hash matches the expected public_key_element
    builder.assert_eq_v("hash_chain_end_check", current_hash, public_key_element);
    hashers
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::util::pack_bytes_into_wires_le;
    use proptest::{prelude::*, strategy::Just};
    use sha3::{Digest, Keccak256};

    use super::{super::hashing::build_chain_hash, *};

    proptest! {
        #[test]
        fn test_circuit_chain(
            (starting_position_val, max_chain_len) in (0u64..10).prop_flat_map(|start_pos| {
                // max_chain_len must be > starting_position_val for any hashing to occur
                // Generate max_chain_len in range [start_pos + 1, start_pos + 8]
                (Just(start_pos), (start_pos + 1)..=(start_pos + 8))
            }),
            chain_index_val in 0u64..100,
            domain_param_bytes in prop::collection::vec(any::<u8>(), 1..120), // Variable length domain param (1-119 bytes)
            signature_hash_bytes in prop::array::uniform32(any::<u8>()),
        ) {
            let builder = CircuitBuilder::new();

            let param_wire_count = domain_param_bytes.len().div_ceil(8);
            let domain_param: Vec<Wire> = (0..param_wire_count).map(|_| builder.add_inout()).collect();
            let chain_index = builder.add_inout();
            let signature_hash: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());
            let starting_position = builder.add_inout();
            let public_key_element: [Wire; 4] = std::array::from_fn(|_| builder.add_inout());

            let hashers = circuit_chain(
                &builder,
                &domain_param,
                domain_param_bytes.len(),
                chain_index,
                signature_hash,
                starting_position,
                max_chain_len,
                public_key_element,
            );

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();

            w[chain_index] = Word::from_u64(chain_index_val);
            w[starting_position] = Word::from_u64(starting_position_val);

            // Populate domain_param wires (they're reused for all hashers)
            pack_bytes_into_wires_le(&mut w, &domain_param, &domain_param_bytes);

            // Track current hash through the chain
            let mut current_hash: [u8; 32] = signature_hash_bytes;

            for (step, keccak) in hashers.iter().enumerate() {
                // Calculate position for this step
                let position = step as u64 + starting_position_val;
                let position_plus_one = position + 1;

                // Build the message for this hash using current_hash
                let message = build_chain_hash(
                    &domain_param_bytes,
                    &current_hash,
                    chain_index_val,
                    position_plus_one,
                );

                // Populate the Keccak circuit
                keccak.populate_message(&mut w, &message);

                // Compute the hash
                let digest: [u8; 32] = Keccak256::digest(&message).into();
                keccak.populate_digest(&mut w, digest);

                // Update current_hash for next iteration if this hash is selected
                // The multiplexer in the circuit selects next_hash when position_plus_one < max_chain_len
                if position_plus_one < max_chain_len {
                    current_hash = digest;
                }
            }

            pack_bytes_into_wires_le(&mut w, &public_key_element, &current_hash);
            pack_bytes_into_wires_le(&mut w, &signature_hash, &signature_hash_bytes);
            circuit.populate_wire_witness(&mut w).unwrap();

            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }
}
