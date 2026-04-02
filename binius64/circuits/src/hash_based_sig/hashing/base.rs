// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

use crate::{concat::Concat, fixed_byte_vec::ByteVec, keccak::Keccak256};

/// Verify a tweaked Keccak-256 circuit with custom terms.
///
/// This function provides the common setup for both message and chain tweaking,
/// which both follow the pattern: `Keccak256(domain_param || tweak_byte || additional_data)`
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `domain_param_wires` - The cryptographic domain parameter wires
/// * `domain_param_len` - The actual domain parameter length in bytes
/// * `tweak_byte` - The tweak byte value (MESSAGE_TWEAK or CHAIN_TWEAK)
/// * `additional_terms` - Additional concatenation terms after param and tweak
/// * `total_message_len` - Total length of the concatenated message
/// * `digest` - Output digest wires
///
/// # Returns
/// A `Keccak` instance that computes the tweaked hash
pub fn circuit_tweaked_keccak(
    builder: &CircuitBuilder,
    domain_param_wires: Vec<Wire>,
    domain_param_len: usize,
    tweak_byte: u8,
    additional_terms: Vec<ByteVec>,
    total_message_len: usize,
    digest: [Wire; 4],
) -> Keccak256 {
    // Create the message wires for Keccak (LE-packed)
    let n_message_words = total_message_len.div_ceil(8);
    let message_le: Vec<Wire> = (0..n_message_words)
        .map(|_| builder.add_witness())
        .collect();
    let len = builder.add_constant_64(total_message_len as u64);

    let keccak = Keccak256::new(builder, len, digest, message_le.clone());

    let mut terms = Vec::new();
    let domain_param_term = ByteVec {
        len_bytes: builder.add_constant_64(domain_param_len as u64),
        data: domain_param_wires,
    };
    terms.push(domain_param_term);

    let tweak_wire = builder.add_constant_64(tweak_byte as u64);
    let tweak_term = ByteVec {
        len_bytes: builder.add_constant_64(1),
        data: vec![tweak_wire],
    };
    terms.push(tweak_term);
    terms.extend(additional_terms);

    let _message_structure_verifier = Concat::new(builder, len, message_le, terms);
    keccak
}
