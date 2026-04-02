// Copyright 2025 Irreducible Inc.
use binius_field::{BinaryField, PackedField};
use binius_math::BinarySubspace;
use itertools::Itertools;

use super::ntt_lookup::NTTLookup;

/// Creates a new NTTLookup from a prover message domain.
///
/// # Arguments
///
/// * `prover_message_domain` - The domain of the polynomial which is the prover's first round
///   message. Must have dimension at least 1.
///
/// # Panics
///
/// Panics if the prover message domain has dimension less than 1.
///
/// # Details
///
/// This constructor sets up the NTT input and output domains based on the prover message
/// domain:
/// - The NTT input domain is a subspace of dimension `prover_message_domain.dim() - 1`
/// - The NTT output domain is the input domain shifted by a specific basis element
/// - An NTT lookup table is precomputed for efficient polynomial evaluation
pub fn ntt_lookup_from_prover_message_domain<PNTTDomain>(
    prover_message_domain: BinarySubspace<PNTTDomain::Scalar>,
) -> NTTLookup<PNTTDomain>
where
    PNTTDomain: PackedField,
    PNTTDomain::Scalar: BinaryField + From<u8>,
{
    assert!(prover_message_domain.dim() >= 1);
    let basis = prover_message_domain.basis();

    let basis_element_added_to_ntt_input_to_get_ntt_output = basis[prover_message_domain.dim() - 1];

    let ntt_input_domain = prover_message_domain
        .clone()
        .reduce_dim(prover_message_domain.dim() - 1);

    let ntt_output_domain = ntt_input_domain
        .iter()
        .map(|i| basis_element_added_to_ntt_input_to_get_ntt_output + i)
        .collect_vec();

    NTTLookup::new(&ntt_input_domain, &ntt_output_domain)
}
