// Copyright 2026 The Binius Developers

use std::iter;

use binius_field::BinaryField;

use super::common::{FRIParams, vcs_optimal_layers_depths_iter};
use crate::merkle_tree::MerkleTreeScheme;

/// Computes the exact byte-size of a FRI proof (including the initial commitment) without running
/// the prover.
///
/// This accounts for:
/// - **Message channel**: the initial codeword commitment and all round commitments (digests
///   observed by Fiat-Shamir).
/// - **Decommitment channel**: the terminal codeword, Merkle layer digests, per-query branch
///   digests, and per-query coset field values.
///
/// The estimate assumes non-hiding proofs (salt_len = 0).
pub fn proof_size<F, VCS>(params: &FRIParams<F>, vcs: &VCS) -> usize
where
    F: BinaryField,
    VCS: MerkleTreeScheme<F>,
{
    let digest_size = std::mem::size_of::<VCS::Digest>();

    // Serialized byte-size of a single field element.
    let value_size = {
        let mut buf = Vec::new();
        F::default()
            .serialize(&mut buf)
            .expect("default element can be serialized to a resizable buffer");
        buf.len()
    };

    let n_test_queries = params.n_test_queries();

    // Message channel: initial codeword commitment + batched codeword commitment + one per fold
    // arity.
    let commitment_msg_size = (2 + params.fold_arities().len()) * digest_size;

    // Terminal codeword sent in the clear: 2^(log_terminal_dim + log_inv_rate) field elements.
    let log_terminal_dim = params.n_final_challenges();
    let log_inv_rate = params.rs_code().log_inv_rate();
    let terminate_codeword_size = (1 << (log_terminal_dim + log_inv_rate)) * value_size;

    // The arities for each oracle: first the batch interleave fold, then each fold arity.
    let arities: Vec<usize> = iter::once(params.log_batch_size())
        .chain(params.fold_arities().iter().copied())
        .collect();

    // Compute the Merkle proof sizes and coset value sizes across all n_oracles non-terminal
    // oracles. vcs_optimal_layers_depths_iter yields layer depths in the same order as arities.
    let layer_depths: Vec<usize> = vcs_optimal_layers_depths_iter(params, vcs).collect();

    let mut merkle_sizes = 0;
    let mut coset_values_size = 0;
    let mut log_n_cosets = params.log_len();

    for (i, &arity) in arities.iter().enumerate() {
        log_n_cosets -= arity;

        // VCS proof_size covers both the 2^layer_depth layer digests sent once AND the
        // (tree_depth - layer_depth) * n_test_queries branch digests sent across all queries.
        let tree_len = 1 << log_n_cosets;
        merkle_sizes += vcs
            .proof_size(tree_len, n_test_queries, layer_depths[i])
            .expect("layer depth computed with optimal_verify_layer must be valid");

        // Each query opens a coset of 2^arity field values for this oracle.
        coset_values_size += n_test_queries * (1 << arity) * value_size;
    }

    commitment_msg_size + terminate_codeword_size + merkle_sizes + coset_values_size
}
