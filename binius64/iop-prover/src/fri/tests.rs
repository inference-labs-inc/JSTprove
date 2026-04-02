// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::vec;

use binius_field::{
    BinaryField, BinaryField128bGhash as B128, PackedBinaryGhash1x128b, PackedField,
};
use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
use binius_iop::fri::{self, FRIFoldVerifier, FRIParams, verify::FRIQueryVerifier};
use binius_math::{
    BinarySubspace, ReedSolomonCode,
    multilinear::evaluate::evaluate,
    ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
    test_utils::{Packed128b, random_field_buffer},
};
use binius_transcript::{
    ProverTranscript,
    fiat_shamir::{CanSample, HasherChallenger},
};
use binius_utils::checked_arithmetics::log2_strict_usize;
use rand::prelude::*;

use super::{CommitOutput, FRIFoldProver, FoldRoundOutput, commit_interleaved};
use crate::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};

type StdChallenger = HasherChallenger<StdDigest>;

fn test_commit_prove_verify_success<F, P>(
    log_dimension: usize,
    log_inv_rate: usize,
    log_batch_size: usize,
    arities: &[usize],
) where
    F: BinaryField,
    P: PackedField<Scalar = F>,
{
    let mut rng = StdRng::seed_from_u64(0);

    let parallel_compression = ParallelCompressionAdaptor::new(StdCompression::default());
    let merkle_prover = BinaryMerkleTreeProver::<_, StdDigest, _>::new(parallel_compression);

    let committed_rs_code = ReedSolomonCode::<F>::new(log_dimension, log_inv_rate);

    let n_test_queries = 3;
    let params = FRIParams::new(
        committed_rs_code,
        log_batch_size,
        arities.to_vec(),
        n_test_queries,
    )
    .unwrap();

    let subspace = BinarySubspace::with_dim(params.rs_code().log_len());
    let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
    let ntt = NeighborsLastSingleThread::new(domain_context);

    let n_round_commitments = arities.len();

    // Generate a random message
    let msg = random_field_buffer::<P>(&mut rng, params.log_msg_len());

    // Prover commits the message
    let CommitOutput {
        commitment: mut codeword_commitment,
        committed: codeword_committed,
        codeword,
    } = commit_interleaved(&params, &ntt, &merkle_prover, msg.to_ref()).unwrap();

    // Run the prover to generate the proximity proof
    let mut round_prover =
        FRIFoldProver::new(&params, &ntt, &merkle_prover, codeword, &codeword_committed).unwrap();

    let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
    prover_challenger.message().write(&codeword_commitment);

    // Note: The prover does an initial fold round before receiving any challenges
    // This is round 0, which won't produce a commitment when log_batch_size > 0
    let fold_round_output = round_prover.execute_fold_round().unwrap();
    if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
        prover_challenger.message().write(&round_commitment);
    }

    for _i in 0..params.n_fold_rounds() {
        let challenge = prover_challenger.sample();
        round_prover.receive_challenge(challenge);

        let fold_round_output = round_prover.execute_fold_round().unwrap();
        if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
            prover_challenger.message().write(&round_commitment);
        }
    }

    round_prover.finish_proof(&mut prover_challenger).unwrap();
    // Now run the verifier
    let mut verifier_challenger = prover_challenger.into_verifier();
    codeword_commitment = verifier_challenger.message().read().unwrap();
    let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());

    assert_eq!(params.fold_arities().len(), n_round_commitments);

    // The prover executes fold rounds starting from round 0, then receives challenges and continues
    // We need to match this pattern in the verifier
    let mut fri_fold_verifier = FRIFoldVerifier::new(&params);

    // Process initial round (before any challenges) - round 0
    fri_fold_verifier
        .process_round(&mut verifier_challenger.message())
        .unwrap();

    // Process remaining rounds with challenges
    for _ in 0..params.n_fold_rounds() {
        verifier_challenges.push(verifier_challenger.sample());
        fri_fold_verifier
            .process_round(&mut verifier_challenger.message())
            .unwrap();
    }

    let round_commitments = fri_fold_verifier.finalize().unwrap();

    assert_eq!(verifier_challenges.len(), params.n_fold_rounds());

    let verifier = FRIQueryVerifier::new(
        &params,
        merkle_prover.scheme(),
        &codeword_commitment,
        &round_commitments,
        &verifier_challenges,
    )
    .unwrap();

    let mut cloned_verifier_challenger = verifier_challenger.clone();

    let terminate_codeword_len =
        1 << (params.n_final_challenges() + params.rs_code().log_inv_rate());

    let mut advice = verifier_challenger.decommitment();
    let terminate_codeword: Vec<F> = advice.read_scalar_slice(terminate_codeword_len).unwrap();

    let log_batch_size =
        log2_strict_usize(terminate_codeword.len()).saturating_sub(params.rs_code().log_inv_rate());

    let (commitment, tree) = merkle_prover
        .commit(&terminate_codeword, 1 << log_batch_size)
        .unwrap();

    // Ensure that the terminate_codeword commitment is correct
    let last_round_commitment = round_commitments.last().unwrap_or(&codeword_commitment);
    assert_eq!(*last_round_commitment, commitment.root);

    // Verify that the Merkle tree has exactly inv_rate leaves.
    assert_eq!(tree.log_len, params.rs_code().log_inv_rate());

    // check c == t(r'_0, ..., r'_{\ell-1})
    // note that the prover is claiming that the final_message is [c]
    let mut eval_point = verifier_challenges.clone();
    eval_point.reverse();
    let computed_eval = evaluate(&msg, &eval_point);

    let final_fri_value = verifier.verify(&mut cloned_verifier_challenger).unwrap();
    assert_eq!(computed_eval, final_fri_value);
}

#[test]
fn test_commit_prove_verify_success_128b_full() {
    // This tests the case where we have a round commitment for every round
    let log_dimension = 8;
    let log_final_dimension = 1;
    let log_inv_rate = 2;
    let arities = vec![1; log_dimension - log_final_dimension];

    // TODO: Make this test pass with non-trivial packing width
    test_commit_prove_verify_success::<B128, PackedBinaryGhash1x128b>(
        log_dimension,
        log_inv_rate,
        0,
        &arities,
    );
}

#[test]
fn test_commit_prove_verify_success_128b_higher_arity() {
    let log_dimension = 8;
    let log_inv_rate = 2;
    let arities = [3, 2, 1];

    // TODO: Make this test pass with non-trivial packing width
    test_commit_prove_verify_success::<B128, PackedBinaryGhash1x128b>(
        log_dimension,
        log_inv_rate,
        0,
        &arities,
    );
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved() {
    let log_dimension = 6;
    let log_inv_rate = 2;
    let log_batch_size = 2;
    let arities = [3, 2, 1];

    test_commit_prove_verify_success::<B128, Packed128b>(
        log_dimension,
        log_inv_rate,
        log_batch_size,
        &arities,
    );
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved_packed() {
    let log_dimension = 6;
    let log_inv_rate = 2;
    let log_batch_size = 2;
    let arities = [3, 2, 1];

    test_commit_prove_verify_success::<B128, Packed128b>(
        log_dimension,
        log_inv_rate,
        log_batch_size,
        &arities,
    );
}

#[test]
fn test_commit_prove_verify_success_without_folding() {
    let log_dimension = 4;
    let log_inv_rate = 2;
    let log_batch_size = 2;

    test_commit_prove_verify_success::<B128, Packed128b>(
        log_dimension,
        log_inv_rate,
        log_batch_size,
        &[],
    );
}

/// Runs the FRI prover and returns the proof bytes along with the FRI params and Merkle scheme,
/// so the caller can check the estimated proof size.
fn generate_fri_proof<F, P>(
    log_dimension: usize,
    log_inv_rate: usize,
    log_batch_size: usize,
    arities: &[usize],
) -> (
    Vec<u8>,
    FRIParams<F>,
    binius_iop::merkle_tree::BinaryMerkleTreeScheme<F, StdDigest, StdCompression>,
)
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
{
    let mut rng = StdRng::seed_from_u64(0);

    let parallel_compression = ParallelCompressionAdaptor::new(StdCompression::default());
    let merkle_prover = BinaryMerkleTreeProver::<_, StdDigest, _>::new(parallel_compression);

    let committed_rs_code = ReedSolomonCode::<F>::new(log_dimension, log_inv_rate);

    let n_test_queries = 3;
    let params = FRIParams::new(
        committed_rs_code,
        log_batch_size,
        arities.to_vec(),
        n_test_queries,
    )
    .unwrap();

    let subspace = BinarySubspace::with_dim(params.rs_code().log_len());
    let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
    let ntt = NeighborsLastSingleThread::new(domain_context);

    let msg = random_field_buffer::<P>(&mut rng, params.log_msg_len());

    let CommitOutput {
        commitment: codeword_commitment,
        committed: codeword_committed,
        codeword,
    } = commit_interleaved(&params, &ntt, &merkle_prover, msg.to_ref()).unwrap();

    let mut round_prover =
        FRIFoldProver::new(&params, &ntt, &merkle_prover, codeword, &codeword_committed).unwrap();

    let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
    prover_transcript.message().write(&codeword_commitment);

    let fold_round_output = round_prover.execute_fold_round().unwrap();
    if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
        prover_transcript.message().write(&round_commitment);
    }

    for _ in 0..params.n_fold_rounds() {
        let challenge = prover_transcript.sample();
        round_prover.receive_challenge(challenge);

        let fold_round_output = round_prover.execute_fold_round().unwrap();
        if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
            prover_transcript.message().write(&round_commitment);
        }
    }

    round_prover.finish_proof(&mut prover_transcript).unwrap();

    let scheme = merkle_prover.scheme().clone();
    let proof_bytes = prover_transcript.finalize();
    (proof_bytes, params, scheme)
}

#[test]
fn test_proof_size_higher_arity() {
    let (proof_bytes, params, scheme) =
        generate_fri_proof::<B128, PackedBinaryGhash1x128b>(8, 2, 0, &[3, 2, 1]);
    assert_eq!(proof_bytes.len(), fri::proof_size(&params, &scheme));
}

#[test]
fn test_proof_size_interleaved() {
    let (proof_bytes, params, scheme) = generate_fri_proof::<B128, Packed128b>(6, 2, 2, &[3, 2, 1]);
    assert_eq!(proof_bytes.len(), fri::proof_size(&params, &scheme));
}

#[test]
fn test_proof_size_no_folding() {
    let (proof_bytes, params, scheme) = generate_fri_proof::<B128, Packed128b>(4, 2, 2, &[]);
    assert_eq!(proof_bytes.len(), fri::proof_size(&params, &scheme));
}
