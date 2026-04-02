// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{BinaryField, PackedField};
use binius_iop::{fri::FRIParams, merkle_tree::MerkleTreeScheme};
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT};
use binius_utils::{rand::par_rand, rayon::prelude::*};
use rand::{CryptoRng, rngs::StdRng};

use super::error::Error;
use crate::merkle_tree::MerkleTreeProver;

#[derive(Debug)]
pub struct CommitOutput<P: PackedField, VCSCommitment, VCSCommitted> {
    pub commitment: VCSCommitment,
    pub committed: VCSCommitted,
    pub codeword: FieldBuffer<P>,
}

/// Encodes and commits the input message.
///
/// ## Arguments
///
/// * `rs_code` - the Reed-Solomon code to use for encoding
/// * `params` - common FRI protocol parameters.
/// * `merkle_prover` - the merke tree prover to use for committing
/// * `message` - the interleaved message to encode and commit
pub fn commit_interleaved<F, P, NTT, MerkleProver, VCS>(
    params: &FRIParams<F>,
    ntt: &NTT,
    merkle_prover: &MerkleProver,
    message: FieldSlice<P>,
) -> Result<CommitOutput<P, VCS::Digest, MerkleProver::Committed>, Error>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
    VCS: MerkleTreeScheme<F>,
{
    if message.log_len() != params.log_msg_len() {
        return Err(Error::InvalidArgs(
            "interleaved message length does not match code parameters".to_string(),
        ));
    }

    let rs_code = params.rs_code();
    let log_batch_size = params.log_batch_size();

    let _scope = tracing::debug_span!(
        "FRI Commit",
        log_batch_size,
        log_dim = rs_code.log_dim(),
        log_inv_rate = rs_code.log_inv_rate(),
        field_bits = F::N_BITS,
    )
    .entered();

    let encoded = tracing::debug_span!("Reed-Solomon Encode")
        .in_scope(|| rs_code.encode_batch(ntt, message.to_ref(), log_batch_size));

    let merkle_tree_span = tracing::debug_span!("Merkle Tree").entered();
    let (commitment, vcs_committed) = if log_batch_size > P::LOG_WIDTH {
        let iterated_big_chunks = to_par_scalar_big_chunks(encoded.as_ref(), 1 << log_batch_size);
        merkle_prover.commit_iterated(iterated_big_chunks)?
    } else {
        let iterated_small_chunks =
            to_par_scalar_small_chunks(encoded.as_ref(), 1 << log_batch_size);
        merkle_prover.commit_iterated(iterated_small_chunks)?
    };
    drop(merkle_tree_span);

    Ok(CommitOutput {
        commitment: commitment.root,
        committed: vcs_committed,
        codeword: encoded,
    })
}

#[derive(Debug)]
pub struct CommitMaskedOutput<P: PackedField, VCSCommitment, VCSCommitted> {
    pub commitment: VCSCommitment,
    pub committed: VCSCommitted,
    pub codeword: FieldBuffer<P>,
    pub mask: FieldBuffer<P>,
}

/// Generates a random mask, interleaves it with the message, and commits.
///
/// This is used for zero-knowledge FRI commitments. The function generates a random mask of
/// equal length to the input message, concatenates `message || mask` as the interleaved message
/// (with `log_batch_size = 1`), performs Reed-Solomon encoding, and commits via Merkle tree.
///
/// ## Arguments
///
/// * `params` - FRI parameters. Must have `log_batch_size() == 1` and `rs_code().log_dim() ==
///   message.log_len()`.
/// * `ntt` - the additive NTT for Reed-Solomon encoding
/// * `merkle_prover` - the Merkle tree prover for commitments
/// * `message` - the raw message to commit (not doubled)
/// * `rng` - cryptographic RNG for mask generation
///
/// ## Returns
///
/// A [`CommitMaskedOutput`] containing the commitment, Merkle tree data, encoded codeword, and
/// the generated mask buffer.
pub fn commit_masked<F, P, NTT, MerkleProver, VCS>(
    params: &FRIParams<F>,
    ntt: &NTT,
    merkle_prover: &MerkleProver,
    message: FieldSlice<P>,
    mut rng: impl CryptoRng,
) -> Result<CommitMaskedOutput<P, VCS::Digest, MerkleProver::Committed>, Error>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
    VCS: MerkleTreeScheme<F>,
{
    assert_eq!(
        params.log_batch_size(),
        1,
        "commit_masked requires log_batch_size == 1"
    );
    assert_eq!(
        params.rs_code().log_dim(),
        message.log_len(),
        "commit_masked requires rs_code().log_dim() == message.log_len()"
    );

    // Generate random mask of equal length to message.
    let log_len = message.log_len();
    let packed_len = 1usize << log_len.saturating_sub(P::LOG_WIDTH);
    let mask = FieldBuffer::<P>::new(
        log_len,
        par_rand::<StdRng, _, _>(packed_len, &mut rng, P::random).collect(),
    );

    // TODO: The concatenation here is sequential and a performance issue. Ideally, commit should
    // not allocate and copy the memory into a temp buffer.
    let combined_packed_len = packed_len * 2;
    let mut combined_values = Vec::with_capacity(combined_packed_len);
    combined_values.extend_from_slice(message.as_ref());
    combined_values.extend_from_slice(mask.as_ref());
    let combined = FieldBuffer::new(log_len + 1, combined_values.into_boxed_slice());

    let CommitOutput {
        commitment,
        committed,
        codeword,
    } = commit_interleaved(params, ntt, merkle_prover, combined.to_ref())?;

    Ok(CommitMaskedOutput {
        commitment,
        committed,
        codeword,
        mask,
    })
}

/// Creates a parallel iterator over scalars of subfield elements. Assumes chunk_size to be a power
/// of two.
fn to_par_scalar_big_chunks<P>(
    packed_slice: &[P],
    chunk_size: usize,
) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + '_>
where
    P: PackedField,
{
    packed_slice
        .par_chunks(chunk_size / P::WIDTH)
        .map(|chunk| PackedField::iter_slice(chunk))
}

fn to_par_scalar_small_chunks<P>(
    packed_slice: &[P],
    chunk_size: usize,
) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + '_>
where
    P: PackedField,
{
    (0..packed_slice.len() * P::WIDTH)
        .into_par_iter()
        .step_by(chunk_size)
        .map(move |start_index| {
            let packed_item = &packed_slice[start_index / P::WIDTH];
            packed_item
                .iter()
                .skip(start_index % P::WIDTH)
                .take(chunk_size)
        })
}

#[cfg(test)]
mod tests {
    use binius_field::{
        BinaryField128bGhash as B128, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
        PackedBinaryGhash4x128b,
    };
    use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
    use binius_iop::fri::FRIParams;
    use binius_math::{
        BinarySubspace, FieldBuffer,
        ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
        test_utils::{random_field_buffer, random_scalars},
    };
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::merkle_tree::prover::BinaryMerkleTreeProver;

    #[test]
    fn test_commit_masked() {
        type F = B128;
        type P = PackedBinaryGhash1x128b;

        let mut rng = StdRng::seed_from_u64(42);

        let log_dim = 6;
        let log_inv_rate = 1;
        let log_batch_size = 1;
        let n_test_queries = 3;

        let merkle_prover = BinaryMerkleTreeProver::<F, StdDigest, _>::new(
            ParallelCompressionAdaptor::new(StdCompression::default()),
        );

        let subspace = BinarySubspace::with_dim(log_dim + log_inv_rate);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
        let ntt = NeighborsLastSingleThread::new(domain_context);

        let params = FRIParams::with_strategy(
            &ntt,
            merkle_prover.scheme(),
            log_dim + log_batch_size,
            Some(log_batch_size),
            log_inv_rate,
            n_test_queries,
            &binius_iop::fri::ConstantArityStrategy::new(2),
        )
        .unwrap();

        assert_eq!(params.log_batch_size(), 1);
        assert_eq!(params.rs_code().log_dim(), log_dim);

        let message = random_field_buffer::<P>(&mut rng, log_dim);

        let output: CommitMaskedOutput<P, _, _> =
            commit_masked(&params, &ntt, &merkle_prover, message.to_ref(), &mut rng).unwrap();

        // Verify mask has correct dimensions.
        assert_eq!(output.mask.log_len(), log_dim);

        // Verify the codeword has expected length (log_dim + log_batch_size + log_inv_rate).
        assert_eq!(
            output.codeword.log_len(),
            log_dim + log_batch_size + log_inv_rate
        );
    }

    #[test]
    fn test_parallel_iterator() {
        let mut rng = StdRng::seed_from_u64(0);

        // Compare results for small and large chunk sizes to ensure that they're identical
        let data = random_scalars::<B128>(&mut rng, 64);

        let data_packed_2 = FieldBuffer::<PackedBinaryGhash2x128b, _>::from_values(&data);
        let data_packed_4 = FieldBuffer::<PackedBinaryGhash4x128b, _>::from_values(&data);

        let packing_smaller_than_chunk = to_par_scalar_big_chunks(data_packed_2.as_ref(), 2);
        let packing_bigger_than_chunk = to_par_scalar_small_chunks(data_packed_4.as_ref(), 2);

        let collected_smaller: Vec<_> = packing_smaller_than_chunk
            .map(|inner| inner.collect::<Vec<_>>())
            .collect();
        let collected_bigger: Vec<_> = packing_bigger_than_chunk
            .map(|inner| inner.collect::<Vec<_>>())
            .collect();
        assert_eq!(collected_smaller, collected_bigger);
    }
}
