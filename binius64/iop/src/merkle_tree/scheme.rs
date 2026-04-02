// Copyright 2024-2025 Irreducible Inc.

use std::{array, fmt::Debug, marker::PhantomData};

use binius_hash::{PseudoCompressionFunction, hash_serialize};
use binius_transcript::{Buf, TranscriptReader};
use binius_utils::{
    DeserializeBytes, SerializeBytes,
    checked_arithmetics::{log2_ceil_usize, log2_strict_usize},
};
use digest::{Digest, Output, core_api::BlockSizeUser};
use getset::{CopyGetters, Getters};

use super::{
    error::{Error, VerificationError},
    merkle_tree_vcs::MerkleTreeScheme,
};

#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct BinaryMerkleTreeScheme<T, H, C> {
    #[getset(get = "pub")]
    compression: C,
    #[getset(get_copy = "pub")]
    salt_len: usize,
    // This makes it so that `BinaryMerkleTreeScheme` remains Send + Sync
    // See https://doc.rust-lang.org/nomicon/phantom-data.html#table-of-phantomdata-patterns
    _phantom: PhantomData<fn() -> (T, H)>,
}

impl<T, H, C> BinaryMerkleTreeScheme<T, H, C> {
    pub fn new(compression: C) -> Self {
        Self::hiding(compression, 0)
    }

    pub fn hiding(compression: C, salt_len: usize) -> Self {
        Self {
            compression,
            salt_len,
            _phantom: PhantomData,
        }
    }
}

impl<T, H, C> BinaryMerkleTreeScheme<T, H, C>
where
    T: SerializeBytes + DeserializeBytes,
    H: Digest + BlockSizeUser,
    C: PseudoCompressionFunction<Output<H>, 2>,
{
    fn compute_leaf_digest<B: Buf>(
        &self,
        values: &[T],
        proof: &mut TranscriptReader<B>,
    ) -> Result<Output<H>, Error> {
        let salt = proof.read_vec::<T>(self.salt_len)?;
        hash_serialize::<T, H>(values.iter().chain(&salt)).map_err(Error::Serialization)
    }
}

impl<T, H, C> MerkleTreeScheme<T> for BinaryMerkleTreeScheme<T, H, C>
where
    T: SerializeBytes + DeserializeBytes,
    H: Digest + BlockSizeUser,
    C: PseudoCompressionFunction<Output<H>, 2>,
{
    type Digest = Output<H>;

    /// This layer allows minimizing the proof size.
    fn optimal_verify_layer(&self, n_queries: usize, tree_depth: usize) -> usize {
        log2_ceil_usize(n_queries).min(tree_depth)
    }

    fn proof_size(&self, len: usize, n_queries: usize, layer_depth: usize) -> Result<usize, Error> {
        if !len.is_power_of_two() {
            return Err(Error::PowerOfTwoLengthRequired);
        }

        let log_len = log2_strict_usize(len);

        if layer_depth > log_len {
            return Err(Error::IncorrectLayerDepth);
        }

        Ok(((log_len - layer_depth) * n_queries + (1 << layer_depth))
            * <H as Digest>::output_size())
    }

    fn verify_vector<B: Buf>(
        &self,
        root: &Self::Digest,
        data: &[T],
        batch_size: usize,
        proof: &mut TranscriptReader<B>,
    ) -> Result<(), Error> {
        if !data.len().is_multiple_of(batch_size) {
            return Err(Error::IncorrectBatchSize);
        }

        let digests = data
            .chunks(batch_size)
            .map(|chunk| self.compute_leaf_digest(chunk, proof))
            .collect::<Result<Vec<_>, _>>()?;

        if fold_digests_vector_inplace(&self.compression, digests) != *root {
            return Err(VerificationError::InvalidProof.into());
        }
        Ok(())
    }

    fn verify_layer(
        &self,
        root: &Self::Digest,
        layer_depth: usize,
        layer_digests: &[Self::Digest],
    ) -> Result<(), Error> {
        if 1 << layer_depth != layer_digests.len() {
            return Err(VerificationError::IncorrectVectorLength.into());
        }

        let computed_root = fold_digests_vector_inplace(&self.compression, layer_digests.to_vec());
        if computed_root != *root {
            return Err(VerificationError::InvalidProof.into());
        }
        Ok(())
    }

    fn verify_opening<B: Buf>(
        &self,
        mut index: usize,
        values: &[T],
        layer_depth: usize,
        tree_depth: usize,
        layer_digests: &[Self::Digest],
        proof: &mut TranscriptReader<B>,
    ) -> Result<(), Error> {
        if (1 << layer_depth) != layer_digests.len() {
            return Err(VerificationError::IncorrectVectorLength.into());
        }

        if index >= (1 << tree_depth) {
            return Err(Error::IndexOutOfRange {
                max: (1 << tree_depth) - 1,
            });
        }

        let mut leaf_digest = self.compute_leaf_digest(values, proof)?;
        for branch_node in proof.read_vec(tree_depth - layer_depth)? {
            leaf_digest = self.compression.compress(if index & 1 == 0 {
                [leaf_digest, branch_node]
            } else {
                [branch_node, leaf_digest]
            });
            index >>= 1;
        }

        (leaf_digest == layer_digests[index])
            .then_some(())
            .ok_or_else(|| VerificationError::InvalidProof.into())
    }
}

/// Compute the Merkle root over a vector of leaf digests.
///
/// Consumes digests because it modifies the vector in place.
///
/// # Preconditions
/// - `digests.len()` is a power of two
fn fold_digests_vector_inplace<C, D>(compression: &C, mut digests: Vec<D>) -> D
where
    C: PseudoCompressionFunction<D, 2>,
    D: Clone + Default + Send + Sync + Debug,
{
    let log_len = log2_strict_usize(digests.len()); // pre-condition
    for layer in (0..log_len).rev() {
        for i in 0..1 << layer {
            digests[i] = compression.compress(array::from_fn(|j| digests[2 * i + j].clone()));
        }
    }
    digests[0].clone()
}
