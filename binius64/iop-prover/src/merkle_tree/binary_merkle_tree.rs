// Copyright 2024-2025 Irreducible Inc.

use std::{fmt::Debug, mem::MaybeUninit};

use binius_field::Field;
use binius_hash::{ParallelDigest, ParallelPseudoCompression};
use binius_iop::merkle_tree::Error;
use binius_utils::{
    checked_arithmetics::log2_strict_usize,
    mem::slice_assume_init_mut,
    rand::par_rand,
    rayon::{prelude::*, slice::ParallelSlice},
};
use digest::{FixedOutputReset, Output, crypto_common::BlockSizeUser};
use rand::{CryptoRng, Rng, rngs::StdRng};

/// A binary Merkle tree that commits batches of vectors.
///
/// The vector entries at each index in a batch are hashed together into leaf digests. Then a
/// Merkle tree is constructed over the leaf digests. The implementation requires that the vector
/// lengths are all equal to each other and a power of two.
#[derive(Debug, Clone)]
pub struct BinaryMerkleTree<D, F> {
    /// Base-2 logarithm of the number of leaves
    pub log_len: usize,
    /// The inner nodes, arranged as a flattened array of layers with the root at the end
    pub inner_nodes: Vec<D>,
    /// Salt values for each leaf (if using hiding commitments)
    pub salts: Vec<F>,
}

pub fn build<F, H, C, R>(
    compression: &C,
    elements: &[F],
    batch_size: usize,
    salt_len: usize,
    rng: R,
) -> Result<BinaryMerkleTree<Output<H::Digest>, F>, Error>
where
    F: Field,
    H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
    C: ParallelPseudoCompression<Output<H::Digest>, 2>,
    R: Rng + CryptoRng,
{
    if !elements.len().is_multiple_of(batch_size) {
        return Err(Error::IncorrectBatchSize);
    }

    let len = elements.len() / batch_size;

    if !len.is_power_of_two() {
        return Err(Error::PowerOfTwoLengthRequired);
    }

    build_from_iterator::<_, H, _, _, _>(
        compression,
        elements
            .par_chunks(batch_size)
            .map(|chunk| chunk.iter().copied()),
        salt_len,
        rng,
    )
}

pub fn build_from_iterator<F, H, C, R, ParIter>(
    compression: &C,
    iterated_chunks: ParIter,
    salt_len: usize,
    mut rng: R,
) -> Result<BinaryMerkleTree<Output<H::Digest>, F>, Error>
where
    F: Field,
    H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
    C: ParallelPseudoCompression<Output<H::Digest>, 2>,
    R: Rng + CryptoRng,
    ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
{
    let log_len = log2_strict_usize(iterated_chunks.len()); // precondition

    // Generate salts if needed
    let salts =
        par_rand::<StdRng, _, _>(salt_len << log_len, &mut rng, F::random).collect::<Vec<_>>();

    let total_length = (1 << (log_len + 1)) - 1;
    let mut inner_nodes = Vec::with_capacity(total_length);
    hash_leaves::<F, H, _>(
        iterated_chunks,
        &mut inner_nodes.spare_capacity_mut()[..(1 << log_len)],
        &salts,
    );

    let (prev_layer, mut remaining) = inner_nodes.spare_capacity_mut().split_at_mut(1 << log_len);

    let mut prev_layer = unsafe {
        // SAFETY: prev-layer was initialized by hash_leaves
        slice_assume_init_mut(prev_layer)
    };
    for i in 1..(log_len + 1) {
        let (next_layer, next_remaining) = remaining.split_at_mut(1 << (log_len - i));
        remaining = next_remaining;

        compression.parallel_compress(prev_layer, next_layer);

        prev_layer = unsafe {
            // SAFETY: next_layer was just initialized by compress_layer
            slice_assume_init_mut(next_layer)
        };
    }

    unsafe {
        // SAFETY: inner_nodes should be entirely initialized by now
        // Note that we don't incrementally update inner_nodes.len() since
        // that doesn't play well with using split_at_mut on spare capacity.
        inner_nodes.set_len(total_length);
    }
    Ok(BinaryMerkleTree {
        log_len,
        inner_nodes,
        salts,
    })
}

impl<D: Clone, F> BinaryMerkleTree<D, F> {
    pub fn root(&self) -> D {
        self.inner_nodes
            .last()
            .expect("MerkleTree inner nodes can't be empty")
            .clone()
    }

    /// Returns the salt values associated with a specific leaf index in the Merkle tree.
    ///
    /// # Arguments
    /// * `index` - The index of the leaf. Must be less than 2^log_len (the total number of leaves).
    pub fn get_salt(&self, index: usize) -> &[F] {
        assert!(index < (1 << self.log_len));
        let salt_len = self.salts.len() >> self.log_len;
        &self.salts[index * salt_len..(index + 1) * salt_len]
    }

    pub fn layer(&self, layer_depth: usize) -> Result<&[D], Error> {
        if layer_depth > self.log_len {
            return Err(Error::IncorrectLayerDepth);
        }
        let range_start = self.inner_nodes.len() + 1 - (1 << (layer_depth + 1));

        Ok(&self.inner_nodes[range_start..range_start + (1 << layer_depth)])
    }

    /// Get a Merkle branch for the given index
    ///
    /// Throws if the index is out of range
    pub fn branch(&self, index: usize, layer_depth: usize) -> Result<Vec<D>, Error> {
        if index >= 1 << self.log_len || layer_depth > self.log_len {
            return Err(Error::IndexOutOfRange {
                max: (1 << self.log_len) - 1,
            });
        }

        let branch = (0..self.log_len - layer_depth)
            .map(|j| {
                let node_index = (((1 << j) - 1) << (self.log_len + 1 - j)) | (index >> j) ^ 1;
                self.inner_nodes[node_index].clone()
            })
            .collect();

        Ok(branch)
    }
}

/// Hashes the elements in chunks of a vector into digests.
///
/// Given a vector of elements and an output buffer of N hash digests, this splits the elements
/// into N equal-sized chunks and hashes each chunks into the corresponding output digest. This
/// returns the number of elements hashed into each digest.
#[tracing::instrument("hash_leaves", skip_all, level = "debug")]
fn hash_leaves<F, H, ParIter>(
    iterated_chunks: ParIter,
    digests: &mut [MaybeUninit<Output<H::Digest>>],
    salts: &[F],
) where
    F: Field,
    H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
    ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
{
    if salts.is_empty() {
        // Need special-case handling when salts is empty, otherwise salt_len is 0 and par_chunks
        // cannot handle chunk size of 0.
        let hasher = H::new();
        hasher.digest(iterated_chunks, digests);
    } else {
        assert!(salts.len().is_multiple_of(digests.len()));

        let salt_len = salts.len() / digests.len();

        // Create an iterator that chains each chunk with its salt
        let salted_iter = iterated_chunks
            .zip(salts.par_chunks(salt_len))
            .map(|(chunk, salt)| chunk.into_iter().chain(salt.iter().copied()));

        let hasher = H::new();
        hasher.digest(salted_iter, digests);
    }
}
