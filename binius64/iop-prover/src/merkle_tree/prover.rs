// Copyright 2024-2025 Irreducible Inc.

use std::sync::Mutex;

use binius_field::Field;
use binius_hash::{ParallelDigest, ParallelPseudoCompression};
use binius_iop::merkle_tree::{BinaryMerkleTreeScheme, Commitment, Error, MerkleTreeScheme};
use binius_transcript::{BufMut, TranscriptWriter};
use binius_utils::rayon::iter::IndexedParallelIterator;
use digest::{FixedOutputReset, Output, core_api::BlockSizeUser};
use getset::Getters;
use rand::{CryptoRng, SeedableRng, rngs::StdRng};

use super::MerkleTreeProver;
use crate::merkle_tree::binary_merkle_tree::{self, BinaryMerkleTree};

#[derive(Debug, Getters)]
pub struct BinaryMerkleTreeProver<T, H: ParallelDigest, C>
where
    C: ParallelPseudoCompression<Output<H::Digest>, 2>,
{
    #[getset(get = "pub")]
    scheme: BinaryMerkleTreeScheme<T, H::Digest, C::Compression>,
    parallel_compression: C,
    salt_rng: Mutex<StdRng>,
}

impl<T, C, H: ParallelDigest> BinaryMerkleTreeProver<T, H, C>
where
    C: ParallelPseudoCompression<Output<H::Digest>, 2>,
    C::Compression: Clone,
{
    pub fn new(parallel_compression: C) -> Self {
        Self {
            scheme: BinaryMerkleTreeScheme::new(parallel_compression.compression().clone()),
            parallel_compression,
            // We can construct a dummy Rng with a deterministic seed because it will be unused.
            salt_rng: Mutex::new(StdRng::seed_from_u64(0)),
        }
    }

    pub fn hiding(parallel_compression: C, mut rng: impl CryptoRng, salt_len: usize) -> Self {
        Self {
            scheme: BinaryMerkleTreeScheme::hiding(
                parallel_compression.compression().clone(),
                salt_len,
            ),
            parallel_compression,
            salt_rng: Mutex::new(StdRng::from_rng(&mut rng)),
        }
    }
}

impl<F, H, C> MerkleTreeProver<F> for BinaryMerkleTreeProver<F, H, C>
where
    F: Field,
    H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
    C: ParallelPseudoCompression<Output<H::Digest>, 2>,
{
    type Scheme = BinaryMerkleTreeScheme<F, H::Digest, C::Compression>;
    type Committed = BinaryMerkleTree<Output<H::Digest>, F>;

    fn scheme(&self) -> &Self::Scheme {
        &self.scheme
    }

    fn layer<'a>(
        &self,
        committed: &'a Self::Committed,
        depth: usize,
    ) -> Result<&'a [Output<H::Digest>], Error> {
        committed.layer(depth)
    }

    fn prove_opening<B: BufMut>(
        &self,
        committed: &Self::Committed,
        layer_depth: usize,
        index: usize,
        proof: &mut TranscriptWriter<B>,
    ) -> Result<(), Error> {
        let salt = committed.get_salt(index >> layer_depth);
        proof.write_slice(salt);

        let branch = committed.branch(index, layer_depth)?;
        proof.write_slice(&branch);
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn commit_iterated<ParIter>(
        &self,
        leaves: ParIter,
    ) -> Result<
        (
            Commitment<<Self::Scheme as MerkleTreeScheme<F>>::Digest>,
            Self::Committed,
        ),
        Error,
    >
    where
        ParIter: IndexedParallelIterator<Item: IntoIterator<Item = F, IntoIter: Send>>,
    {
        let salt_rng = {
            // If mutex is poisoned, panic.
            let mut root_rng = self.salt_rng.lock().unwrap();
            StdRng::from_rng(&mut *root_rng)
        };
        let tree = binary_merkle_tree::build_from_iterator::<F, H, _, _, _>(
            &self.parallel_compression,
            leaves,
            self.scheme.salt_len(),
            salt_rng,
        )?;

        let commitment = Commitment {
            root: tree.root(),
            depth: tree.log_len,
        };

        Ok((commitment, tree))
    }
}
