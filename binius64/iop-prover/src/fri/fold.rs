// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, Field, PackedField};
use binius_iop::{
    fri::{FRIParams, fold::fold_chunk},
    merkle_tree::MerkleTreeScheme,
};
use binius_math::{
    FieldBuffer, FieldSlice, inner_product::inner_product_buffers,
    multilinear::eq::eq_ind_partial_eval, ntt::AdditiveNTT,
};
use binius_transcript::{
    ProverTranscript,
    fiat_shamir::{CanSampleBits, Challenger},
};
use binius_utils::{SerializeBytes, checked_arithmetics::log2_strict_usize, rayon::prelude::*};
use tracing::instrument;

use super::{error::Error, query::FRIQueryProver};
use crate::merkle_tree::MerkleTreeProver;

/// The type of the termination round codeword in the FRI protocol.
pub type TerminateCodeword<F> = FieldBuffer<F>;

pub enum FoldRoundOutput<VCSCommitment> {
    NoCommitment,
    Commitment(VCSCommitment),
}

/// A stateful prover for the FRI fold phase.
pub struct FRIFoldProver<'a, F, P, NTT, MerkleProver>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    MerkleProver: MerkleTreeProver<F>,
{
    params: &'a FRIParams<F>,
    ntt: &'a NTT,
    merkle_prover: &'a MerkleProver,
    codeword: FieldBuffer<P>,
    codeword_committed: &'a MerkleProver::Committed,
    round_committed: Vec<(FieldBuffer<F>, MerkleProver::Committed)>,
    curr_round: usize,
    next_commit_round: Option<usize>,
    unprocessed_challenges: Vec<F>,
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver> FRIFoldProver<'a, F, P, NTT, MerkleProver>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
    MerkleProver: MerkleTreeProver<F, Scheme = MerkleScheme>,
{
    /// Constructs a new folder.
    pub fn new(
        params: &'a FRIParams<F>,
        ntt: &'a NTT,
        merkle_prover: &'a MerkleProver,
        committed_codeword: FieldBuffer<P>,
        committed: &'a MerkleProver::Committed,
    ) -> Result<Self, Error> {
        if committed_codeword.len() < 1 << params.log_len() {
            return Err(Error::InvalidArgs(
                "Reed-Solomon code length must match interleaved codeword length".to_string(),
            ));
        }

        let next_commit_round = Some(params.log_batch_size());
        Ok(Self {
            params,
            ntt,
            merkle_prover,
            codeword: committed_codeword,
            codeword_committed: committed,
            round_committed: Vec::with_capacity(params.n_oracles()),
            curr_round: 0,
            next_commit_round,
            unprocessed_challenges: Vec::with_capacity(params.rs_code().log_dim()),
        })
    }

    /// Number of fold rounds, including the final fold.
    pub fn n_rounds(&self) -> usize {
        self.params.n_fold_rounds()
    }

    /// Number of times `execute_fold_round` has been called.
    pub const fn curr_round(&self) -> usize {
        self.curr_round
    }

    /// The length of the current codeword.
    pub fn current_codeword_len(&self) -> usize {
        match self.round_committed.last() {
            Some((codeword, _)) => codeword.len(),
            None => self.codeword.len(),
        }
    }

    fn is_commitment_round(&self) -> bool {
        self.next_commit_round
            .is_some_and(|round| round == self.curr_round)
    }

    pub fn receive_challenge(&mut self, challenge: F) {
        self.unprocessed_challenges.push(challenge);
        self.curr_round += 1;
    }

    /// Executes the next fold round and returns the folded codeword commitment.
    ///
    /// As a memory efficient optimization, this method may not actually do the folding, but instead
    /// accumulate the folding challenge for processing at a later time. This saves us from storing
    /// intermediate folded codewords.
    pub fn execute_fold_round(&mut self) -> Result<FoldRoundOutput<MerkleScheme::Digest>, Error> {
        if !self.is_commitment_round() {
            return Ok(FoldRoundOutput::NoCommitment);
        }

        let _fri_round_scope = tracing::debug_span!(
            "FRI Round",
            log_len = log2_strict_usize(self.current_codeword_len()),
            arity = self.unprocessed_challenges.len()
        )
        .entered();

        // Fold the last codeword with the accumulated folding challenges.
        let fri_fold_span = tracing::debug_span!("FRI Fold").entered();
        let folded_codeword = match self.round_committed.last() {
            Some((prev_codeword, _)) => {
                // Fold a full codeword committed in the previous FRI round into a codeword with
                // reduced dimension and rate.
                fold_codeword(
                    self.ntt,
                    prev_codeword.to_ref(),
                    &self.unprocessed_challenges,
                )
            }
            None => {
                // Fold the interleaved codeword that was originally committed into a single
                // codeword with the same block length.
                fold_interleaved(
                    self.codeword.to_ref(),
                    &self.unprocessed_challenges,
                    self.params.rs_code().log_len(),
                    self.params.log_batch_size(),
                )
            }
        };
        drop(fri_fold_span);
        self.unprocessed_challenges.clear();

        // take the first arity as coset_log_len, or use inv_rate if arities are empty
        let next_arity = self
            .params
            .fold_arities()
            .get(self.round_committed.len())
            .copied();
        let log_coset_size = next_arity.unwrap_or_else(|| self.params.n_final_challenges());
        let coset_size = 1 << log_coset_size;
        let merkle_tree_span = tracing::debug_span!("Merkle Tree").entered();
        let (commitment, committed) = self
            .merkle_prover
            .commit(folded_codeword.as_ref(), coset_size)?;
        drop(merkle_tree_span);

        self.next_commit_round = self.next_commit_round.take().and_then(|next_commit_round| {
            let arity = next_arity?;
            Some(next_commit_round + arity)
        });

        self.round_committed.push((folded_codeword, committed));
        Ok(FoldRoundOutput::Commitment(commitment.root))
    }

    /// Finalizes the FRI folding process.
    ///
    /// This step will process any unprocessed folding challenges to produce the
    /// final folded codeword. Then it will decode this final folded codeword
    /// to get the final message. The result is the final message and a query prover instance.
    ///
    /// This returns the final message and a query prover instance.
    #[instrument(skip_all, name = "fri::FRIFolder::finalize", level = "debug")]
    #[allow(clippy::type_complexity)]
    pub fn finalize(
        mut self,
    ) -> Result<
        (
            TerminateCodeword<F>,
            FRIQueryProver<'a, F, P, MerkleProver, MerkleScheme>,
        ),
        Error,
    > {
        if self.curr_round != self.n_rounds() {
            return Err(Error::EarlyProverFinish);
        }

        let terminate_codeword = self
            .round_committed
            .last()
            .map(|(codeword, _)| codeword.clone())
            .unwrap_or_else(|| {
                let scalars: Vec<F> = self.codeword.iter_scalars().collect();
                FieldBuffer::from_values(&scalars)
            });

        self.unprocessed_challenges.clear();

        let Self {
            params,
            codeword,
            codeword_committed,
            round_committed,
            merkle_prover,
            ..
        } = self;

        let query_prover = FRIQueryProver {
            params,
            codeword,
            codeword_committed,
            round_committed,
            merkle_prover,
        };
        Ok((terminate_codeword, query_prover))
    }

    pub fn finish_proof<Challenger_>(
        self,
        transcript: &mut ProverTranscript<Challenger_>,
    ) -> Result<(), Error>
    where
        Challenger_: Challenger,
    {
        let (terminate_codeword, query_prover) = self.finalize()?;
        let mut advice = transcript.decommitment();
        advice.write_scalar_slice(terminate_codeword.as_ref());

        let layers = query_prover.vcs_optimal_layers()?;
        for layer in layers {
            advice.write_slice(&layer);
        }

        let params = query_prover.params;
        for _ in 0..params.n_test_queries() {
            let index = transcript.sample_bits(params.index_bits()) as usize;
            query_prover.prove_query(index, &mut transcript.decommitment())?;
        }

        Ok(())
    }
}

/// Fold the interleaved codeword into a single codeword using the given challenges.
///
/// ## Arguments
///
/// * `codeword` - an interleaved codeword.
/// * `challenges` - the folding challenges. The length must be at least `log_batch_size`.
/// * `log_len` - the binary logarithm of the code length.
/// * `log_batch_size` - the binary logarithm of the interleaved code batch size.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[instrument(skip_all, level = "debug")]
fn fold_interleaved<F, P>(
    codeword: FieldSlice<P>,
    challenges: &[F],
    log_len: usize,
    log_batch_size: usize,
) -> FieldBuffer<F>
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    assert_eq!(codeword.log_len(), log_len + log_batch_size);
    assert_eq!(challenges.len(), log_batch_size);

    let tensor = eq_ind_partial_eval(challenges);

    // For each chunk of size `2^chunk_size` in the interleaved codeword, fold it with the folding
    // challenges.
    let values = codeword
        .chunks_par(log_batch_size)
        .map(|chunk| inner_product_buffers(&chunk, &tensor))
        .collect::<Vec<_>>();
    FieldBuffer::new(log_len, values.into_boxed_slice())
}

/// FRI-fold the codeword using the given challenges.
///
/// ## Arguments
///
/// * `ntt` - the NTT instance, used to look up the twiddle values.
/// * `codeword` - an interleaved codeword.
/// * `challenges` - the folding challenges. The length must be at least `log_batch_size`.
/// * `log_len` - the binary logarithm of the code length.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[instrument(skip_all, level = "debug")]
fn fold_codeword<F, NTT>(ntt: &NTT, codeword: FieldSlice<F>, challenges: &[F]) -> FieldBuffer<F>
where
    F: BinaryField,
    NTT: AdditiveNTT<Field = F> + Sync,
{
    let log_len = codeword.log_len();
    assert!(challenges.len() <= log_len);

    let folded_log_len = log_len - challenges.len();

    // For each coset of size `2^chunk_size` in the codeword, fold it with the folding challenges.
    let chunk_size = 1 << challenges.len();
    let values: Vec<F> = codeword
        .chunks_par(challenges.len())
        .enumerate()
        .map_init(
            || vec![F::default(); chunk_size],
            |scratch_buffer, (i, chunk)| {
                scratch_buffer.copy_from_slice(chunk.as_ref());
                fold_chunk(ntt, log_len, i, scratch_buffer, challenges)
            },
        )
        .collect();
    FieldBuffer::new(folded_log_len, values.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use binius_field::BinaryField128bGhash as B128;
    use binius_math::{
        BinarySubspace,
        fold::fold_cols,
        ntt::{NeighborsLastReference, domain_context::GenericOnTheFly},
        test_utils::{random_field_buffer, random_scalars},
    };
    use proptest::prelude::*;
    use rand::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn test_fri_compatible_ntt_domains(log_dim in 0..8usize, arity in 0..4usize) {
            test_help_fri_compatible_ntt_domains(log_dim, arity);
        }
    }

    fn test_help_fri_compatible_ntt_domains(log_dim: usize, arity: usize) {
        let subspace = BinarySubspace::with_dim(32);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
        let ntt = NeighborsLastReference { domain_context };

        let mut rng = StdRng::seed_from_u64(0);
        let msg = random_field_buffer(&mut rng, log_dim + arity);
        let challenges = random_scalars(&mut rng, arity);

        let query = eq_ind_partial_eval::<B128>(&challenges);

        // Fold the message using regular folding.
        let mut folded_msg = fold_cols(&msg, &query);
        assert_eq!(folded_msg.log_len(), log_dim);

        // Encode the message over the large domain.
        let mut codeword = msg;
        ntt.forward_transform(codeword.to_mut(), 0, 0);

        // Fold the encoded message using FRI folding.
        let folded_codeword = fold_codeword(&ntt, codeword.to_ref(), &challenges);

        // Encode the folded message.
        ntt.forward_transform(folded_msg.to_mut(), 0, 0);

        // Check that folding and encoding commute.
        assert_eq!(folded_codeword, folded_msg);
    }
}
