// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::BinaryField;
use binius_math::{
    FieldBuffer,
    multilinear::eq::eq_ind_partial_eval,
    ntt::{AdditiveNTT, NeighborsLastSingleThread, domain_context::GenericOnTheFly},
};
use binius_transcript::{
    TranscriptReader, VerifierTranscript,
    fiat_shamir::{CanSampleBits, Challenger},
};
use binius_utils::DeserializeBytes;
use bytes::Buf;
use itertools::izip;

use super::{
    common::{FRIParams, vcs_optimal_layers_depths_iter},
    error::{Error, VerificationError},
};
use crate::{
    fri::fold::{fold_chunk, fold_interleaved_chunk},
    merkle_tree::MerkleTreeScheme,
};

/// A verifier for the FRI query phase.
///
/// The verifier is instantiated after the folding rounds and is used to test consistency of the
/// round messages and the original purported codeword.
#[derive(Debug)]
pub struct FRIQueryVerifier<'a, F, VCS>
where
    F: BinaryField,
    VCS: MerkleTreeScheme<F>,
{
    vcs: &'a VCS,
    params: &'a FRIParams<F>,
    /// Received commitment to the codeword.
    codeword_commitment: &'a VCS::Digest,
    /// Received commitments to the round messages.
    round_commitments: &'a [VCS::Digest],
    /// The challenges for each round.
    interleave_tensor: FieldBuffer<F>,
    /// The challenges for each round.
    fold_challenges: &'a [F],
}

impl<'a, F, VCS> FRIQueryVerifier<'a, F, VCS>
where
    F: BinaryField,
    VCS: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: &'a FRIParams<F>,
        vcs: &'a VCS,
        codeword_commitment: &'a VCS::Digest,
        round_commitments: &'a [VCS::Digest],
        challenges: &'a [F],
    ) -> Result<Self, Error> {
        if round_commitments.len() != params.n_oracles() {
            return Err(Error::InvalidArgs(format!(
                "got {} round commitments, expected {}",
                round_commitments.len(),
                params.n_oracles(),
            )));
        }

        if challenges.len() != params.n_fold_rounds() {
            return Err(Error::InvalidArgs(format!(
                "got {} folding challenges, expected {}",
                challenges.len(),
                params.n_fold_rounds(),
            )));
        }

        let (interleave_challenges, fold_challenges) = challenges.split_at(params.log_batch_size());

        let interleave_tensor = eq_ind_partial_eval(interleave_challenges);
        Ok(Self {
            params,
            vcs,
            codeword_commitment,
            round_commitments,
            interleave_tensor,
            fold_challenges,
        })
    }

    /// Number of oracles sent during the fold rounds.
    pub fn n_oracles(&self) -> usize {
        self.params.n_oracles()
    }

    pub fn verify<Challenger_>(
        &self,
        transcript: &mut VerifierTranscript<Challenger_>,
    ) -> Result<F, Error>
    where
        Challenger_: Challenger,
    {
        let subspace = self.params.rs_code().subspace();
        let domain_context = GenericOnTheFly::generate_from_subspace(subspace);
        let ntt = NeighborsLastSingleThread::new(domain_context);

        // Verify that the last oracle sent is a codeword.
        let terminate_codeword_len =
            1 << (self.params.n_final_challenges() + self.params.rs_code().log_inv_rate());
        let mut advice = transcript.decommitment();
        let terminate_codeword = advice
            .read_scalar_slice(terminate_codeword_len)
            .map_err(Error::TranscriptError)?;
        let final_value = self.verify_last_oracle(&ntt, &terminate_codeword, &mut advice)?;

        // Verify that the provided layers match the commitments.
        let layers = vcs_optimal_layers_depths_iter(self.params, self.vcs)
            .map(|layer_depth| advice.read_vec(1 << layer_depth))
            .collect::<Result<Vec<_>, _>>()?;
        for (commitment, layer_depth, layer) in izip!(
            iter::once(self.codeword_commitment).chain(self.round_commitments),
            vcs_optimal_layers_depths_iter(self.params, self.vcs),
            &layers
        ) {
            self.vcs.verify_layer(commitment, layer_depth, layer)?;
        }

        // Verify the random openings against the decommitted layers.
        for _ in 0..self.params.n_test_queries() {
            let index = transcript.sample_bits(self.params.index_bits()) as usize;
            self.verify_query(
                index,
                &ntt,
                &terminate_codeword,
                &layers,
                &mut transcript.decommitment(),
            )?
        }

        Ok(final_value)
    }

    /// Verifies that the last oracle sent is a codeword.
    ///
    /// Returns the fully-folded message value.
    pub fn verify_last_oracle<B: Buf>(
        &self,
        ntt: &impl AdditiveNTT<Field = F>,
        terminate_codeword: &[F],
        advice: &mut TranscriptReader<B>,
    ) -> Result<F, Error> {
        let n_final_challenges = self.params.n_final_challenges();
        let terminal_commitment = self
            .round_commitments
            .last()
            .expect("round_commitments is non-empty as an invariant");

        self.vcs.verify_vector(
            terminal_commitment,
            terminate_codeword,
            1 << n_final_challenges,
            advice,
        )?;

        let n_prior_challenges = self.fold_challenges.len() - n_final_challenges;
        let final_challenges = &self.fold_challenges[n_prior_challenges..];

        let mut scratch_buffer = vec![F::default(); 1 << n_final_challenges];
        let repetition_codeword = terminate_codeword
            .chunks(1 << n_final_challenges)
            .enumerate()
            .map(|(i, coset_values)| {
                scratch_buffer.copy_from_slice(coset_values);
                fold_chunk(
                    ntt,
                    n_final_challenges + self.params.rs_code().log_inv_rate(),
                    i,
                    &mut scratch_buffer,
                    final_challenges,
                )
            })
            .collect::<Vec<_>>();

        let final_value = repetition_codeword[0];

        // Check that the fully-folded purported codeword is a repetition codeword.
        if repetition_codeword[1..]
            .iter()
            .any(|&entry| entry != final_value)
        {
            return Err(VerificationError::IncorrectDegree.into());
        }

        Ok(final_value)
    }

    /// Verifies a FRI challenge query.
    ///
    /// A FRI challenge query tests for consistency between all consecutive oracles sent by the
    /// prover. The verifier has full access to the last oracle sent, and this is probabilistically
    /// verified to be a codeword by `Self::verify_last_oracle`.
    ///
    /// ## Arguments
    ///
    /// * `index` - an index into the original codeword domain
    /// * `proof` - a query proof
    pub fn verify_query<B: Buf>(
        &self,
        mut index: usize,
        ntt: &impl AdditiveNTT<Field = F>,
        terminate_codeword: &[F],
        layers: &[Vec<VCS::Digest>],
        advice: &mut TranscriptReader<B>,
    ) -> Result<(), Error> {
        let mut layer_depths_iter = vcs_optimal_layers_depths_iter(self.params, self.vcs);
        let mut layers_iter = layers.iter();

        // Check the first fold round before the main loop. It is special because in the first
        // round we need to fold as an interleaved chunk instead of a regular coset.
        let first_layer_depth = layer_depths_iter
            .next()
            .expect("protocol guarantees at least one commitment opening");
        let first_layer = layers_iter
            .next()
            .expect("protocol guarantees at least one commitment opening");
        let values = verify_coset_opening(
            self.vcs,
            index,
            self.params.log_batch_size(),
            first_layer_depth,
            self.params.index_bits(),
            first_layer,
            advice,
        )?;
        let mut next_value = fold_interleaved_chunk(
            self.params.log_batch_size(),
            &values,
            self.interleave_tensor.as_ref(),
        );

        // This is the round of the folding phase that the codeword to be folded is committed to.
        let mut fold_round = 0;
        let mut log_n_cosets = self.params.index_bits();
        for (i, (&arity, layer, optimal_layer_depth)) in
            izip!(self.params.fold_arities(), layers_iter, layer_depths_iter).enumerate()
        {
            let coset_index = index >> arity;
            log_n_cosets -= arity;

            let mut values = verify_coset_opening(
                self.vcs,
                coset_index,
                arity,
                optimal_layer_depth,
                log_n_cosets,
                layer,
                advice,
            )?;

            if next_value != values[index % (1 << arity)] {
                return Err(VerificationError::IncorrectFold {
                    query_round: i,
                    index,
                }
                .into());
            }

            next_value = fold_chunk(
                ntt,
                self.params.rs_code().log_len() - fold_round,
                coset_index,
                &mut values,
                &self.fold_challenges[fold_round..fold_round + arity],
            );
            index = coset_index;
            fold_round += arity;
        }

        if next_value != terminate_codeword[index] {
            return Err(VerificationError::IncorrectFold {
                query_round: self.n_oracles() - 1,
                index,
            }
            .into());
        }

        Ok(())
    }
}

/// Verifies that the coset opening provided in the proof is consistent with the VCS commitment.
#[allow(clippy::too_many_arguments)]
fn verify_coset_opening<F, MTScheme, B>(
    vcs: &MTScheme,
    coset_index: usize,
    log_coset_size: usize,
    optimal_layer_depth: usize,
    tree_depth: usize,
    layer_digests: &[MTScheme::Digest],
    advice: &mut TranscriptReader<B>,
) -> Result<Vec<F>, Error>
where
    F: BinaryField,
    MTScheme: MerkleTreeScheme<F>,
    B: Buf,
{
    let values = advice.read_scalar_slice::<F>(1 << log_coset_size)?;
    vcs.verify_opening(
        coset_index,
        &values,
        optimal_layer_depth,
        tree_depth,
        layer_digests,
        advice,
    )?;
    Ok(values)
}
