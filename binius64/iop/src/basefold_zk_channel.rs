// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP verifier channel.
//!
//! This module provides [`BaseFoldZKVerifierChannel`], which implements [`IOPVerifierChannel`]
//! using FRI commitment and ZK BaseFold opening protocols. Unlike [`super::basefold_channel`],
//! this channel always applies zero-knowledge blinding to all oracles.

use binius_field::BinaryField;
use binius_ip::channel::IPVerifierChannel;
use binius_transcript::{
    VerifierTranscript,
    fiat_shamir::{CanSample, Challenger},
};
use binius_utils::DeserializeBytes;

use crate::{
    basefold,
    channel::{Error, IOPVerifierChannel, OracleLinearRelation, OracleSpec},
    fri::FRIParams,
    merkle_tree::MerkleTreeScheme,
};

/// Oracle handle returned by [`BaseFoldZKVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldZKOracle {
    index: usize,
}

/// A verifier channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// This channel always applies zero-knowledge blinding. The FRI parameters must be set up
/// with `log_batch_size = 1` and `log_msg_len = witness_log_len + 1` to account for the mask.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed references
/// - `F`: The binary field type
/// - `MerkleScheme_`: The Merkle tree scheme for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldZKVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F>,
    Challenger_: Challenger,
{
    transcript: &'a mut VerifierTranscript<Challenger_>,
    merkle_scheme: &'a MerkleScheme_,
    oracle_specs: &'a [OracleSpec],
    fri_params: &'a [FRIParams<F>],
    oracle_commitments: Vec<MerkleScheme_::Digest>,
    next_oracle_index: usize,
}

impl<'a, F, MerkleScheme_, Challenger_> BaseFoldZKVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
    Challenger_: Challenger,
{
    /// Creates a new BaseFold ZK verifier channel from precomputed FRI parameters.
    ///
    /// The FRI parameters should already account for ZK (log_batch_size = 1, doubled message
    /// length).
    pub fn from_precomputed(
        transcript: &'a mut VerifierTranscript<Challenger_>,
        merkle_scheme: &'a MerkleScheme_,
        oracle_specs: &'a [OracleSpec],
        fri_params: &'a [FRIParams<F>],
    ) -> Self {
        Self {
            transcript,
            merkle_scheme,
            oracle_specs,
            fri_params,
            oracle_commitments: Vec::new(),
            next_oracle_index: 0,
        }
    }

    /// Returns a reference to the underlying transcript.
    pub fn transcript(&self) -> &VerifierTranscript<Challenger_> {
        self.transcript
    }
}

impl<F, MerkleScheme_, Challenger_> IPVerifierChannel<F>
    for BaseFoldZKVerifierChannel<'_, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
    Challenger_: Challenger,
{
    type Elem = F;

    fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
        self.transcript
            .message()
            .read_scalar()
            .map_err(|_| binius_ip::channel::Error::ProofEmpty)
    }

    fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
        self.transcript
            .message()
            .read_scalar_slice(n)
            .map_err(|_| binius_ip::channel::Error::ProofEmpty)
    }

    fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
        self.transcript
            .message()
            .read()
            .map_err(|_| binius_ip::channel::Error::ProofEmpty)
    }

    fn sample(&mut self) -> F {
        CanSample::sample(&mut self.transcript)
    }

    fn observe_one(&mut self, val: F) -> F {
        self.transcript.observe().write_scalar(val);
        val
    }

    fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
        self.transcript.observe().write_scalar_slice(vals);
        vals.to_vec()
    }

    fn assert_zero(&mut self, val: F) -> Result<(), binius_ip::channel::Error> {
        if val == F::ZERO {
            Ok(())
        } else {
            Err(binius_ip::channel::Error::InvalidAssert)
        }
    }
}

impl<F, MerkleScheme_, Challenger_> IOPVerifierChannel<F>
    for BaseFoldZKVerifierChannel<'_, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
    Challenger_: Challenger,
{
    type Oracle = BaseFoldZKOracle;

    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.oracle_specs[self.next_oracle_index..]
    }

    fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
        assert!(
            !self.remaining_oracle_specs().is_empty(),
            "recv_oracle called but no remaining oracle specs"
        );

        let index = self.next_oracle_index;

        let commitment = self
            .transcript
            .message()
            .read::<MerkleScheme_::Digest>()
            .map_err(|_| Error::ProofEmpty)?;

        self.oracle_commitments.push(commitment);
        self.next_oracle_index += 1;

        Ok(BaseFoldZKOracle { index })
    }

    fn verify_oracle_relations(
        &mut self,
        oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, Self::Elem>>,
    ) -> Result<(), Error> {
        assert!(
            self.remaining_oracle_specs().is_empty(),
            "verify_oracle_relations called but {} oracle specs remaining",
            self.remaining_oracle_specs().len()
        );

        for relation in oracle_relations {
            let index = relation.oracle.index;
            assert!(
                index < self.oracle_commitments.len(),
                "oracle index {index} out of bounds, expected < {}",
                self.oracle_commitments.len()
            );

            let fri_params = &self.fri_params[index];
            let commitment = self.oracle_commitments[index].clone();

            // Always use ZK verification.
            let basefold::ReducedOutput {
                final_fri_value,
                final_sumcheck_value,
                challenges,
            } = basefold::verify_zk(
                fri_params,
                self.merkle_scheme,
                commitment,
                relation.claim,
                self.transcript,
            )?;

            // Strip batch challenge (challenges[0]) and reverse remaining for eval point.
            let mut eval_point: Vec<F> = challenges[1..].to_vec();
            eval_point.reverse();

            let transparent_eval = (relation.transparent)(&eval_point);

            self.assert_zero(final_sumcheck_value - final_fri_value * transparent_eval)?;
        }

        Ok(())
    }
}
