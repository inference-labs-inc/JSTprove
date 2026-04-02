// Copyright 2026 The Binius Developers

//! BaseFold-based implementation of the IOP verifier channel.
//!
//! This module provides [`BaseFoldVerifierChannel`], which implements [`IOPVerifierChannel`] using
//! FRI commitment and BaseFold opening protocols.

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

/// Oracle handle returned by [`BaseFoldVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
    index: usize,
}

/// A verifier channel that uses BaseFold for oracle commitment and opening.
///
/// This channel wraps a [`VerifierTranscript`] and provides oracle operations using
/// FRI commitment (Reed-Solomon encoding + Merkle tree) and BaseFold opening protocols.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed references
/// - `F`: The binary field type
/// - `MerkleScheme_`: The Merkle tree scheme for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F>,
    Challenger_: Challenger,
{
    /// Verifier transcript for Fiat-Shamir (borrowed).
    transcript: &'a mut VerifierTranscript<Challenger_>,
    /// Merkle tree scheme (borrowed).
    merkle_scheme: &'a MerkleScheme_,
    /// Oracle specifications (borrowed).
    oracle_specs: &'a [OracleSpec],
    /// Precomputed FRI params per oracle (borrowed).
    fri_params: &'a [FRIParams<F>],
    /// Received oracle commitments.
    oracle_commitments: Vec<MerkleScheme_::Digest>,
    /// Next oracle index.
    next_oracle_index: usize,
}

impl<'a, F, MerkleScheme_, Challenger_> BaseFoldVerifierChannel<'a, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
    Challenger_: Challenger,
{
    /// Creates a new BaseFold verifier channel from precomputed FRI parameters.
    ///
    /// This constructor is useful when FRI parameters have already been computed
    /// (e.g., by a [`crate::basefold_compiler::BaseFoldVerifierCompiler`]).
    ///
    /// # Arguments
    ///
    /// * `transcript` - The verifier transcript for Fiat-Shamir (borrowed)
    /// * `merkle_scheme` - The Merkle tree scheme (borrowed)
    /// * `oracle_specs` - Specifications for each oracle to be committed (borrowed)
    /// * `fri_params` - Precomputed FRI parameters for each oracle (borrowed)
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
    for BaseFoldVerifierChannel<'_, F, MerkleScheme_, Challenger_>
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
    for BaseFoldVerifierChannel<'_, F, MerkleScheme_, Challenger_>
where
    F: BinaryField,
    MerkleScheme_: MerkleTreeScheme<F, Digest: DeserializeBytes>,
    Challenger_: Challenger,
{
    type Oracle = BaseFoldOracle;

    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.oracle_specs[self.next_oracle_index..]
    }

    fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
        assert!(
            !self.remaining_oracle_specs().is_empty(),
            "recv_oracle called but no remaining oracle specs"
        );

        let index = self.next_oracle_index;

        // Read commitment from transcript
        let commitment = self
            .transcript
            .message()
            .read::<MerkleScheme_::Digest>()
            .map_err(|_| Error::ProofEmpty)?;

        self.oracle_commitments.push(commitment);
        self.next_oracle_index += 1;

        Ok(BaseFoldOracle { index })
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

        // Process each oracle relation with its own BaseFold verification
        for relation in oracle_relations {
            let index = relation.oracle.index;
            assert!(
                index < self.oracle_commitments.len(),
                "oracle index {index} out of bounds, expected < {}",
                self.oracle_commitments.len()
            );

            let fri_params = &self.fri_params[index];
            let commitment = self.oracle_commitments[index].clone();

            // Run BaseFold verification (non-ZK variant).
            let basefold::ReducedOutput {
                final_fri_value,
                final_sumcheck_value,
                challenges,
            } = basefold::verify(
                fri_params,
                self.merkle_scheme,
                commitment,
                relation.claim,
                self.transcript,
            )?;

            // Reverse challenges to get evaluation point in correct order (low-to-high)
            let mut eval_point = challenges;
            eval_point.reverse();

            // Evaluate the transparent polynomial at the challenge point
            let transparent_eval = (relation.transparent)(&eval_point);

            // Verify: eval_numerator == eval_denominator * transparent_eval
            self.assert_zero(final_sumcheck_value - final_fri_value * transparent_eval)?;
        }

        Ok(())
    }
}
