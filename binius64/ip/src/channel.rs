// Copyright 2026 The Binius Developers

//! Channel abstraction for public-coin interactive protocol verifiers.
//!
//! In a public-coin interactive protocol, the verifier's messages consist entirely of random
//! challenges, while the prover sends deterministic messages based on the protocol state.
//! This module provides the [`IPVerifierChannel`] trait that models the verifier's view of such
//! an interaction.
//!
//! The trait abstracts over:
//! - Receiving prover messages (field elements)
//! - Sampling random challenges (which, in the Fiat-Shamir transform, are derived deterministically
//!   from the transcript)
//!
//! This abstraction allows protocol implementations to be generic over the underlying
//! communication mechanism, whether it's an actual interactive channel or a non-interactive
//! transcript using the Fiat-Shamir heuristic.

use std::iter::repeat_with;

use binius_field::{Field, field::FieldOps};
use binius_transcript::{
    VerifierTranscript,
    fiat_shamir::{CanSample, Challenger},
};

/// Channel for receiving prover messages and sampling challenges in a public-coin interactive
/// protocol.
///
/// In a public-coin protocol, the verifier only sends random challenges (no secret information),
/// so the verifier's role is to:
/// 1. Receive field elements from the prover via `recv_*` methods
/// 2. Sample random challenges via `sample`
///
/// When used with a Fiat-Shamir transcript, the challenges are derived deterministically from
/// the transcript state, making the protocol non-interactive.
pub trait IPVerifierChannel<F> {
    /// The element type returned by receive and sample methods.
    type Elem: FieldOps;

    /// Receives a single field element from the prover.
    fn recv_one(&mut self) -> Result<Self::Elem, Error>;

    /// Receives `n` field elements from the prover.
    fn recv_many(&mut self, n: usize) -> Result<Vec<Self::Elem>, Error> {
        repeat_with(|| self.recv_one()).take(n).collect()
    }

    /// Receives a fixed-size array of field elements from the prover.
    fn recv_array<const N: usize>(&mut self) -> Result<[Self::Elem; N], Error>;

    /// Samples a random challenge.
    ///
    /// In a Fiat-Shamir transcript, this derives the challenge deterministically from
    /// the current transcript state.
    fn sample(&mut self) -> Self::Elem;

    /// Samples `n` random challenges.
    fn sample_many(&mut self, n: usize) -> Vec<Self::Elem> {
        repeat_with(|| self.sample()).take(n).collect()
    }

    /// Samples a fixed-size array of random challenges.
    fn sample_array<const N: usize>(&mut self) -> [Self::Elem; N] {
        std::array::from_fn(|_| self.sample())
    }

    /// Observes a single field element, feeding it into the Fiat-Shamir state.
    ///
    /// Returns the element converted to `Self::Elem`.
    fn observe_one(&mut self, val: F) -> Self::Elem;

    /// Observes multiple field elements, feeding them into the Fiat-Shamir state.
    ///
    /// Returns the elements converted to `Vec<Self::Elem>`.
    fn observe_many(&mut self, vals: &[F]) -> Vec<Self::Elem>;

    /// Asserts that a value is zero.
    ///
    /// Returns [`Error::InvalidAssert`] if the value is not zero.
    fn assert_zero(&mut self, val: Self::Elem) -> Result<(), Error>;
}

impl<F, Challenger_> IPVerifierChannel<F> for VerifierTranscript<Challenger_>
where
    F: Field,
    Challenger_: Challenger,
{
    type Elem = F;

    fn recv_one(&mut self) -> Result<F, Error> {
        self.message().read_scalar().map_err(|_| Error::ProofEmpty)
    }

    fn recv_many(&mut self, n: usize) -> Result<Vec<F>, Error> {
        self.message()
            .read_scalar_slice(n)
            .map_err(|_| Error::ProofEmpty)
    }

    fn recv_array<const N: usize>(&mut self) -> Result<[F; N], Error> {
        self.message().read().map_err(|_| Error::ProofEmpty)
    }

    fn sample(&mut self) -> F {
        CanSample::sample(self)
    }

    fn observe_one(&mut self, val: F) -> F {
        self.observe().write_scalar(val);
        val
    }

    fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
        self.observe().write_scalar_slice(vals);
        vals.to_vec()
    }

    fn assert_zero(&mut self, val: F) -> Result<(), Error> {
        if val == F::ZERO {
            Ok(())
        } else {
            Err(Error::InvalidAssert)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("proof is empty")]
    ProofEmpty,
    #[error("invalid assertion: value is not zero")]
    InvalidAssert,
}
