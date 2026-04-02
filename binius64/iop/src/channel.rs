// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) verifiers.
//!
//! An IOP extends the public-coin interactive protocol model with oracle access: the prover can
//! commit to oracles (e.g., Merkle trees) that the verifier can query at specific positions. This
//! module provides the [`IOPVerifierChannel`] trait that models the verifier's view of such an
//! interaction.
//!
//! The trait extends [`IPVerifierChannel`] with additional methods for:
//! - Receiving oracle commitments from the prover
//! - Querying oracle positions and receiving opening proofs
//!
//! This abstraction allows protocol implementations to be generic over the underlying
//! communication and commitment mechanisms.

use binius_field::Field;
use binius_ip::channel::IPVerifierChannel;

use crate::basefold;

/// Error type for IOP verifier channel operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("proof is empty")]
    ProofEmpty,
    #[error("BaseFold verification failed: {0}")]
    BaseFold(#[from] basefold::Error),
    #[error("IP channel error: {0}")]
    IPChannel(#[from] binius_ip::channel::Error),
}

/// Specification for an oracle to be committed in the IOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSpec {
    /// Log2 of the message length (number of field elements).
    pub log_msg_len: usize,
}

/// A boxed closure that evaluates a transparent MLE at a given point.
pub type TransparentEvalFn<Elem> = Box<dyn Fn(&[Elem]) -> Elem>;

/// An oracle linear relation specifying an inner product claim between a committed oracle
/// polynomial and a transparent polynomial.
///
/// The claim asserts that `<oracle_poly, transparent_poly> = claim`, where `transparent_poly` is
/// the multilinear extension defined by the `transparent` closure evaluated at the challenge point
/// sampled during the protocol.
pub struct OracleLinearRelation<Oracle, Elem> {
    /// The oracle handle for the committed polynomial.
    pub oracle: Oracle,
    /// A closure that evaluates the transparent MLE at a given point.
    ///
    /// The closure receives the challenge point (sampled during `verify_oracle_relations`) and
    /// returns the evaluation of the transparent polynomial's MLE at that point.
    pub transparent: TransparentEvalFn<Elem>,
    /// The claimed inner product of the oracle polynomial and the transparent polynomial.
    pub claim: Elem,
}

/// Channel for IOP verifiers that extends the IP verifier channel with oracle operations.
///
/// In an IOP, the verifier can:
/// 1. Receive field elements from the prover via `recv_*` methods (inherited)
/// 2. Sample random challenges via `sample` (inherited)
/// 3. Receive oracle commitments from the prover
/// 4. Query oracles at specific positions and verify opening proofs
///
/// # Contract
///
/// The caller must call `recv_oracle()` exactly `remaining_oracle_specs().len()` times before
/// calling `verify_oracle_relations()`. The oracles must be received in order and match their
/// specifications.
pub trait IOPVerifierChannel<F: Field>: IPVerifierChannel<F> {
    type Oracle: Clone;

    /// Returns the specifications for the remaining oracles to be received.
    ///
    /// This slice shrinks as oracles are received via `recv_oracle()`.
    fn remaining_oracle_specs(&self) -> &[OracleSpec];

    /// Receives an oracle commitment from the prover.
    ///
    /// # Preconditions
    ///
    /// `remaining_oracle_specs()` must be non-empty.
    fn recv_oracle(&mut self) -> Result<Self::Oracle, Error>;

    /// Verifies all oracle linear relations by running opening protocols.
    ///
    /// For each oracle relation, this method:
    /// 1. Runs the opening protocol (e.g., BaseFold) to obtain the oracle evaluation and challenge
    ///    point
    /// 2. Evaluates the transparent polynomial at the challenge point
    /// 3. Verifies that `eval_numerator == eval_denominator * transparent_eval` using `assert_zero`
    ///    on the underlying channel
    ///
    /// # Preconditions
    ///
    /// * `remaining_oracle_specs()` must be empty (all oracles received).
    /// * All oracle handles in `oracle_relations` must be valid handles returned by
    ///   `recv_oracle()`.
    fn verify_oracle_relations(
        &mut self,
        oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, Self::Elem>>,
    ) -> Result<(), Error>;
}
