// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) provers.
//!
//! An IOP extends the public-coin interactive protocol model with oracle access: the prover can
//! commit to oracles (e.g., Merkle trees) that the verifier can query at specific positions. This
//! module provides the [`IOPProverChannel`] trait that models the prover's view of such an
//! interaction.
//!
//! The trait extends [`IPProverChannel`] with additional methods for:
//! - Committing oracles to the verifier
//! - Responding to oracle queries with opening proofs
//!
//! This abstraction allows protocol implementations to be generic over the underlying
//! communication and commitment mechanisms.

use binius_field::PackedField;
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice};

/// Channel for IOP provers that extends the IP prover channel with oracle operations.
///
/// In an IOP, the prover can:
/// 1. Send field elements to the verifier via `send_*` methods (inherited)
/// 2. Sample random challenges via `sample` (inherited)
/// 3. Commit oracles to the verifier
/// 4. Respond to oracle queries with opening proofs
///
/// # Contract
///
/// The caller must call `send_oracle()` exactly `remaining_oracle_specs().len()` times before
/// calling `prove_oracle_relations()`. Each oracle buffer must match the corresponding
/// specification.
pub trait IOPProverChannel<P: PackedField>: IPProverChannel<P::Scalar> {
    type Oracle: Clone;

    /// Returns the specifications for the remaining oracles to be committed.
    ///
    /// This slice shrinks as oracles are committed via `send_oracle()`.
    fn remaining_oracle_specs(&self) -> &[OracleSpec];

    /// Commits an oracle to the verifier.
    ///
    /// # Preconditions
    ///
    /// * `remaining_oracle_specs()` must be non-empty.
    /// * `buffer.log_len()` must match the expected length from the next oracle spec.
    fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle;

    /// Generates opening proofs for all oracle linear relations.
    ///
    /// # Preconditions
    ///
    /// * `remaining_oracle_specs()` must be empty (all oracles committed).
    /// * All oracle handles in `oracle_relations` must be valid handles returned by
    ///   `send_oracle()`.
    fn prove_oracle_relations(
        &mut self,
        oracle_relations: impl IntoIterator<Item = (Self::Oracle, FieldBuffer<P>, P::Scalar)>,
    );
}
