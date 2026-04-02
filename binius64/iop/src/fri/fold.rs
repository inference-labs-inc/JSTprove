// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, ExtensionField, Field, PackedField};
use binius_math::{
    inner_product::inner_product_packed, line::extrapolate_line_packed, ntt::AdditiveNTT,
};
use binius_transcript::TranscriptReader;
use binius_utils::DeserializeBytes;
use bytes::Buf;

use super::{FRIParams, error::Error};

/// Calculate fold of `values` at `index` with `r` random coefficient.
///
/// See [DP24], Def. 3.6.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
fn fold_pair<F, FS, NTT>(ntt: &NTT, round: usize, index: usize, values: (F, F), r: F) -> F
where
    F: BinaryField + ExtensionField<FS>,
    FS: BinaryField,
    NTT: AdditiveNTT<Field = FS>,
{
    // Perform inverse additive NTT butterfly
    let t = ntt.twiddle(round - 1, index);
    let (mut u, mut v) = values;
    v += u;
    u += v * t;
    extrapolate_line_packed(u, v, r)
}

/// Calculate FRI fold of `values` at a `chunk_index` with random folding challenges.
///
/// Folds a coset of a Reed–Solomon codeword into a single value using the FRI folding algorithm.
/// The coset has size $2^n$, where $n$ is the number of challenges.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// NB: This method is on a hot path and does not perform any allocations or
/// precondition checks.
///
/// ## Arguments
///
/// * `math` - the NTT instance, used to look up the twiddle values.
/// * `log_len` - the binary logarithm of the code length.
/// * `chunk_index` - the index of the chunk, of size $2^n$, in the full codeword.
/// * `values` - mutable slice of values to fold, modified in place.
/// * `challenges` - the sequence of folding challenges, with length $n$.
///
/// ## Pre-conditions
///
/// - `challenges.len() <= log_len`.
/// - `log_len <= math.log_domain_size()`, so that the NTT domain is large enough.
/// - `values.len() == 1 << challenges.len()`.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
pub fn fold_chunk<F, FS, NTT>(
    ntt: &NTT,
    mut log_len: usize,
    chunk_index: usize,
    values: &mut [F],
    challenges: &[F],
) -> F
where
    F: BinaryField + ExtensionField<FS>,
    FS: BinaryField,
    NTT: AdditiveNTT<Field = FS>,
{
    let mut log_size = challenges.len();

    // Preconditions
    debug_assert!(log_size <= log_len);
    debug_assert!(log_len <= ntt.log_domain_size());
    debug_assert_eq!(values.len(), 1 << log_size);

    // FRI-fold the values in place.
    for &challenge in challenges {
        // Fold the (2i) and (2i+1)th cells of the scratch buffer in-place into the i-th cell
        for index_offset in 0..1 << (log_size - 1) {
            let pair = (values[index_offset << 1], values[(index_offset << 1) | 1]);
            values[index_offset] = fold_pair(
                ntt,
                log_len,
                (chunk_index << (log_size - 1)) | index_offset,
                pair,
                challenge,
            )
        }

        log_len -= 1;
        log_size -= 1;
    }

    values[0]
}

/// Calculate the fold of an interleaved chunk of values with random folding challenges.
///
/// The elements in the `values` vector are the interleaved cosets of a batch of codewords at the
/// index `coset_index`. That is, the layout of elements in the values slice is
///
/// ```text
/// [a0, b0, c0, d0, a1, b1, c1, d1, ...]
/// ```
///
/// where `a0, a1, ...` form a coset of a codeword `a`, `b0, b1, ...` form a coset of a codeword
/// `b`, and similarly for `c` and `d`.
///
/// The fold operation first folds the adjacent symbols in the slice using regular multilinear
/// tensor folding for the symbols from different cosets and FRI folding for the cosets themselves
/// using the remaining challenges.
//
/// NB: This method is on a hot path and does not perform any allocations or
/// precondition checks.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
pub fn fold_interleaved_chunk<F, P>(log_batch_size: usize, values: &[P], tensor: &[P]) -> F
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    inner_product_packed(
        log_batch_size,
        values.iter().copied(),
        tensor.iter().copied(),
    )
}

/// A stateful verifier for the FRI fold phase that tracks when to read commitments.
///
/// This verifier encapsulates the logic of determining which FRI rounds require
/// commitments and handles reading them from the transcript at the appropriate times.
pub struct FRIFoldVerifier<'a, F, Digest>
where
    F: BinaryField,
{
    /// Indicates which rounds require reading a commitment
    commit_rounds: Vec<bool>,
    /// The round commitments read from the transcript
    round_commitments: Vec<Digest>,
    /// Current round number
    curr_round: usize,
    _phantom: std::marker::PhantomData<&'a F>,
}

impl<'a, F, Digest> FRIFoldVerifier<'a, F, Digest>
where
    F: BinaryField,
    Digest: DeserializeBytes + Clone,
{
    /// Creates a new FRI fold verifier.
    ///
    /// ## Arguments
    ///
    /// * `params` - The FRI parameters
    /// * `n_rounds` - The total number of folding rounds (typically equals n_vars for sumcheck)
    pub fn new(params: &'a FRIParams<F>) -> Self {
        let commit_rounds = calculate_fri_commit_rounds(
            params.log_batch_size(),
            params.fold_arities(),
            params.n_fold_rounds() + 1,
        );

        let expected_oracles = params.n_oracles();

        Self {
            commit_rounds,
            round_commitments: Vec::with_capacity(expected_oracles),
            curr_round: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Processes the next round, reading a commitment from the transcript if needed.
    ///
    /// ## Arguments
    ///
    /// * `transcript` - The transcript to read the commitment from (if needed)
    ///
    /// ## Returns
    ///
    /// * `Ok(Some(commitment))` if a commitment was read in this round
    /// * `Ok(None)` if no commitment was needed in this round
    pub fn process_round<B: Buf>(
        &mut self,
        transcript: &mut TranscriptReader<B>,
    ) -> Result<Option<Digest>, Error> {
        if self.curr_round >= self.n_rounds() {
            return Err(Error::InvalidArgs(
                "FRI fold verifier: too many rounds".to_string(),
            ));
        }

        let needs_commitment = self.commit_rounds[self.curr_round];
        let commitment = if needs_commitment {
            let commitment: Digest = transcript.read()?;
            self.round_commitments.push(commitment.clone());
            Some(commitment)
        } else {
            None
        };

        self.curr_round += 1;
        Ok(commitment)
    }

    /// Checks if the current round requires a commitment.
    pub fn needs_commitment(&self) -> bool {
        *self.commit_rounds.get(self.curr_round).unwrap_or(&false)
    }

    /// Returns true if all rounds have been processed.
    pub fn is_complete(&self) -> bool {
        self.curr_round == self.n_rounds()
    }

    /// Finalizes the fold verifier and returns the collected commitments.
    ///
    /// ## Returns
    ///
    /// The collected round commitments
    pub fn finalize(self) -> Result<Vec<Digest>, Error> {
        if !self.is_complete() {
            return Err(Error::InvalidArgs(format!(
                "FRI fold verifier not complete: processed {} of {} rounds",
                self.curr_round,
                self.n_rounds()
            )));
        }

        Ok(self.round_commitments)
    }

    /// Returns the current round number.
    pub fn current_round(&self) -> usize {
        self.curr_round
    }

    /// Returns the total number of rounds.
    pub fn n_rounds(&self) -> usize {
        self.commit_rounds.len()
    }
}

/// Calculates which rounds require FRI commitments.
///
/// ## Arguments
///
/// * `log_batch_size` - The log2 of the batch size
/// * `fold_arities` - The folding arities for each commitment round after the first
/// * `n_rounds` - The total number of rounds
///
/// ## Returns
///
/// A vector of booleans where `true` indicates a commitment is needed in that round.
fn calculate_fri_commit_rounds(
    log_batch_size: usize,
    fold_arities: &[usize],
    n_rounds: usize,
) -> Vec<bool> {
    let mut result = vec![false; n_rounds];
    let mut round_idx = 0;

    // First commitment happens after log_batch_size rounds
    for arity in iter::once(log_batch_size).chain(fold_arities.iter().copied()) {
        round_idx += arity;
        if round_idx < n_rounds {
            result[round_idx] = true;
        } else if round_idx == n_rounds {
            // The last round might need special handling - it's the termination round
            // We'll mark it as needing a commitment
            break;
        }
    }

    result
}
