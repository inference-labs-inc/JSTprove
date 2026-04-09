//! WHIR open / verify glue for the sparse-MLE commitment.
//!
//! Bridges the high-level "open constituent `i` of the (setup |
//! per-eval) commitment at sub-point `r ∈ E^μ`" abstraction onto the
//! low-level [`crate::whir::pcs_trait_impl::whir_open`] /
//! [`crate::whir::pcs_trait_impl::whir_verify`] entry points. The
//! glue handles three things:
//!
//! 1. **Combined-point construction.** Translates a constituent slot
//!    identifier and a sub-point into the combined-polynomial eval
//!    point via [`super::combined_point::combined_eval_point`] (for
//!    setup-commit constituents) or
//!    [`super::eval_commit::eval_combined_point`] (for per-evaluation
//!    constituents).
//!
//! 2. **Sub-point padding.** When the multiset random point is
//!    shorter than the combined polynomial's `μ`, the missing
//!    high-order variables are padded with `E::ZERO`. This is sound
//!    by the multilinearity identity
//!
//!    ```text
//!        slot_mle( (r, 0_high) )  =  constituent_mle(r)
//!    ```
//!
//!    when the constituent fills only the low part of the slot and
//!    the padding entries are all zero (which is exactly how
//!    `populate_slot` writes them in [`super::commit::sparse_commit`]
//!    and `write_constituent_limbs` writes them in
//!    [`super::eval_commit::sparse_commit_eval_tables`]).
//!
//! 3. **Claim computation.** Each WHIR open carries a claimed
//!    evaluation alongside the proof; the claim is the dense MLE
//!    of the constituent polynomial at the sub-point and is
//!    computed once via [`MultiLinearPoly::evaluate_with_buffer`]
//!    against the slot's source vector. Computing the claim
//!    against the *original* constituent vector (not the padded
//!    slot) avoids any work proportional to `2^μ`.

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::Transcript;
use polynomials::MultiLinearPoly;
use serdes::{ExpSerde, SerdeResult};
use tree::Tree;

use crate::whir::pcs_trait_impl::{whir_open, whir_verify};
use crate::whir::types::{WhirCommitment, WhirOpening};

/// A WHIR opening together with the field element the prover
/// asserts is the polynomial's evaluation at the (implicit) point.
///
/// The eval point itself is not transmitted — both the prover and
/// the verifier reconstruct it from the same `(layout, slot,
/// sub_point)` triple via the combined-point translator. Sending
/// only `(claim, opening)` keeps the wire format minimal.
#[derive(Debug, Clone, Default)]
pub struct WhirOpenWithClaim<E: ExtensionField> {
    pub claim: E,
    pub opening: WhirOpening<E>,
}

impl<E: ExtensionField> ExpSerde for WhirOpenWithClaim<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.claim.serialize_into(&mut writer)?;
        self.opening.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let claim = E::deserialize_from(&mut reader)?;
        let opening = WhirOpening::<E>::deserialize_from(&mut reader)?;
        Ok(Self { claim, opening })
    }
}

/// Errors raised by the sparse WHIR-glue layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhirGlueError {
    /// `sub_point.len() > target_mu` — sub-point cannot be embedded
    /// into the combined polynomial's mu-variable space without
    /// truncation.
    SubPointTooLong { len: usize, max: usize },
    /// The opening's WHIR open did not pass `whir_verify` against
    /// the supplied commitment and reconstructed eval point.
    WhirVerifyRejected,
}

impl std::fmt::Display for WhirGlueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubPointTooLong { len, max } => {
                write!(
                    f,
                    "sub-point length {len} exceeds combined-polynomial mu {max}"
                )
            }
            Self::WhirVerifyRejected => write!(f, "whir_verify rejected the opening"),
        }
    }
}

impl std::error::Error for WhirGlueError {}

/// Pad a sub-point to a target length by appending `E::ZERO` to the
/// high-order variables. Returns `Err` if `sub_point.len() >
/// target_mu`.
pub fn pad_sub_point<E: Field>(sub_point: &[E], target_mu: usize) -> Result<Vec<E>, WhirGlueError> {
    if sub_point.len() > target_mu {
        return Err(WhirGlueError::SubPointTooLong {
            len: sub_point.len(),
            max: target_mu,
        });
    }
    let mut padded = Vec::with_capacity(target_mu);
    padded.extend_from_slice(sub_point);
    while padded.len() < target_mu {
        padded.push(E::ZERO);
    }
    Ok(padded)
}

/// Compute `constituent_mle(sub_point)` for a base-field constituent
/// vector at an extension-field sub-point.
///
/// Uses [`MultiLinearPoly::evaluate_with_buffer`] which is the
/// preferred allocation-free MLE evaluation routine in the
/// polynomials crate. The output lives in `E`.
///
/// Pre-condition: `sub_point.len()` must equal `log₂ source.len()`,
/// or else the function pads `source` with zeros to the next
/// power of two and matches the padded length. The latter behavior
/// is the one used by Phase 1d-1 / Phase 1d-4b's slot population.
pub fn evaluate_constituent_at_sub_point<F, E>(source: &[F], sub_point: &[E]) -> E
where
    F: Field,
    E: Field
        + From<F>
        + std::ops::Mul<F, Output = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<E, Output = E>,
{
    // Pad source to 2^sub_point.len() with zeros so the MLE
    // evaluation is over the same hypercube as the layout's slot.
    let target_len = 1usize << sub_point.len();
    if source.len() == target_len {
        let mut scratch = vec![E::ZERO; source.len()];
        return MultiLinearPoly::evaluate_with_buffer::<E, E>(source, sub_point, &mut scratch);
    }
    let mut padded = vec![F::ZERO; target_len];
    padded[..source.len()].copy_from_slice(source);
    let mut scratch = vec![E::ZERO; padded.len()];
    MultiLinearPoly::evaluate_with_buffer::<E, E>(&padded, sub_point, &mut scratch)
}

/// Open a single constituent of a WHIR-committed combined polynomial
/// at the given combined eval point, returning the claim alongside
/// the WHIR open transcript.
///
/// `combined_evals` and `combined_codeword` are the dense vector and
/// the WHIR codeword the commitment was built from (i.e. what the
/// prover passed to `whir_commit::<F>`). `combined_tree` is the
/// matching Merkle tree. `combined_eval_point` is the
/// already-translated eval point in `E^{total_vars}`.
///
/// The claim is computed by evaluating the *combined* dense MLE at
/// the eval point. For real openings the caller will typically
/// supply a precomputed claim from the original constituent
/// (avoiding `O(2^total_vars)` work) and use this function only for
/// the WHIR open call. A convenience wrapper that takes a
/// `(slot, sub_point)` is layered on top by the sparse open phase.
pub fn whir_open_at_combined_point<F, E>(
    combined_evals: &[F],
    combined_codeword: &[F],
    combined_tree: &Tree,
    combined_num_vars: usize,
    combined_eval_point: &[E],
    transcript: &mut impl Transcript,
) -> WhirOpenWithClaim<E>
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>
        + FFTField
        + SimdField<Scalar = E>
        + std::ops::Add<F, Output = E>
        + std::ops::Mul<F, Output = E>,
{
    debug_assert_eq!(combined_evals.len(), 1usize << combined_num_vars);
    debug_assert_eq!(combined_eval_point.len(), combined_num_vars);

    // Compute the claim against the dense combined polynomial. This
    // gives the *exact* value the verifier will check whir_verify
    // against, which is what we want — the verifier reconstructs the
    // same eval point and the WHIR transcript binds claim ↔ opening.
    let mut scratch = vec![E::ZERO; combined_evals.len()];
    let claim = MultiLinearPoly::evaluate_with_buffer::<E, E>(
        combined_evals,
        combined_eval_point,
        &mut scratch,
    );

    let opening = whir_open::<F, E>(
        combined_evals,
        combined_codeword,
        combined_tree,
        combined_num_vars,
        combined_eval_point,
        transcript,
    );

    WhirOpenWithClaim { claim, opening }
}

/// Verify a single sparse-MLE constituent opening against the
/// supplied commitment.
///
/// The verifier reconstructs the combined eval point from `(layout,
/// slot, sub_point)` exactly the way the prover built it, then
/// invokes `whir_verify` against the commitment. The opening's
/// `claim` field is the polynomial evaluation the prover asserts;
/// the verifier passes it through to `whir_verify` and the WHIR
/// internal sumcheck is what binds the claim to the commitment.
pub fn whir_verify_with_claim<F, E>(
    commitment: &WhirCommitment,
    combined_eval_point: &[E],
    open_with_claim: &WhirOpenWithClaim<E>,
    transcript: &mut impl Transcript,
) -> Result<(), WhirGlueError>
where
    F: FFTField,
    E: ExtensionField<BaseField = F> + FFTField,
{
    let ok = whir_verify::<F, E>(
        commitment,
        combined_eval_point,
        open_with_claim.claim,
        &open_with_claim.opening,
        transcript,
    );
    if ok {
        Ok(())
    } else {
        Err(WhirGlueError::WhirVerifyRejected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::pcs_trait_impl::whir_commit;
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt4};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use transcript::BytesHashTranscript;
    type Sha2T = BytesHashTranscript<gkr_hashers::SHA256hasher>;

    fn rng_for(label: &str) -> ChaCha20Rng {
        let mut seed = [0u8; 32];
        let bytes = label.as_bytes();
        let n = bytes.len().min(32);
        seed[..n].copy_from_slice(&bytes[..n]);
        ChaCha20Rng::from_seed(seed)
    }

    #[test]
    fn pad_sub_point_pads_with_zero() {
        let sub: Vec<GoldilocksExt4> =
            vec![GoldilocksExt4::from(7u64), GoldilocksExt4::from(11u64)];
        let padded = pad_sub_point(&sub, 5).unwrap();
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[..2], sub[..]);
        for i in 2..5 {
            assert_eq!(padded[i], GoldilocksExt4::ZERO);
        }
    }

    #[test]
    fn pad_sub_point_rejects_overlong_input() {
        let sub: Vec<GoldilocksExt4> = vec![GoldilocksExt4::ZERO; 6];
        let err = pad_sub_point(&sub, 4).unwrap_err();
        assert!(matches!(err, WhirGlueError::SubPointTooLong { .. }));
    }

    #[test]
    fn whir_open_round_trip_against_dense_oracle() {
        // Build a small base-field dense polynomial, commit it via
        // whir_commit, open it at a random extension-field point via
        // whir_open_at_combined_point, then run whir_verify_with_claim
        // and assert it accepts. The claim is computed inside the
        // open helper from the dense oracle, so any disagreement
        // between the open's transcript and the verify's transcript
        // would surface as a verification failure.
        let mut rng = rng_for("whir_open_round_trip");
        let n_vars = 5; // 32-element polynomial
        let evals: Vec<Goldilocks> = (0..(1usize << n_vars))
            .map(|_| Goldilocks::random_unsafe(&mut rng))
            .collect();
        let (commitment, tree, codeword) = whir_commit(&evals);
        let point: Vec<GoldilocksExt4> = (0..n_vars)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        let mut p_t = Sha2T::new();
        let opening = whir_open_at_combined_point::<Goldilocks, GoldilocksExt4>(
            &evals, &codeword, &tree, n_vars, &point, &mut p_t,
        );

        // Sanity check: the claim equals the dense MLE eval at the point
        let mut scratch = vec![GoldilocksExt4::ZERO; evals.len()];
        let dense_eval = MultiLinearPoly::evaluate_with_buffer::<GoldilocksExt4, GoldilocksExt4>(
            &evals,
            &point,
            &mut scratch,
        );
        assert_eq!(opening.claim, dense_eval);

        let mut v_t = Sha2T::new();
        whir_verify_with_claim::<Goldilocks, GoldilocksExt4>(
            &commitment,
            &point,
            &opening,
            &mut v_t,
        )
        .expect("verify must accept honest open");
    }

    #[test]
    fn evaluate_constituent_at_sub_point_pads_short_source() {
        // Source is 5 entries, target is 8 entries (sub_point of length 3).
        // The padding-with-zeros pre-pass should give the same MLE eval as
        // an explicit padded copy.
        let mut rng = rng_for("eval_constituent_pad");
        let source: Vec<Goldilocks> = (0..5)
            .map(|_| Goldilocks::random_unsafe(&mut rng))
            .collect();
        let sub_point: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        let v_helper =
            evaluate_constituent_at_sub_point::<Goldilocks, GoldilocksExt4>(&source, &sub_point);

        let mut padded = vec![Goldilocks::ZERO; 8];
        padded[..5].copy_from_slice(&source);
        let mut scratch = vec![GoldilocksExt4::ZERO; 8];
        let v_direct = MultiLinearPoly::evaluate_with_buffer::<GoldilocksExt4, GoldilocksExt4>(
            &padded,
            &sub_point,
            &mut scratch,
        );

        assert_eq!(v_helper, v_direct);
    }

    #[test]
    fn evaluate_constituent_at_sub_point_passes_through_exact_length() {
        // Source already has 8 entries; helper must use it directly
        // without padding (and produce the same value as direct
        // evaluate_with_buffer).
        let mut rng = rng_for("eval_constituent_exact");
        let source: Vec<Goldilocks> = (0..8)
            .map(|_| Goldilocks::random_unsafe(&mut rng))
            .collect();
        let sub_point: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let v_helper =
            evaluate_constituent_at_sub_point::<Goldilocks, GoldilocksExt4>(&source, &sub_point);
        let mut scratch = vec![GoldilocksExt4::ZERO; source.len()];
        let v_direct = MultiLinearPoly::evaluate_with_buffer::<GoldilocksExt4, GoldilocksExt4>(
            &source,
            &sub_point,
            &mut scratch,
        );
        assert_eq!(v_helper, v_direct);
    }
}
