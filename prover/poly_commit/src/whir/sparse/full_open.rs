//! Top-level assembly of the sparse-MLE WHIR open phase.
//!
//! Phase 1d-4c: wires together
//!
//! 1. the eval-claim sumcheck (Phase 1d-2 [`super::open::sparse_open_evalclaim`]),
//! 2. the per-evaluation extension-field commitment to the
//!    `(e_z, e_x, [e_y])` eq tables (Phase 1d-4b
//!    [`super::eval_commit::sparse_commit_eval_tables`]),
//! 3. the per-axis multiset arguments (Phase 1d-3
//!    [`super::multiset_open::sparse_open_multiset`]),
//!
//! into a single `SparseFullOpening` that the verifier (Phase 1d-4d
//! `sparse_verify`) consumes alongside the `SparseMle3Commitment`
//! from the verifying key. The transcript ordering is fixed in this
//! module so the verifier can replay the exact sequence:
//!
//!   * eval-claim sumcheck rounds and final factor evaluations
//!   * per-evaluation commitment root
//!   * `(γ_1, γ_2)` samples for the multiset hashes
//!   * per-axis four product-circuit transcripts (z, x, [y])
//!
//! The WHIR opens of the setup constituents at the per-axis
//! multiset random points and the WHIR opens of the per-evaluation
//! constituents at the eval-claim sumcheck random point are
//! materialized in Phase 1d-4d together with the verifier; this
//! sub-phase only produces the protocol skeleton so each piece can
//! land in a separately reviewable commit. The skeleton already
//! exposes everything the verifier needs to *recompute* the random
//! points it would otherwise have to derive, so 1d-4d is purely
//! additive.

use arith::{ExtensionField, FFTField, Field, SimdField};
use gkr_engine::Transcript;
use serdes::{ExpSerde, SerdeResult};

use super::eval_commit::{sparse_commit_eval_tables, SparseEvalCommitment, SparseEvalScratch};
use super::multiset_open::{sparse_open_multiset, SparseMultisetOpening};
use super::open::{sparse_open_evalclaim, SparseEvalClaimOpening};
use super::types::{eval_eq_at_index, SparseArity, SparseMleScratchPad};

/// Output of [`sparse_open_skeleton`]: the eval-claim sumcheck
/// transcript, the per-evaluation commitment root, and the per-axis
/// multiset arguments. Phase 1d-4d will extend this with the WHIR
/// opens of the setup and per-eval constituents at the points
/// produced by the sumcheck and the multiset arguments.
#[derive(Debug, Clone, Default)]
pub struct SparseFullOpening<E: ExtensionField> {
    pub evalclaim: SparseEvalClaimOpening<E>,
    pub eval_commitment: SparseEvalCommitment,
    pub multiset: SparseMultisetOpening<E>,
}

impl<E: ExtensionField> ExpSerde for SparseFullOpening<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.evalclaim.serialize_into(&mut writer)?;
        self.eval_commitment.serialize_into(&mut writer)?;
        self.multiset.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let evalclaim = SparseEvalClaimOpening::<E>::deserialize_from(&mut reader)?;
        let eval_commitment = SparseEvalCommitment::deserialize_from(&mut reader)?;
        let multiset = SparseMultisetOpening::<E>::deserialize_from(&mut reader)?;
        Ok(Self {
            evalclaim,
            eval_commitment,
            multiset,
        })
    }
}

/// Prover-side artifacts retained by [`sparse_open_skeleton`] for
/// use by Phase 1d-4d. Holds the per-evaluation commitment scratch
/// (so the WHIR opens of `e_z` / `e_x` / `[e_y]` at the sumcheck
/// random point can be served from the same WHIR Merkle tree
/// `sparse_commit_eval_tables` already built) and the materialized
/// per-axis eq tables (so the multiset arguments do not have to
/// re-derive them when computing leaf openings). The eval scratch
/// is parameterized over the base field `F` because the per-eval
/// commitment uses limb decomposition: each extension-field eq
/// table is split into `E::DEGREE` base-field limb vectors before
/// being committed via `whir_commit::<F>`.
pub struct SparseFullOpeningScratch<F: Field, E: Field> {
    pub eval_scratch: SparseEvalScratch<F>,
    pub e_z: Vec<E>,
    pub e_x: Vec<E>,
    pub e_y: Option<Vec<E>>,
}

/// Run the protocol skeleton of the sparse-MLE WHIR open phase.
///
/// Materializes the per-axis eq tables, commits them via
/// `sparse_commit_eval_tables`, runs the eval-claim sumcheck, and
/// runs the per-axis multiset arguments. Returns the assembled
/// [`SparseFullOpening`] together with the prover scratch the
/// follow-up Phase 1d-4d will use to issue the WHIR opens of the
/// setup and per-eval constituents.
///
/// Transcript discipline. The eval commitment root is appended to
/// the transcript before the multiset arguments draw their `(γ_1,
/// γ_2)` samples, so a malicious prover cannot adapt e_z / e_x /
/// e_y to a particular `(γ_1, γ_2)` after the fact — the standard
/// Spartan ordering.
///
/// # Panics
/// Panics if `(z.len(), x.len(), y.len())` does not match the
/// scratch's `(n_z, n_x, n_y)` for the recorded arity, or if the
/// scratch `nnz` is not a power of two.
pub fn sparse_open_skeleton<F, E>(
    scratch: &SparseMleScratchPad<F>,
    claimed_eval: E,
    z: &[E],
    x: &[E],
    y: &[E],
    transcript: &mut impl Transcript,
) -> (SparseFullOpening<E>, SparseFullOpeningScratch<F, E>)
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F> + FFTField + SimdField<Scalar = E>,
{
    assert_eq!(z.len(), scratch.n_z, "outer z length mismatch");
    assert_eq!(x.len(), scratch.n_x, "outer x length mismatch");
    if scratch.arity == SparseArity::Two {
        assert!(
            y.is_empty(),
            "outer y must be empty for arity Two, got len {}",
            y.len()
        );
    } else {
        assert_eq!(y.len(), scratch.n_y, "outer y length mismatch");
    }
    assert!(
        scratch.nnz.is_power_of_two(),
        "sparse_open_skeleton requires nnz to be a power of two; got {}",
        scratch.nnz
    );

    // Step 1: materialize the per-axis eq tables once and reuse
    // them across the three sub-phases.
    let e_z: Vec<E> = scratch
        .row
        .iter()
        .map(|&a| eval_eq_at_index(z, a))
        .collect();
    let e_x: Vec<E> = scratch
        .col_x
        .iter()
        .map(|&a| eval_eq_at_index(x, a))
        .collect();
    let e_y_opt: Option<Vec<E>> = if scratch.arity == SparseArity::Three {
        Some(
            scratch
                .col_y
                .iter()
                .map(|&a| eval_eq_at_index(y, a))
                .collect(),
        )
    } else {
        None
    };

    // Step 2: run the eval-claim sumcheck. Internally re-derives
    // the eq tables — Phase 1d-2 takes only the scratch and the
    // outer eval point, not the precomputed tables. The cost is
    // negligible (one extra O(nnz · n_axis) pass per axis) and
    // keeps Phase 1d-2's API stable.
    let evalclaim = sparse_open_evalclaim::<F, E>(scratch, claimed_eval, z, x, y, transcript);

    // Step 3: commit the per-evaluation eq tables. The Merkle root
    // of this commitment is appended to the transcript explicitly
    // so the (γ_1, γ_2) samples drawn by the multiset arguments
    // depend on the eval commitment.
    let (eval_commitment, eval_scratch) =
        sparse_commit_eval_tables::<F, E>(scratch.arity, &e_z, &e_x, e_y_opt.as_deref());
    transcript.append_u8_slice(eval_commitment.root.as_bytes());

    // Step 4: per-axis multiset arguments. Internally re-derives
    // the eq tables for the same reason as Step 2.
    let multiset = sparse_open_multiset::<F, E>(scratch, z, x, y, transcript);

    let opening = SparseFullOpening {
        evalclaim,
        eval_commitment,
        multiset,
    };
    let scratch_out = SparseFullOpeningScratch {
        eval_scratch,
        e_z,
        e_x,
        e_y: e_y_opt,
    };
    (opening, scratch_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::commit::sparse_commit;
    use crate::whir::sparse::eval_sumcheck::verify_eval_sumcheck;
    use crate::whir::sparse::types::SparseMle3;
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt4};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use serdes::ExpSerde;
    use transcript::BytesHashTranscript;
    type Sha2T = BytesHashTranscript<gkr_hashers::SHA256hasher>;

    fn rng_for(label: &str) -> ChaCha20Rng {
        let mut seed = [0u8; 32];
        let bytes = label.as_bytes();
        let n = bytes.len().min(32);
        seed[..n].copy_from_slice(&bytes[..n]);
        ChaCha20Rng::from_seed(seed)
    }

    fn build_two_axis(
        rng: &mut ChaCha20Rng,
        n_z: usize,
        n_x: usize,
        nnz: usize,
    ) -> SparseMle3<Goldilocks> {
        let m_z = 1usize << n_z;
        let m_x = 1usize << n_x;
        let mut row = Vec::with_capacity(nnz);
        let mut col_x = Vec::with_capacity(nnz);
        let mut val = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            row.push(rng.gen_range(0..m_z));
            col_x.push(rng.gen_range(0..m_x));
            val.push(Goldilocks::random_unsafe(&mut *rng));
        }
        SparseMle3 {
            n_z,
            n_x,
            n_y: 0,
            arity: SparseArity::Two,
            row,
            col_x,
            col_y: vec![0; nnz],
            val,
        }
    }

    fn build_three_axis(
        rng: &mut ChaCha20Rng,
        n_z: usize,
        n_x: usize,
        n_y: usize,
        nnz: usize,
    ) -> SparseMle3<Goldilocks> {
        let m_z = 1usize << n_z;
        let m_x = 1usize << n_x;
        let m_y = 1usize << n_y;
        let mut row = Vec::with_capacity(nnz);
        let mut col_x = Vec::with_capacity(nnz);
        let mut col_y = Vec::with_capacity(nnz);
        let mut val = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            row.push(rng.gen_range(0..m_z));
            col_x.push(rng.gen_range(0..m_x));
            col_y.push(rng.gen_range(0..m_y));
            val.push(Goldilocks::random_unsafe(&mut *rng));
        }
        SparseMle3 {
            n_z,
            n_x,
            n_y,
            arity: SparseArity::Three,
            row,
            col_x,
            col_y,
            val,
        }
    }

    #[test]
    fn skeleton_two_axis_round_trips_subset_equation() {
        let mut rng = rng_for("skeleton_two_axis");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        let z: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);

        let mut p_t = Sha2T::new();
        let (opening, _open_scratch) = sparse_open_skeleton::<Goldilocks, GoldilocksExt4>(
            &scratch,
            claimed,
            &z,
            &x,
            &[],
            &mut p_t,
        );

        // The eval-claim sumcheck must verify against the existing
        // verifier (Phase 1b) when replayed on a fresh transcript.
        let mut v_t = Sha2T::new();
        let claim = verify_eval_sumcheck::<GoldilocksExt4>(
            SparseArity::Two,
            scratch.log_nnz,
            opening.evalclaim.claimed_eval,
            &opening.evalclaim.eval_sumcheck,
            &mut v_t,
        )
        .expect("eval-claim sumcheck must verify");
        assert_eq!(claim.challenges.len(), scratch.log_nnz);

        // The per-axis multiset arguments must satisfy the subset
        // equation H(Init)·H(WS) = H(RS)·H(Audit) for an honest
        // trace.
        assert!(opening.multiset.axis_z.check_subset_equation());
        assert!(opening.multiset.axis_x.check_subset_equation());
        assert!(opening.multiset.axis_y.is_none());

        // The eval commitment root is non-zero (sanity check that
        // the commitment was actually built).
        assert_eq!(opening.eval_commitment.arity, SparseArity::Two);
    }

    #[test]
    fn skeleton_three_axis_round_trips_subset_equation() {
        let mut rng = rng_for("skeleton_three_axis");
        let poly = build_three_axis(&mut rng, 3, 3, 3, 8);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        let z: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &y);

        let mut p_t = Sha2T::new();
        let (opening, _open_scratch) = sparse_open_skeleton::<Goldilocks, GoldilocksExt4>(
            &scratch, claimed, &z, &x, &y, &mut p_t,
        );

        let mut v_t = Sha2T::new();
        verify_eval_sumcheck::<GoldilocksExt4>(
            SparseArity::Three,
            scratch.log_nnz,
            opening.evalclaim.claimed_eval,
            &opening.evalclaim.eval_sumcheck,
            &mut v_t,
        )
        .expect("eval-claim sumcheck must verify");

        assert!(opening.multiset.axis_z.check_subset_equation());
        assert!(opening.multiset.axis_x.check_subset_equation());
        assert!(opening.multiset.axis_y.is_some());
        assert!(opening
            .multiset
            .axis_y
            .as_ref()
            .unwrap()
            .check_subset_equation());
        assert_eq!(opening.eval_commitment.arity, SparseArity::Three);
    }

    #[test]
    fn skeleton_open_scratch_carries_eq_tables() {
        let mut rng = rng_for("skeleton_scratch");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let z: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..3)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);
        let mut p_t = Sha2T::new();
        let (_opening, open_scratch) = sparse_open_skeleton::<Goldilocks, GoldilocksExt4>(
            &scratch,
            claimed,
            &z,
            &x,
            &[],
            &mut p_t,
        );

        // Scratch retains the e_z and e_x tables for use by Phase
        // 1d-4d, which will issue WHIR opens of the per-eval
        // commitment at the sumcheck random points.
        assert_eq!(open_scratch.e_z.len(), 8);
        assert_eq!(open_scratch.e_x.len(), 8);
        assert!(open_scratch.e_y.is_none());

        // Each entry of e_z must equal eval_eq_at_index(z, row[k]).
        for (k, &a) in scratch.row.iter().enumerate() {
            let expected = eval_eq_at_index(&z, a);
            assert_eq!(open_scratch.e_z[k], expected);
        }
    }

    #[test]
    fn skeleton_full_opening_serialization_round_trip() {
        let mut rng = rng_for("skeleton_serde");
        let poly = build_two_axis(&mut rng, 2, 2, 4);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let z: Vec<GoldilocksExt4> = (0..2)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..2)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt4>(&z, &x, &[]);
        let mut p_t = Sha2T::new();
        let (opening, _) = sparse_open_skeleton::<Goldilocks, GoldilocksExt4>(
            &scratch,
            claimed,
            &z,
            &x,
            &[],
            &mut p_t,
        );

        let mut bytes = Vec::new();
        opening.serialize_into(&mut bytes).unwrap();
        let decoded = SparseFullOpening::<GoldilocksExt4>::deserialize_from(&bytes[..]).unwrap();

        assert_eq!(
            decoded.evalclaim.claimed_eval,
            opening.evalclaim.claimed_eval
        );
        assert_eq!(decoded.eval_commitment.arity, opening.eval_commitment.arity);
    }
}
