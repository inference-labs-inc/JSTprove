//! Open phase for the sparse-MLE WHIR commitment.
//!
//! Discharges the eval claim
//!
//! ```text
//!     M(z, x, y) = v
//! ```
//!
//! at a verifier-supplied outer point `(z, x, y) ∈ E^{n_z} × E^{n_x} × E^{n_y}`.
//!
//! Phase 1d-2 (this module): the eval-claim half of the open phase.
//! Materializes the per-axis eq tables `e_z[k] = ẽq(row[k], z)`,
//! `e_x[k] = ẽq(col_x[k], x)`, `e_y[k] = ẽq(col_y[k], y)` and runs
//! the SPARK eval-claim sumcheck (Phase 1b) to reduce
//!
//! ```text
//!     v = Σ_k val[k] · e_z[k] · e_x[k] · e_y[k]
//! ```
//!
//! to a single point claim `(val(r_sc), e_z(r_sc), e_x(r_sc),
//! [e_y(r_sc)])` at a sumcheck-random `r_sc ∈ E^{log nnz}`.
//!
//! The output of this sub-phase is a [`SparseEvalClaimOpening`] that
//! the next sub-phases (1d-3 product-circuit GKR for memcheck and
//! 1d-4 WHIR opens of constituents) extend with the data the
//! verifier needs to bind these final factor evaluations to the
//! committed constituent polynomials.

use arith::{ExtensionField, Field};
use gkr_engine::Transcript;
use serdes::{ExpSerde, SerdeError, SerdeResult};

use super::eval_sumcheck::{prove_eval_sumcheck, EvalSumcheckProof};
use super::types::{eval_eq_at_index, SparseArity, SparseMleScratchPad};

/// Materialize an "eq lookup" table for one address axis.
///
/// Given the outer eval point `r` (a vector of `r.len()` extension
/// field elements) and a vector of addresses `addrs` into the
/// hypercube `{0,1}^{r.len()}`, returns the dense vector of length
/// `addrs.len()` whose `k`-th entry is `ẽq(addrs[k], r)`.
///
/// This is the per-axis "e_row" / "e_col" / "e_col_y" vector from
/// Spartan §7.2.1; the prover materializes it at open time and uses
/// it both as a factor in the eval-claim sumcheck (Phase 1b) and as
/// the "value" half of the offline-memory-checking sets in the per-
/// axis multiset arguments (Phase 1d-3).
///
/// Cost is `O(nnz · n_axis)` field operations: each entry is one
/// pass over `r.len()` to compute the eq value at the address bits.
/// For `r.len() ≤ 32` (capped by `SPARSE_MLE_MAX_LOG_DOMAIN`) this is
/// linear in `nnz` with a small constant.
///
/// # Panics
/// Panics if any address in `addrs` does not fit in `r.len()` bits.
#[must_use]
pub fn compute_eq_table_from_addresses<E: Field>(r: &[E], addrs: &[usize]) -> Vec<E> {
    let bound = if r.len() >= usize::BITS as usize {
        usize::MAX
    } else {
        1usize << r.len()
    };
    addrs
        .iter()
        .map(|&a| {
            assert!(
                a < bound,
                "compute_eq_table_from_addresses: address {a} exceeds 2^{} bound",
                r.len()
            );
            eval_eq_at_index(r, a)
        })
        .collect()
}

/// Output of the eval-claim half of the sparse-MLE open phase.
///
/// Carries the eval-claim sumcheck transcript and the final factor
/// evaluations the prover asserts at the post-sumcheck random point.
/// The 1d-3 / 1d-4 sub-phases extend this with the multiset
/// argument transcripts and the WHIR opens of the constituent
/// polynomials.
#[derive(Debug, Clone, Default)]
pub struct SparseEvalClaimOpening<E: ExtensionField> {
    pub claimed_eval: E,
    pub eval_sumcheck: EvalSumcheckProof<E>,
}

impl<E: ExtensionField> ExpSerde for SparseEvalClaimOpening<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.claimed_eval.serialize_into(&mut writer)?;
        // EvalSumcheckProof: rounds + final_evals as length-prefixed arrays
        (self.eval_sumcheck.rounds.len() as u64).serialize_into(&mut writer)?;
        for round in &self.eval_sumcheck.rounds {
            (round.evals.len() as u64).serialize_into(&mut writer)?;
            for ev in &round.evals {
                ev.serialize_into(&mut writer)?;
            }
        }
        (self.eval_sumcheck.final_evals.len() as u64).serialize_into(&mut writer)?;
        for ev in &self.eval_sumcheck.final_evals {
            ev.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        const MAX_ROUNDS: usize = 1 << 8;
        const MAX_EVALS_PER_ROUND: usize = 16;
        const MAX_FINAL_EVALS: usize = 16;

        let claimed_eval = E::deserialize_from(&mut reader)?;
        let n_rounds = u64::deserialize_from(&mut reader)? as usize;
        if n_rounds > MAX_ROUNDS {
            return Err(SerdeError::DeserializeError);
        }
        let mut rounds = Vec::with_capacity(n_rounds);
        for _ in 0..n_rounds {
            let n_evals = u64::deserialize_from(&mut reader)? as usize;
            if n_evals > MAX_EVALS_PER_ROUND {
                return Err(SerdeError::DeserializeError);
            }
            let mut evals = Vec::with_capacity(n_evals);
            for _ in 0..n_evals {
                evals.push(E::deserialize_from(&mut reader)?);
            }
            rounds.push(super::eval_sumcheck::EvalSumcheckRound { evals });
        }
        let n_final = u64::deserialize_from(&mut reader)? as usize;
        if n_final > MAX_FINAL_EVALS {
            return Err(SerdeError::DeserializeError);
        }
        let mut final_evals = Vec::with_capacity(n_final);
        for _ in 0..n_final {
            final_evals.push(E::deserialize_from(&mut reader)?);
        }
        Ok(Self {
            claimed_eval,
            eval_sumcheck: EvalSumcheckProof {
                rounds,
                final_evals,
            },
        })
    }
}

/// Run the eval-claim half of the sparse-MLE WHIR open phase.
///
/// * `scratch` is the prover scratch pad produced by `sparse_commit`,
///   carrying the dense `val` / `row` / `col_x` / `col_y` vectors
///   and the per-axis memory-checking timestamps.
/// * `claimed_eval` is the value `v = M(z, x, y)` the prover is
///   reducing. Computed once by the caller via
///   `SparseMle3::evaluate` (or recovered from a higher protocol).
/// * `(z, x, y)` is the verifier-supplied outer eval point. For
///   arity `Two` the `y` slice must be empty.
/// * `transcript` is the Fiat-Shamir transcript shared with the
///   verifier; the prover appends round messages and pulls the
///   sumcheck challenges from it.
///
/// Returns a [`SparseEvalClaimOpening`] containing the eval-claim
/// sumcheck transcript and the final factor evaluations. Phase 1d-3
/// will append the per-axis multiset arguments to this and Phase
/// 1d-4 will append the WHIR opens.
///
/// # Panics
/// Panics if `(z.len(), x.len(), y.len())` does not match the
/// scratch's `(n_z, n_x, n_y)` for the recorded arity.
pub fn sparse_open_evalclaim<F, E>(
    scratch: &SparseMleScratchPad<F>,
    claimed_eval: E,
    z: &[E],
    x: &[E],
    y: &[E],
    transcript: &mut impl Transcript,
) -> SparseEvalClaimOpening<E>
where
    F: Field,
    E: ExtensionField<BaseField = F>,
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

    // Materialize per-axis eq tables. For arity Two we omit the y
    // table; the eval-claim sumcheck is invoked with `e_y = None`
    // and the product becomes degree-3 (val · e_z · e_x).
    let e_z = compute_eq_table_from_addresses(z, &scratch.row);
    let e_x = compute_eq_table_from_addresses(x, &scratch.col_x);
    let e_y_opt: Option<Vec<E>> = if scratch.arity == SparseArity::Three {
        Some(compute_eq_table_from_addresses(y, &scratch.col_y))
    } else {
        None
    };

    let eval_sumcheck = prove_eval_sumcheck::<F, E>(
        scratch.arity,
        &scratch.val,
        &e_z,
        &e_x,
        e_y_opt.as_deref(),
        claimed_eval,
        transcript,
    );

    SparseEvalClaimOpening {
        claimed_eval,
        eval_sumcheck,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::commit::sparse_commit;
    use crate::whir::sparse::eval_sumcheck::verify_eval_sumcheck;
    use crate::whir::sparse::types::{SparseArity, SparseMle3};
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt3, GoldilocksExt4};
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
    fn compute_eq_table_matches_naive() {
        let mut rng = rng_for("eq_table_matches_naive");
        let n = 4;
        let r: Vec<GoldilocksExt3> = (0..n)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let addrs: Vec<usize> = (0..16).map(|_| rng.gen_range(0..(1usize << n))).collect();
        let table = compute_eq_table_from_addresses(&r, &addrs);
        for (k, &a) in addrs.iter().enumerate() {
            let expected = eval_eq_at_index(&r, a);
            assert_eq!(table[k], expected, "entry {k} mismatch for addr {a}");
        }
    }

    #[test]
    fn open_evalclaim_two_axis_round_trips_through_verifier() {
        let mut rng = rng_for("open_evalclaim_two_axis");
        let poly = build_two_axis(&mut rng, 3, 4, 8);
        let (_commitment, scratch, _tree, _codeword) =
            sparse_commit::<Goldilocks>(&poly).expect("valid commit");

        let z: Vec<GoldilocksExt3> = (0..3)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..4)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt3> = vec![];

        let claimed = poly.evaluate::<GoldilocksExt3>(&z, &x, &y);

        let mut p_t = Sha2T::new();
        let opening = sparse_open_evalclaim::<Goldilocks, GoldilocksExt3>(
            &scratch, claimed, &z, &x, &y, &mut p_t,
        );

        // Verifier replays just the eval-claim sumcheck. The full
        // sparse-MLE verifier will additionally check the final
        // factor evaluations against committed constituents (Phase
        // 1d-4); here we only check that the sumcheck arithmetic
        // round-trips and that the final product equals the running
        // claim.
        let mut v_t = Sha2T::new();
        let claim = verify_eval_sumcheck::<GoldilocksExt3>(
            SparseArity::Two,
            scratch.log_nnz,
            opening.claimed_eval,
            &opening.eval_sumcheck,
            &mut v_t,
        )
        .expect("eval-claim sumcheck must verify");

        assert_eq!(claim.challenges.len(), scratch.log_nnz);
        assert_eq!(claim.final_evals.len(), 3);
    }

    #[test]
    fn open_evalclaim_three_axis_round_trips_through_verifier() {
        let mut rng = rng_for("open_evalclaim_three_axis");
        let poly = build_three_axis(&mut rng, 3, 3, 3, 16);
        let (_commitment, scratch, _tree, _codeword) =
            sparse_commit::<Goldilocks>(&poly).expect("valid commit");

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
        let opening = sparse_open_evalclaim::<Goldilocks, GoldilocksExt4>(
            &scratch, claimed, &z, &x, &y, &mut p_t,
        );

        let mut v_t = Sha2T::new();
        let claim = verify_eval_sumcheck::<GoldilocksExt4>(
            SparseArity::Three,
            scratch.log_nnz,
            opening.claimed_eval,
            &opening.eval_sumcheck,
            &mut v_t,
        )
        .expect("eval-claim sumcheck must verify");

        assert_eq!(claim.challenges.len(), scratch.log_nnz);
        assert_eq!(claim.final_evals.len(), 4);
    }

    #[test]
    fn open_evalclaim_serialization_round_trip() {
        let mut rng = rng_for("open_evalclaim_serde");
        let poly = build_two_axis(&mut rng, 2, 2, 4);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let z: Vec<GoldilocksExt3> = (0..2)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..2)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let claimed = poly.evaluate::<GoldilocksExt3>(&z, &x, &[]);
        let mut p_t = Sha2T::new();
        let opening = sparse_open_evalclaim::<Goldilocks, GoldilocksExt3>(
            &scratch,
            claimed,
            &z,
            &x,
            &[],
            &mut p_t,
        );

        let mut bytes = Vec::new();
        opening.serialize_into(&mut bytes).unwrap();
        let decoded =
            SparseEvalClaimOpening::<GoldilocksExt3>::deserialize_from(&bytes[..]).unwrap();

        assert_eq!(decoded.claimed_eval, opening.claimed_eval);
        assert_eq!(
            decoded.eval_sumcheck.rounds.len(),
            opening.eval_sumcheck.rounds.len()
        );
        assert_eq!(
            decoded.eval_sumcheck.final_evals,
            opening.eval_sumcheck.final_evals
        );
    }

    #[test]
    #[should_panic(expected = "outer z length mismatch")]
    fn open_evalclaim_rejects_wrong_z_length() {
        let mut rng = rng_for("open_evalclaim_wrong_z");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let bad_z: Vec<GoldilocksExt3> = vec![GoldilocksExt3::ONE]; // expected 3
        let x: Vec<GoldilocksExt3> = vec![GoldilocksExt3::ONE; 3];
        let mut p_t = Sha2T::new();
        let _ = sparse_open_evalclaim::<Goldilocks, GoldilocksExt3>(
            &scratch,
            GoldilocksExt3::ZERO,
            &bad_z,
            &x,
            &[],
            &mut p_t,
        );
    }
}
