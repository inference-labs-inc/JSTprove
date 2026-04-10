//! SPARK eval-claim sumcheck.
//!
//! Reduces the claim
//!
//! ```text
//!     v = Σ_k val[k] · e_z[k] · e_x[k] · e_y[k]
//! ```
//!
//! over `k ∈ {0, 1}^{log nnz}` to a single point claim
//! `(val(r), e_z(r), e_x(r), e_y(r))` at a sumcheck-random point
//! `r ∈ E^{log nnz}`. The number of factors in the product is `4` for
//! arity `Three` (the `mul(z,x,y)` selector) and `3` for arity `Two`
//! (the `add(z,x)` selector); the round polynomial is therefore
//! degree `4` or `3` respectively.
//!
//! This is the analogue of Spartan §6 (R1CS sumcheck) and §7.2.1
//! ("`Circuit_{eval-opt}`" subroutine that computes
//! `v ← Σ_{k=0}^{n-1} val[k] · e_row[k] · e_col[k]`), extended by one
//! product factor for the second input axis. The protocol is
//! self-contained: the verifier consumes a [`EvalSumcheckProof`]
//! plus a starting `claimed_v` and returns the random point and the
//! final factor evaluations the caller must verify against the
//! sparse-MLE constituent commitments (Phase 1d).
//!
//! Soundness: standard sumcheck soundness over a degree-`d` round
//! polynomial gives error at most `d / |F|` per round, totalling
//! `(log nnz) · d / |F|`. For our parameters (Goldilocks Ext3 / Ext4,
//! `d ≤ 4`, `log nnz ≤ 32`) this is `< 2^{-180}`, well above the
//! 128-bit target.

use arith::{ExtensionField, Field};
use gkr_engine::Transcript;

use super::types::SparseArity;

/// Number of multilinear factors in the SPARK eval-claim product.
///
/// `arity::Two`   → factors are `(val, e_z, e_x)` so the product has
///                  arity `3`.
/// `arity::Three` → factors are `(val, e_z, e_x, e_y)` so the product
///                  has arity `4`.
#[inline]
#[must_use]
pub const fn product_arity(arity: SparseArity) -> usize {
    match arity {
        SparseArity::Two => 3,
        SparseArity::Three => 4,
    }
}

/// Degree of the univariate round polynomial sent by the prover in
/// each sumcheck round. Equals the number of multilinear factors in
/// the product, since each factor contributes degree `1` in the
/// variable being summed over.
#[inline]
#[must_use]
pub const fn round_poly_degree(arity: SparseArity) -> usize {
    product_arity(arity)
}

/// One round of the SPARK eval sumcheck.
///
/// The prover sends `d + 1` evaluations of the univariate round
/// polynomial `g_i(X)` at the integer points `X = 0, 1, …, d`.
/// Sending all `d + 1` evaluations explicitly is slightly redundant
/// (the constraint `g_i(0) + g_i(1) = running_claim` lets the verifier
/// recover one of them) but costs only one extra extension-field
/// element per round and keeps the verifier check trivially auditable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalSumcheckRound<F: Field> {
    pub evals: Vec<F>,
}

/// Full transcript of a SPARK eval sumcheck.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalSumcheckProof<F: Field> {
    pub rounds: Vec<EvalSumcheckRound<F>>,
    /// Final factor evaluations at the random point in the order
    /// `(val, e_z, e_x[, e_y])`. Length is `product_arity(arity)`.
    pub final_evals: Vec<F>,
}

/// Output of [`verify_eval_sumcheck`]: the sumcheck-random point and
/// the final factor evaluations the caller must check against the
/// sparse-MLE constituent commitments.
#[derive(Debug, Clone)]
pub struct EvalSumcheckClaim<F: Field> {
    pub challenges: Vec<F>,
    pub final_evals: Vec<F>,
    pub final_product: F,
}

/// Run the prover side of the SPARK eval sumcheck.
///
/// * `arity` selects the product arity (`Two` ⇒ 3-factor, `Three` ⇒
///   4-factor).
/// * `val` is the dense base-field vector of length `nnz` (lifted to
///   the extension field internally).
/// * `e_z`, `e_x`, `e_y` are the dense extension-field eq tables of
///   length `nnz`. Pass `e_y = None` for arity `Two`.
/// * `claimed_v` is the value the protocol is reducing.
/// * `transcript` is the Fiat-Shamir transcript; the prover appends
///   round messages and pulls challenges from it.
///
/// # Panics
/// Panics if any input vector length disagrees with `val.len()`, if
/// `val.len()` is not a power of two, or if `e_y.is_some()` does not
/// match `arity`.
pub fn prove_eval_sumcheck<F, E>(
    arity: SparseArity,
    val: &[F],
    e_z: &[E],
    e_x: &[E],
    e_y: Option<&[E]>,
    claimed_v: E,
    transcript: &mut impl Transcript,
) -> EvalSumcheckProof<E>
where
    F: Field,
    E: ExtensionField<BaseField = F>,
{
    let nnz = val.len();
    assert_eq!(e_z.len(), nnz, "e_z length must match val length");
    assert_eq!(e_x.len(), nnz, "e_x length must match val length");
    let three_axis = e_y.is_some();
    assert_eq!(
        three_axis,
        arity == SparseArity::Three,
        "e_y presence must match arity"
    );
    if let Some(e_y_slice) = e_y {
        assert_eq!(e_y_slice.len(), nnz, "e_y length must match val length");
    }
    assert!(nnz.is_power_of_two(), "nnz must be a power of two");
    let log_nnz = nnz.trailing_zeros() as usize;

    let mut val_table: Vec<E> = val.iter().map(|&v| E::from(v)).collect();
    let mut e_z_table: Vec<E> = e_z.to_vec();
    let mut e_x_table: Vec<E> = e_x.to_vec();
    let mut e_y_table: Vec<E> = if three_axis {
        e_y.expect("checked above").to_vec()
    } else {
        Vec::new()
    };

    let d = round_poly_degree(arity);
    let mut rounds = Vec::with_capacity(log_nnz);
    let mut running_claim = claimed_v;

    transcript.append_field_element(&claimed_v);

    for _ in 0..log_nnz {
        let half = val_table.len() / 2;

        // For each evaluation point e ∈ {0, 1, …, d}, compute
        //   g_i(e) = Σ_{k=0}^{half-1} P( (1-e)·T[2k] + e·T[2k+1] )
        // where P is the product over the active factors.
        let mut evals = vec![E::ZERO; d + 1];
        for (e_idx, slot) in evals.iter_mut().enumerate() {
            let e_pt = E::from(e_idx as u32);
            let one_minus_e = E::ONE - e_pt;
            let mut acc = E::ZERO;
            for k in 0..half {
                let v = val_table[2 * k] * one_minus_e + val_table[2 * k + 1] * e_pt;
                let z = e_z_table[2 * k] * one_minus_e + e_z_table[2 * k + 1] * e_pt;
                let x = e_x_table[2 * k] * one_minus_e + e_x_table[2 * k + 1] * e_pt;
                if three_axis {
                    let y = e_y_table[2 * k] * one_minus_e + e_y_table[2 * k + 1] * e_pt;
                    acc += v * z * x * y;
                } else {
                    acc += v * z * x;
                }
            }
            *slot = acc;
        }

        // Sumcheck invariant: g_i(0) + g_i(1) must equal the running
        // claim. If this fails the prover constructed an inconsistent
        // table; we treat it as a programming error rather than a
        // protocol-level rejection (an honest prover never trips it).
        debug_assert_eq!(
            evals[0] + evals[1],
            running_claim,
            "eval sumcheck invariant violated at round {} of {}",
            rounds.len(),
            log_nnz
        );

        for ev in &evals {
            transcript.append_field_element(ev);
        }
        let r: E = transcript.generate_field_element();

        // Fold every active table at the challenge.
        let one_minus_r = E::ONE - r;
        for k in 0..half {
            val_table[k] = val_table[2 * k] * one_minus_r + val_table[2 * k + 1] * r;
            e_z_table[k] = e_z_table[2 * k] * one_minus_r + e_z_table[2 * k + 1] * r;
            e_x_table[k] = e_x_table[2 * k] * one_minus_r + e_x_table[2 * k + 1] * r;
            if three_axis {
                e_y_table[k] = e_y_table[2 * k] * one_minus_r + e_y_table[2 * k + 1] * r;
            }
        }
        val_table.truncate(half);
        e_z_table.truncate(half);
        e_x_table.truncate(half);
        if three_axis {
            e_y_table.truncate(half);
        }

        running_claim = lagrange_interpolate_at(&evals, r);
        rounds.push(EvalSumcheckRound { evals });
    }

    let final_evals = if three_axis {
        vec![val_table[0], e_z_table[0], e_x_table[0], e_y_table[0]]
    } else {
        vec![val_table[0], e_z_table[0], e_x_table[0]]
    };

    EvalSumcheckProof {
        rounds,
        final_evals,
    }
}

/// Run the verifier side of the SPARK eval sumcheck.
///
/// Returns `Some(EvalSumcheckClaim)` carrying the random point and
/// the final factor evaluations on success. Returns `None` if the
/// transcript is malformed (wrong number of rounds, wrong polynomial
/// degree, sumcheck invariant violation, or final-product mismatch).
/// The caller must additionally verify the final factor evaluations
/// against the sparse-MLE constituent commitments before accepting
/// the opening.
pub fn verify_eval_sumcheck<E: ExtensionField>(
    arity: SparseArity,
    log_nnz: usize,
    claimed_v: E,
    proof: &EvalSumcheckProof<E>,
    transcript: &mut impl Transcript,
) -> Option<EvalSumcheckClaim<E>> {
    let d = round_poly_degree(arity);
    let arity_n = product_arity(arity);
    if proof.rounds.len() != log_nnz {
        return None;
    }
    if proof.final_evals.len() != arity_n {
        return None;
    }

    let mut running_claim = claimed_v;
    transcript.append_field_element(&claimed_v);
    let mut challenges = Vec::with_capacity(log_nnz);
    for round in &proof.rounds {
        if round.evals.len() != d + 1 {
            return None;
        }
        if round.evals[0] + round.evals[1] != running_claim {
            return None;
        }
        for ev in &round.evals {
            transcript.append_field_element(ev);
        }
        let r: E = transcript.generate_field_element();
        challenges.push(r);
        running_claim = lagrange_interpolate_at(&round.evals, r);
    }

    let final_product = proof
        .final_evals
        .iter()
        .copied()
        .fold(E::ONE, |acc, v| acc * v);
    if final_product != running_claim {
        return None;
    }

    Some(EvalSumcheckClaim {
        challenges,
        final_evals: proof.final_evals.clone(),
        final_product,
    })
}

/// Lagrange-interpolate the polynomial defined by
/// `(0, evals[0]), (1, evals[1]), …, (n-1, evals[n-1])` and evaluate
/// it at the field point `r`.
///
/// Used to evaluate each round polynomial at the verifier's
/// challenge after it has been received in evaluation form. Cost is
/// `O(n^2)` field operations; for our `n ≤ 5` (degree 4 + 1) this is
/// negligible.
fn lagrange_interpolate_at<F: Field>(evals: &[F], r: F) -> F {
    let n = evals.len();
    let mut result = F::ZERO;
    for i in 0..n {
        let mut numerator = F::ONE;
        let mut denominator = F::ONE;
        let xi = F::from(i as u32);
        for j in 0..n {
            if j == i {
                continue;
            }
            let xj = F::from(j as u32);
            numerator *= r - xj;
            denominator *= xi - xj;
        }
        let inv = denominator
            .inv()
            .expect("Lagrange basis denominator is non-zero for distinct integer nodes");
        result += evals[i] * numerator * inv;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::types::eval_eq_at_index;
    use crate::whir::sparse::SparseMle3;
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt3, GoldilocksExt4};
    use rand::{Rng, SeedableRng};
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

    fn build_sparse_two_axis(
        rng: &mut ChaCha20Rng,
        n_z: usize,
        n_x: usize,
        nnz: usize,
    ) -> SparseMle3<Goldilocks> {
        // Generate `nnz` random (row, col_x) positions and values. We
        // do not deduplicate addresses; the test only needs the
        // sparse representation to be a valid encoding of *some*
        // multilinear function.
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

    fn build_sparse_three_axis(
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

    fn eq_table_for_addresses<F: Field, E: ExtensionField<BaseField = F>>(
        r: &[E],
        addrs: &[usize],
    ) -> Vec<E> {
        addrs.iter().map(|&a| eval_eq_at_index(r, a)).collect()
    }

    #[test]
    fn round_poly_degrees_match_arity() {
        assert_eq!(round_poly_degree(SparseArity::Two), 3);
        assert_eq!(round_poly_degree(SparseArity::Three), 4);
        assert_eq!(product_arity(SparseArity::Two), 3);
        assert_eq!(product_arity(SparseArity::Three), 4);
    }

    #[test]
    fn lagrange_interpolation_round_trips_quartic() {
        // y_i = i^4 - 2i^3 + 3i for i in 0..5
        let mut rng = rng_for("lagrange_quartic");
        let coeffs = [
            GoldilocksExt4::from(7u64),
            GoldilocksExt4::from(11u64),
            GoldilocksExt4::from(13u64),
            -GoldilocksExt4::from(5u64),
            GoldilocksExt4::from(2u64),
        ];
        let f = |x: GoldilocksExt4| -> GoldilocksExt4 {
            let mut acc = GoldilocksExt4::ZERO;
            let mut x_pow = GoldilocksExt4::ONE;
            for c in &coeffs {
                acc += *c * x_pow;
                x_pow *= x;
            }
            acc
        };
        let evals: Vec<GoldilocksExt4> =
            (0..5).map(|i| f(GoldilocksExt4::from(i as u32))).collect();
        // Random evaluation point
        let r = GoldilocksExt4::random_unsafe(&mut rng);
        let interpolated = lagrange_interpolate_at(&evals, r);
        assert_eq!(interpolated, f(r));
    }

    #[test]
    fn eval_sumcheck_two_axis_roundtrip_log_nnz_3() {
        let mut rng = rng_for("eval_sc_two_axis_log3");
        let n_z = 3;
        let n_x = 4;
        let nnz = 8;
        let m = build_sparse_two_axis(&mut rng, n_z, n_x, nnz);

        // Pick a random evaluation point in the extension field.
        let z: Vec<GoldilocksExt3> = (0..n_z)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..n_x)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();

        let v = m.evaluate::<GoldilocksExt3>(&z, &x, &[]);
        let e_z = eq_table_for_addresses::<Goldilocks, GoldilocksExt3>(&z, &m.row);
        let e_x = eq_table_for_addresses::<Goldilocks, GoldilocksExt3>(&x, &m.col_x);

        let mut p_transcript = Sha2T::new();
        let proof = prove_eval_sumcheck::<Goldilocks, GoldilocksExt3>(
            SparseArity::Two,
            &m.val,
            &e_z,
            &e_x,
            None,
            v,
            &mut p_transcript,
        );

        let mut v_transcript = Sha2T::new();
        let claim = verify_eval_sumcheck::<GoldilocksExt3>(
            SparseArity::Two,
            nnz.trailing_zeros() as usize,
            v,
            &proof,
            &mut v_transcript,
        )
        .expect("verifier accepts honest proof");

        assert_eq!(claim.challenges.len(), 3);
        assert_eq!(claim.final_evals.len(), 3);
        assert_eq!(
            claim.final_product,
            claim.final_evals.iter().copied().product()
        );
    }

    #[test]
    fn eval_sumcheck_three_axis_roundtrip_log_nnz_4() {
        let mut rng = rng_for("eval_sc_three_axis_log4");
        let n_z = 3;
        let n_x = 3;
        let n_y = 3;
        let nnz = 16;
        let m = build_sparse_three_axis(&mut rng, n_z, n_x, n_y, nnz);

        let z: Vec<GoldilocksExt4> = (0..n_z)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..n_x)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt4> = (0..n_y)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        let v = m.evaluate::<GoldilocksExt4>(&z, &x, &y);
        let e_z = eq_table_for_addresses::<Goldilocks, GoldilocksExt4>(&z, &m.row);
        let e_x = eq_table_for_addresses::<Goldilocks, GoldilocksExt4>(&x, &m.col_x);
        let e_y = eq_table_for_addresses::<Goldilocks, GoldilocksExt4>(&y, &m.col_y);

        let mut p_transcript = Sha2T::new();
        let proof = prove_eval_sumcheck::<Goldilocks, GoldilocksExt4>(
            SparseArity::Three,
            &m.val,
            &e_z,
            &e_x,
            Some(&e_y),
            v,
            &mut p_transcript,
        );

        let mut v_transcript = Sha2T::new();
        let claim = verify_eval_sumcheck::<GoldilocksExt4>(
            SparseArity::Three,
            nnz.trailing_zeros() as usize,
            v,
            &proof,
            &mut v_transcript,
        )
        .expect("verifier accepts honest proof");

        assert_eq!(claim.challenges.len(), 4);
        assert_eq!(claim.final_evals.len(), 4);
        // The final product must equal the running claim, which (by
        // sumcheck soundness) equals the original claim folded
        // through the round polynomials. We do not check it equals
        // `v` directly — `v` is the *initial* claim and the equality
        // we want is `final_product == g_n(r_n)`, which is enforced
        // by verify_eval_sumcheck before returning Some.
    }

    #[test]
    fn eval_sumcheck_rejects_wrong_initial_claim() {
        let mut rng = rng_for("eval_sc_wrong_v");
        let n_z = 2;
        let n_x = 2;
        let nnz = 4;
        let m = build_sparse_two_axis(&mut rng, n_z, n_x, nnz);

        let z: Vec<GoldilocksExt3> = (0..n_z)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..n_x)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();

        let v = m.evaluate::<GoldilocksExt3>(&z, &x, &[]);
        let wrong_v = v + GoldilocksExt3::ONE;

        let e_z = eq_table_for_addresses::<Goldilocks, GoldilocksExt3>(&z, &m.row);
        let e_x = eq_table_for_addresses::<Goldilocks, GoldilocksExt3>(&x, &m.col_x);

        // The honest prover would refuse to construct this proof
        // since the debug assertion would trip on the first round.
        // Skip prove (or run in release mode); for the verifier
        // rejection test, hand-build a proof from the honest run and
        // hand it to the verifier with the wrong claim.
        let mut p_transcript = Sha2T::new();
        let proof = prove_eval_sumcheck::<Goldilocks, GoldilocksExt3>(
            SparseArity::Two,
            &m.val,
            &e_z,
            &e_x,
            None,
            v,
            &mut p_transcript,
        );

        let mut v_transcript = Sha2T::new();
        let result = verify_eval_sumcheck::<GoldilocksExt3>(
            SparseArity::Two,
            nnz.trailing_zeros() as usize,
            wrong_v,
            &proof,
            &mut v_transcript,
        );
        assert!(
            result.is_none(),
            "verifier must reject when the starting claim does not match the proof"
        );
    }

    #[test]
    fn eval_sumcheck_rejects_tampered_round_message() {
        let mut rng = rng_for("eval_sc_tampered_round");
        let n_z = 2;
        let n_x = 2;
        let nnz = 4;
        let m = build_sparse_two_axis(&mut rng, n_z, n_x, nnz);

        let z: Vec<GoldilocksExt3> = (0..n_z)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..n_x)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();

        let v = m.evaluate::<GoldilocksExt3>(&z, &x, &[]);
        let e_z = eq_table_for_addresses::<Goldilocks, GoldilocksExt3>(&z, &m.row);
        let e_x = eq_table_for_addresses::<Goldilocks, GoldilocksExt3>(&x, &m.col_x);

        let mut p_transcript = Sha2T::new();
        let mut proof = prove_eval_sumcheck::<Goldilocks, GoldilocksExt3>(
            SparseArity::Two,
            &m.val,
            &e_z,
            &e_x,
            None,
            v,
            &mut p_transcript,
        );

        // Corrupt one round message — replace evals[0] with a value
        // that violates the sumcheck invariant g(0) + g(1) = running
        // claim at the first round.
        proof.rounds[0].evals[0] = proof.rounds[0].evals[0] + GoldilocksExt3::ONE;

        let mut v_transcript = Sha2T::new();
        let result = verify_eval_sumcheck::<GoldilocksExt3>(
            SparseArity::Two,
            nnz.trailing_zeros() as usize,
            v,
            &proof,
            &mut v_transcript,
        );
        assert!(result.is_none(), "verifier must reject tampered round 0");
    }

    #[test]
    fn eval_sumcheck_rejects_tampered_final_evals() {
        let mut rng = rng_for("eval_sc_tampered_final");
        let n_z = 3;
        let n_x = 3;
        let n_y = 3;
        let nnz = 8;
        let m = build_sparse_three_axis(&mut rng, n_z, n_x, n_y, nnz);

        let z: Vec<GoldilocksExt4> = (0..n_z)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..n_x)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt4> = (0..n_y)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        let v = m.evaluate::<GoldilocksExt4>(&z, &x, &y);
        let e_z = eq_table_for_addresses::<Goldilocks, GoldilocksExt4>(&z, &m.row);
        let e_x = eq_table_for_addresses::<Goldilocks, GoldilocksExt4>(&x, &m.col_x);
        let e_y = eq_table_for_addresses::<Goldilocks, GoldilocksExt4>(&y, &m.col_y);

        let mut p_transcript = Sha2T::new();
        let mut proof = prove_eval_sumcheck::<Goldilocks, GoldilocksExt4>(
            SparseArity::Three,
            &m.val,
            &e_z,
            &e_x,
            Some(&e_y),
            v,
            &mut p_transcript,
        );

        // Corrupt the final factor evaluations — flipping val(r)
        // breaks the final-product check the verifier performs after
        // all rounds.
        proof.final_evals[0] = proof.final_evals[0] + GoldilocksExt4::ONE;

        let mut v_transcript = Sha2T::new();
        let result = verify_eval_sumcheck::<GoldilocksExt4>(
            SparseArity::Three,
            nnz.trailing_zeros() as usize,
            v,
            &proof,
            &mut v_transcript,
        );
        assert!(
            result.is_none(),
            "verifier must reject tampered final factor evaluations"
        );
    }
}
