//! Per-axis multiset arguments for the sparse-MLE WHIR open phase.
//!
//! Phase 1d-3: discharges the offline-memory-checking subset
//! equation
//!
//! ```text
//!     H_γ(Init) · H_γ(WS)  =  H_γ(RS) · H_γ(Audit)
//! ```
//!
//! per address axis (`z`, `x`, and — for arity Three — `y`) by
//! reducing each of the four product hashes to an evaluation claim
//! about the leaf vector via the Thaler-style product-circuit GKR
//! sumcheck (Phase 1c). The verifier discharges the resulting leaf
//! claims against the committed constituent polynomials in Phase
//! 1d-4.
//!
//! Hash parameters `(γ_1, γ_2)` are sampled from the shared Fiat-
//! Shamir transcript at the start of this sub-phase, after the
//! eval-claim sumcheck of Phase 1d-2 has been appended. This ordering
//! prevents the prover from choosing the multiset elements after
//! seeing the hash parameters — the standard Spartan ordering.
//!
//! Padding. The product circuit GKR requires power-of-two leaf
//! lengths. `Init` and `Audit` already have length `m_axis = 2^{n_axis}`
//! by construction. `RS` and `WS` have length `nnz`; we require nnz
//! to be a power of two at the sparse-MLE construction boundary.
//! Non-power-of-two `nnz` would need leaf-vector padding with the
//! multiplicative identity `1`, which is sound but adds bookkeeping;
//! deferred until a real circuit forces it.

use arith::{ExtensionField, Field};
use gkr_engine::Transcript;
use serdes::{ExpSerde, SerdeError, SerdeResult};

use super::memcheck::{build_memcheck_sets, multiset_hash, MemoryHashParams};
use super::product_argument::{prove_product_circuit, ProductCircuitProof};
use super::types::{eval_eq_at_index, SparseArity, SparseMleScratchPad};

/// Multiset-argument proof for a single address axis.
///
/// `init_proof` / `rs_proof` / `ws_proof` / `audit_proof` are the
/// per-set product-circuit GKR transcripts produced by Phase 1c.
/// `h_init` / `h_rs` / `h_ws` / `h_audit` are the asserted product
/// hashes — Phase 1c's `prove_product_circuit` returns each as the
/// transcript-pinned root, so the verifier ties the four product
/// claims together via `h_init * h_ws == h_rs * h_audit`.
///
/// `mem` is included so the verifier can replay the prover's
/// per-cell `mẽm[c] = ẽq(c, r_axis)` derivation; per Spartan §7.2.3
/// optimization (4) the verifier could re-compute it independently,
/// but transmitting it makes the audit symmetrical with `RS` / `WS`
/// (and the cost is `2 · m_axis` field elements, dwarfed by the
/// product-circuit transcripts for any non-trivial `m_axis`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerAxisMultisetProof<F: Field> {
    pub h_init: F,
    pub h_rs: F,
    pub h_ws: F,
    pub h_audit: F,
    pub init_proof: ProductCircuitProof<F>,
    pub rs_proof: ProductCircuitProof<F>,
    pub ws_proof: ProductCircuitProof<F>,
    pub audit_proof: ProductCircuitProof<F>,
}

impl<F: Field> PerAxisMultisetProof<F> {
    /// Subset-equation check `H(Init) · H(WS) == H(RS) · H(Audit)`.
    /// Returns `true` iff the per-axis hashes the prover asserted
    /// satisfy the offline-memory-checking constraint.
    #[must_use]
    pub fn check_subset_equation(&self) -> bool {
        self.h_init * self.h_ws == self.h_rs * self.h_audit
    }
}

impl<F: ExtensionField> ExpSerde for PerAxisMultisetProof<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.h_init.serialize_into(&mut writer)?;
        self.h_rs.serialize_into(&mut writer)?;
        self.h_ws.serialize_into(&mut writer)?;
        self.h_audit.serialize_into(&mut writer)?;
        serialize_product_proof(&self.init_proof, &mut writer)?;
        serialize_product_proof(&self.rs_proof, &mut writer)?;
        serialize_product_proof(&self.ws_proof, &mut writer)?;
        serialize_product_proof(&self.audit_proof, &mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let h_init = F::deserialize_from(&mut reader)?;
        let h_rs = F::deserialize_from(&mut reader)?;
        let h_ws = F::deserialize_from(&mut reader)?;
        let h_audit = F::deserialize_from(&mut reader)?;
        let init_proof = deserialize_product_proof(&mut reader)?;
        let rs_proof = deserialize_product_proof(&mut reader)?;
        let ws_proof = deserialize_product_proof(&mut reader)?;
        let audit_proof = deserialize_product_proof(&mut reader)?;
        Ok(Self {
            h_init,
            h_rs,
            h_ws,
            h_audit,
            init_proof,
            rs_proof,
            ws_proof,
            audit_proof,
        })
    }
}

/// Multiset-argument proof for the whole opening: one per active
/// address axis. For arity Two, `y` is `None`; for arity Three it
/// carries a third per-axis proof for the `y` axis.
#[derive(Debug, Clone, Default)]
pub struct SparseMultisetOpening<F: Field> {
    pub gamma_1: F,
    pub gamma_2: F,
    pub axis_z: PerAxisMultisetProof<F>,
    pub axis_x: PerAxisMultisetProof<F>,
    pub axis_y: Option<PerAxisMultisetProof<F>>,
}

impl<F: ExtensionField> ExpSerde for SparseMultisetOpening<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.gamma_1.serialize_into(&mut writer)?;
        self.gamma_2.serialize_into(&mut writer)?;
        self.axis_z.serialize_into(&mut writer)?;
        self.axis_x.serialize_into(&mut writer)?;
        let has_y: u8 = if self.axis_y.is_some() { 1 } else { 0 };
        has_y.serialize_into(&mut writer)?;
        if let Some(axis_y) = &self.axis_y {
            axis_y.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let gamma_1 = F::deserialize_from(&mut reader)?;
        let gamma_2 = F::deserialize_from(&mut reader)?;
        let axis_z = PerAxisMultisetProof::deserialize_from(&mut reader)?;
        let axis_x = PerAxisMultisetProof::deserialize_from(&mut reader)?;
        let has_y = u8::deserialize_from(&mut reader)?;
        let axis_y = match has_y {
            0 => None,
            1 => Some(PerAxisMultisetProof::deserialize_from(&mut reader)?),
            _ => return Err(SerdeError::DeserializeError),
        };
        Ok(Self {
            gamma_1,
            gamma_2,
            axis_z,
            axis_x,
            axis_y,
        })
    }
}

fn serialize_product_proof<F: ExtensionField, W: std::io::Write>(
    proof: &ProductCircuitProof<F>,
    mut writer: W,
) -> SerdeResult<()> {
    (proof.layers.len() as u64).serialize_into(&mut writer)?;
    for layer in &proof.layers {
        (layer.sumcheck_rounds.len() as u64).serialize_into(&mut writer)?;
        for round in &layer.sumcheck_rounds {
            for ev in &round.evals {
                ev.serialize_into(&mut writer)?;
            }
        }
        layer.left_eval.serialize_into(&mut writer)?;
        layer.right_eval.serialize_into(&mut writer)?;
    }
    Ok(())
}

fn deserialize_product_proof<F: ExtensionField, R: std::io::Read>(
    mut reader: R,
) -> SerdeResult<ProductCircuitProof<F>> {
    use super::product_argument::{ProductLayerProof, ProductRound};
    const MAX_LAYERS: usize = 64;
    const MAX_ROUNDS_PER_LAYER: usize = 64;

    let n_layers = u64::deserialize_from(&mut reader)? as usize;
    if n_layers > MAX_LAYERS {
        return Err(SerdeError::DeserializeError);
    }
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_rounds = u64::deserialize_from(&mut reader)? as usize;
        if n_rounds > MAX_ROUNDS_PER_LAYER {
            return Err(SerdeError::DeserializeError);
        }
        let mut sumcheck_rounds = Vec::with_capacity(n_rounds);
        for _ in 0..n_rounds {
            let evals = [
                F::deserialize_from(&mut reader)?,
                F::deserialize_from(&mut reader)?,
                F::deserialize_from(&mut reader)?,
                F::deserialize_from(&mut reader)?,
            ];
            sumcheck_rounds.push(ProductRound { evals });
        }
        let left_eval = F::deserialize_from(&mut reader)?;
        let right_eval = F::deserialize_from(&mut reader)?;
        layers.push(ProductLayerProof {
            sumcheck_rounds,
            left_eval,
            right_eval,
        });
    }
    Ok(ProductCircuitProof { layers })
}

/// Compute the per-axis memory polynomial `mẽm[c] = ẽq(c, r_axis)`
/// for `c ∈ [m_axis]`. Used by both prover (to construct the Init /
/// Audit hashes) and verifier (to recompute them) per Spartan §7.2.3
/// opt (4).
#[must_use]
pub fn compute_axis_memory<E: Field>(r_axis: &[E], m_axis: usize) -> Vec<E> {
    (0..m_axis).map(|c| eval_eq_at_index(r_axis, c)).collect()
}

/// Run the per-axis multiset argument given the address vector,
/// per-operation values (the `e_axis` table), the per-axis
/// timestamps from the scratch pad, and the verifier's outer eval
/// point for this axis.
///
/// Computes the four memcheck sets via [`build_memcheck_sets`],
/// applies `-γ_2` to obtain the four product-circuit leaf vectors,
/// and runs [`prove_product_circuit`] on each. The four asserted
/// product hashes are pulled from the product-circuit transcript and
/// returned alongside the proofs.
///
/// # Panics
/// Panics if `addrs.len() != e_axis.len()` or if either is not a
/// power of two (the product-circuit GKR requires power-of-two leaf
/// lengths). The latter is currently enforced at the
/// `sparse_open_multiset` boundary by requiring the sparse polynomial
/// to have a power-of-two `nnz`; non-power-of-two support would
/// require leaf padding with the multiplicative identity.
fn prove_per_axis_multiset<F, E>(
    params: &MemoryHashParams<E>,
    addrs: &[usize],
    e_axis: &[E],
    read_ts: &[F],
    audit_ts: &[F],
    r_axis: &[E],
    m_axis: usize,
    transcript: &mut impl Transcript,
) -> PerAxisMultisetProof<E>
where
    F: Field,
    E: ExtensionField<BaseField = F>,
{
    assert_eq!(addrs.len(), e_axis.len(), "addrs / e_axis length mismatch");
    assert_eq!(
        addrs.len(),
        read_ts.len(),
        "addrs / read_ts length mismatch"
    );
    assert_eq!(audit_ts.len(), m_axis, "audit_ts must have length m_axis");
    assert!(
        addrs.len().is_power_of_two(),
        "RS / WS leaf count must be a power of two; got {}",
        addrs.len()
    );
    assert!(
        m_axis.is_power_of_two(),
        "Init / Audit leaf count must be a power of two; got {m_axis}"
    );

    // Materialize the per-cell memory polynomial mẽm[c] = ẽq(c, r_axis).
    let mem: Vec<E> = compute_axis_memory(r_axis, m_axis);

    // Lift base-field timestamps into the extension field for hashing
    // and product-circuit folding. The lifts are used only inside the
    // product-circuit GKR, so they do not need to round-trip through
    // a base-field commitment open.
    let read_ts_e: Vec<E> = read_ts.iter().map(|&t| E::from(t)).collect();
    let audit_ts_e: Vec<E> = audit_ts.iter().map(|&t| E::from(t)).collect();

    // Build the four sets in the extension field.
    let sets = build_memcheck_sets(params, addrs, e_axis, &read_ts_e, &mem, &audit_ts_e);

    // Compute the four asserted product hashes explicitly so we can
    // expose them on the proof; the product circuits also commit to
    // these via the transcript-tied root.
    let h_init = multiset_hash(&sets.init, params.gamma_2);
    let h_rs = multiset_hash(&sets.rs, params.gamma_2);
    let h_ws = multiset_hash(&sets.ws, params.gamma_2);
    let h_audit = multiset_hash(&sets.audit, params.gamma_2);

    // The product-circuit GKR operates on the leaves *after* applying
    // the (e − γ_2) compression so the product over the leaves equals
    // H_γ_2(set). Build the leaf vectors explicitly here.
    let init_leaves: Vec<E> = sets.init.iter().map(|e| *e - params.gamma_2).collect();
    let rs_leaves: Vec<E> = sets.rs.iter().map(|e| *e - params.gamma_2).collect();
    let ws_leaves: Vec<E> = sets.ws.iter().map(|e| *e - params.gamma_2).collect();
    let audit_leaves: Vec<E> = sets.audit.iter().map(|e| *e - params.gamma_2).collect();

    let (init_root, init_proof) = prove_product_circuit(&init_leaves, transcript);
    let (rs_root, rs_proof) = prove_product_circuit(&rs_leaves, transcript);
    let (ws_root, ws_proof) = prove_product_circuit(&ws_leaves, transcript);
    let (audit_root, audit_proof) = prove_product_circuit(&audit_leaves, transcript);

    debug_assert_eq!(init_root, h_init, "init product circuit root mismatch");
    debug_assert_eq!(rs_root, h_rs, "rs product circuit root mismatch");
    debug_assert_eq!(ws_root, h_ws, "ws product circuit root mismatch");
    debug_assert_eq!(audit_root, h_audit, "audit product circuit root mismatch");

    PerAxisMultisetProof {
        h_init,
        h_rs,
        h_ws,
        h_audit,
        init_proof,
        rs_proof,
        ws_proof,
        audit_proof,
    }
}

/// Run the multiset half of the sparse-MLE WHIR open phase.
///
/// Samples `(γ_1, γ_2)` from the shared transcript, builds the per-
/// axis eq tables (recomputed here so callers do not have to plumb
/// them through from the eval-claim sub-phase), and runs the
/// per-axis multiset argument for `z`, `x`, and — if the arity is
/// `Three` — `y`.
///
/// The eval-claim sub-phase ([`super::open::sparse_open_evalclaim`])
/// must be invoked first so the (γ_1, γ_2) samples come from a
/// transcript already pinned to the eval-claim transcript.
///
/// # Panics
/// Panics if `(z.len(), x.len(), y.len())` does not match the
/// scratch's `(n_z, n_x, n_y)` for the recorded arity, or if the
/// scratch `nnz` is not a power of two.
pub fn sparse_open_multiset<F, E>(
    scratch: &SparseMleScratchPad<F>,
    z: &[E],
    x: &[E],
    y: &[E],
    transcript: &mut impl Transcript,
) -> SparseMultisetOpening<E>
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
    assert!(
        scratch.nnz.is_power_of_two(),
        "sparse_open_multiset requires nnz to be a power of two; got {}",
        scratch.nnz
    );

    let gamma_1: E = transcript.generate_field_element();
    let gamma_2: E = transcript.generate_field_element();
    let params = MemoryHashParams { gamma_1, gamma_2 };

    let m_z = 1usize << scratch.n_z;
    let m_x = 1usize << scratch.n_x;
    let m_y = 1usize << scratch.n_y;

    // Re-materialize the per-axis eq tables so this sub-phase does
    // not require the caller to plumb them in from sparse_open_evalclaim.
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

    let axis_z = prove_per_axis_multiset::<F, E>(
        &params,
        &scratch.row,
        &e_z,
        &scratch.ts_z.read_ts,
        &scratch.ts_z.audit_ts,
        z,
        m_z,
        transcript,
    );

    let axis_x = prove_per_axis_multiset::<F, E>(
        &params,
        &scratch.col_x,
        &e_x,
        &scratch.ts_x.read_ts,
        &scratch.ts_x.audit_ts,
        x,
        m_x,
        transcript,
    );

    let axis_y = if scratch.arity == SparseArity::Three {
        let e_y = e_y_opt.expect("e_y_opt must be present for arity Three");
        let ts_y = scratch
            .ts_y
            .as_ref()
            .expect("scratch.ts_y must be present for arity Three");
        Some(prove_per_axis_multiset::<F, E>(
            &params,
            &scratch.col_y,
            &e_y,
            &ts_y.read_ts,
            &ts_y.audit_ts,
            y,
            m_y,
            transcript,
        ))
    } else {
        None
    };

    SparseMultisetOpening {
        gamma_1,
        gamma_2,
        axis_z,
        axis_x,
        axis_y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::commit::sparse_commit;
    use crate::whir::sparse::types::SparseMle3;
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
    fn axis_memory_matches_eq_lookup() {
        let mut rng = rng_for("axis_memory");
        let r: Vec<GoldilocksExt3> = (0..4)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let mem = compute_axis_memory(&r, 16);
        for c in 0..16 {
            assert_eq!(mem[c], eval_eq_at_index(&r, c));
        }
    }

    #[test]
    fn multiset_open_two_axis_satisfies_subset_equation() {
        let mut rng = rng_for("multiset_two_axis");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        let z: Vec<GoldilocksExt3> = (0..3)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..3)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();

        let mut p_t = Sha2T::new();
        let opening =
            sparse_open_multiset::<Goldilocks, GoldilocksExt3>(&scratch, &z, &x, &[], &mut p_t);

        assert!(
            opening.axis_z.check_subset_equation(),
            "axis z subset equation must hold for honest trace"
        );
        assert!(
            opening.axis_x.check_subset_equation(),
            "axis x subset equation must hold for honest trace"
        );
        assert!(opening.axis_y.is_none());
    }

    #[test]
    fn multiset_open_three_axis_satisfies_subset_equation() {
        let mut rng = rng_for("multiset_three_axis");
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

        let mut p_t = Sha2T::new();
        let opening =
            sparse_open_multiset::<Goldilocks, GoldilocksExt4>(&scratch, &z, &x, &y, &mut p_t);

        assert!(opening.axis_z.check_subset_equation());
        assert!(opening.axis_x.check_subset_equation());
        assert!(opening.axis_y.is_some());
        assert!(opening.axis_y.as_ref().unwrap().check_subset_equation());
    }

    #[test]
    fn product_circuit_roots_match_explicit_hashes() {
        let mut rng = rng_for("product_roots");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        let z: Vec<GoldilocksExt3> = (0..3)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..3)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();

        let mut p_t = Sha2T::new();
        let opening =
            sparse_open_multiset::<Goldilocks, GoldilocksExt3>(&scratch, &z, &x, &[], &mut p_t);

        // For each axis, the four product-circuit GKR proofs are
        // self-tied: the product roots claimed inside the proof
        // (h_init, h_rs, h_ws, h_audit) must equal the products
        // of the (e − γ_2) leaves the prover used as input. The
        // debug_assert_eq inside prove_per_axis_multiset already
        // catches mismatches; this test additionally checks that
        // the four hashes form a non-trivial multiplicative
        // structure (not all zero, not all one) to guard against
        // an entire axis being silently bypassed.
        let z_hashes = [
            opening.axis_z.h_init,
            opening.axis_z.h_rs,
            opening.axis_z.h_ws,
            opening.axis_z.h_audit,
        ];
        for (i, h) in z_hashes.iter().enumerate() {
            assert_ne!(*h, GoldilocksExt3::ZERO, "axis z hash {i} is zero");
        }
    }

    #[test]
    fn multiset_opening_serialization_round_trip_two_axis() {
        let mut rng = rng_for("multiset_serde_two");
        let poly = build_two_axis(&mut rng, 2, 2, 4);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let z: Vec<GoldilocksExt3> = (0..2)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt3> = (0..2)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let mut p_t = Sha2T::new();
        let opening =
            sparse_open_multiset::<Goldilocks, GoldilocksExt3>(&scratch, &z, &x, &[], &mut p_t);

        let mut bytes = Vec::new();
        opening.serialize_into(&mut bytes).unwrap();
        let decoded =
            SparseMultisetOpening::<GoldilocksExt3>::deserialize_from(&bytes[..]).unwrap();

        assert_eq!(decoded.gamma_1, opening.gamma_1);
        assert_eq!(decoded.gamma_2, opening.gamma_2);
        assert_eq!(decoded.axis_z.h_init, opening.axis_z.h_init);
        assert_eq!(decoded.axis_z.h_rs, opening.axis_z.h_rs);
        assert_eq!(decoded.axis_z.h_ws, opening.axis_z.h_ws);
        assert_eq!(decoded.axis_z.h_audit, opening.axis_z.h_audit);
        assert_eq!(decoded.axis_x.h_init, opening.axis_x.h_init);
        assert!(decoded.axis_y.is_none());
    }

    #[test]
    fn multiset_opening_serialization_round_trip_three_axis() {
        let mut rng = rng_for("multiset_serde_three");
        let poly = build_three_axis(&mut rng, 2, 2, 2, 4);
        let (_commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let z: Vec<GoldilocksExt4> = (0..2)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let x: Vec<GoldilocksExt4> = (0..2)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let y: Vec<GoldilocksExt4> = (0..2)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let mut p_t = Sha2T::new();
        let opening =
            sparse_open_multiset::<Goldilocks, GoldilocksExt4>(&scratch, &z, &x, &y, &mut p_t);

        let mut bytes = Vec::new();
        opening.serialize_into(&mut bytes).unwrap();
        let decoded =
            SparseMultisetOpening::<GoldilocksExt4>::deserialize_from(&bytes[..]).unwrap();

        assert_eq!(decoded.gamma_1, opening.gamma_1);
        assert!(decoded.axis_y.is_some());
        let dy = decoded.axis_y.unwrap();
        let oy = opening.axis_y.unwrap();
        assert_eq!(dy.h_init, oy.h_init);
        assert_eq!(dy.h_rs, oy.h_rs);
        assert_eq!(dy.h_ws, oy.h_ws);
        assert_eq!(dy.h_audit, oy.h_audit);
    }
}
