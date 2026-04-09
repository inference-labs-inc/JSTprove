//! Per-evaluation base-field WHIR commitment for the
//! `(e_z, e_x, [e_y])` extension-field eq tables, via limb
//! decomposition.
//!
//! Phase 1d-4b (revised): at sparse-MLE open time, the prover
//! materializes the per-axis eq lookup tables `e_z[k] = ẽq(row[k],
//! z)`, `e_x[k] = ẽq(col_x[k], x)`, and (for arity Three)
//! `e_y[k] = ẽq(col_y[k], y)`. These vectors live in the extension
//! field `E` because the outer eval point `(z, x, y)` is in `E`,
//! so they cannot be committed directly with `whir_commit::<F>`
//! (which requires a base-field input).
//!
//! Naïvely committing them with `whir_commit::<E>` produces
//! commitments that *cannot be opened*: `whir_open::<E, EvalF>`
//! requires `EvalF: ExtensionField<BaseField = E>`, which conflicts
//! with the upstream `impl ExtensionField<BaseField = Goldilocks>
//! for GoldilocksExt4`. The two impls cannot coexist on the same
//! type, so the homogeneous WHIR open path is unreachable for our
//! production fields.
//!
//! The fix is **limb decomposition**. Each `E` element decomposes
//! via [`arith::ExtensionField::to_limbs`] into `DEGREE` base-field
//! coefficients
//!
//! ```text
//!     a  =  a_0  +  a_1 · X  +  a_2 · X²  +  …  +  a_{d-1} · X^{d-1}
//! ```
//!
//! where `X = E::X` is the canonical extension generator. For
//! `GoldilocksExt4`, `DEGREE = 4` and the limbs are the four base
//! `Goldilocks` field elements `(a_0, a_1, a_2, a_3)`.
//!
//! By multilinearity of the MLE,
//!
//! ```text
//!     ã(r)  =  Σ_k eq(k, r) · a[k]
//!           =  Σ_k eq(k, r) · (a_0[k] + a_1[k]·X + …)
//!           =  ã_0(r) + ã_1(r) · X + … + ã_{d-1}(r) · X^{d-1}
//! ```
//!
//! so the extension-field MLE evaluation at `r ∈ E^μ` is recoverable
//! from the `DEGREE` base-field MLE evaluations at the same point.
//! The base-field commitments are produced by the standard
//! `whir_commit::<F>` and opened via the standard `whir_open::<F, E>`
//! — both well-trodden code paths.
//!
//! Layout. Each per-eval extension-field constituent (`Ez`, `Ex`,
//! `Ey`) occupies `DEGREE` slots in the combined dense polynomial,
//! one per limb. Slot order is fixed by [`SparseEvalSlot`]: limbs
//! of `Ez` first, then `Ex`, then (for arity Three) `Ey`. Each slot
//! has length `2^μ_eval = nnz`. The slot count is rounded up to a
//! power of two `k_pad`, and the combined polynomial has length
//! `k_pad · 2^μ_eval`. Opening a single limb at sub-point
//! `r ∈ E^μ_eval` is exactly an opening of the combined polynomial
//! at the extended point `(r, idx_bits(slot))`.

use arith::{ExtensionField, FFTField, Field, SimdField};
use serdes::{ExpSerde, SerdeError, SerdeResult};
use tree::{Node, Tree};

use crate::whir::pcs_trait_impl::whir_commit;

use super::types::SparseArity;

/// Constituent enum identifying which extension-field eq table a
/// limb belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvalConstituent {
    Ez,
    Ex,
    /// 3-arity only.
    Ey,
}

impl EvalConstituent {
    /// Canonical ordinal used for slot indexing inside the combined
    /// per-evaluation polynomial.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Ez => 0,
            Self::Ex => 1,
            Self::Ey => 2,
        }
    }
}

/// Slot identifier inside the per-evaluation combined polynomial.
///
/// `(constituent, limb)` is mapped to a flat slot index by
/// `constituent.index() * degree + limb`. Slot 0 holds `Ez` limb 0,
/// slot 1 holds `Ez` limb 1, …, slot `degree` holds `Ex` limb 0, …
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SparseEvalSlot {
    pub constituent: EvalConstituent,
    pub limb: usize,
}

impl SparseEvalSlot {
    /// Flat slot index, given the extension-field degree.
    #[inline]
    #[must_use]
    pub const fn flat_index(&self, degree: usize) -> usize {
        self.constituent.index() * degree + self.limb
    }
}

/// Layout descriptor for the per-evaluation combined polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseEvalLayout {
    pub arity: SparseArity,
    /// Extension-field degree, e.g. 4 for `GoldilocksExt4`.
    pub degree: usize,
    /// Log size of each slot, equal to `⌈log₂ nnz⌉`.
    pub mu_eval: usize,
    /// Number of constituent slots before power-of-two padding.
    pub num_slots: usize,
    /// Number of slots after power-of-two padding.
    pub k_pad: usize,
    pub log_k_pad: usize,
    pub total_vars: usize,
}

impl SparseEvalLayout {
    /// Compute the layout for a sparse-MLE opening with the given
    /// arity, extension-field degree, and sparse-polynomial `nnz`.
    ///
    /// # Panics
    /// Panics if `nnz == 0`, `nnz` is not a power of two, or
    /// `degree == 0`.
    #[must_use]
    pub fn compute(arity: SparseArity, degree: usize, nnz: usize) -> Self {
        assert!(nnz > 0, "SparseEvalLayout::compute: nnz must be > 0");
        assert!(
            nnz.is_power_of_two(),
            "SparseEvalLayout::compute: nnz must be a power of two; got {nnz}"
        );
        assert!(degree > 0, "SparseEvalLayout::compute: degree must be > 0");
        let mu_eval = nnz.trailing_zeros() as usize;
        let num_constituents: usize = match arity {
            SparseArity::Two => 2,
            SparseArity::Three => 3,
        };
        let num_slots = num_constituents * degree;
        let k_pad = num_slots.next_power_of_two();
        let log_k_pad = k_pad.trailing_zeros() as usize;
        Self {
            arity,
            degree,
            mu_eval,
            num_slots,
            k_pad,
            log_k_pad,
            total_vars: mu_eval + log_k_pad,
        }
    }

    #[inline]
    #[must_use]
    pub fn combined_len(&self) -> usize {
        1usize << self.total_vars
    }

    #[inline]
    #[must_use]
    pub fn slot_len(&self) -> usize {
        1usize << self.mu_eval
    }

    /// Byte offset (in field elements) of `slot` in the combined
    /// dense vector.
    #[inline]
    #[must_use]
    pub fn slot_offset(&self, slot: SparseEvalSlot) -> usize {
        slot.flat_index(self.degree) * self.slot_len()
    }
}

/// Public commitment to the per-evaluation `(e_z, e_x, [e_y])`
/// limb-decomposed combined polynomial.
#[derive(Debug, Clone, Default)]
pub struct SparseEvalCommitment {
    pub arity: SparseArity,
    pub degree: usize,
    pub mu_eval: usize,
    pub batched_num_vars: usize,
    pub root: Node,
}

impl ExpSerde for SparseEvalCommitment {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        let arity_tag: u8 = match self.arity {
            SparseArity::Two => 0,
            SparseArity::Three => 1,
        };
        arity_tag.serialize_into(&mut writer)?;
        (self.degree as u64).serialize_into(&mut writer)?;
        (self.mu_eval as u64).serialize_into(&mut writer)?;
        (self.batched_num_vars as u64).serialize_into(&mut writer)?;
        self.root.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let tag = u8::deserialize_from(&mut reader)?;
        let arity = match tag {
            0 => SparseArity::Two,
            1 => SparseArity::Three,
            _ => return Err(SerdeError::DeserializeError),
        };
        let degree = u64::deserialize_from(&mut reader)? as usize;
        let mu_eval = u64::deserialize_from(&mut reader)? as usize;
        let batched_num_vars = u64::deserialize_from(&mut reader)? as usize;
        let root = Node::deserialize_from(&mut reader)?;
        Ok(Self {
            arity,
            degree,
            mu_eval,
            batched_num_vars,
            root,
        })
    }
}

/// Prover scratch for opening a [`SparseEvalCommitment`]. Holds the
/// underlying base-field WHIR Merkle tree + codeword + the dense
/// combined vector so the open phase can call `whir_open::<F, E>`
/// against it without recomputing.
pub struct SparseEvalScratch<F: Field> {
    pub layout: SparseEvalLayout,
    pub combined: Vec<F>,
    pub tree: Tree,
    pub codeword: Vec<F>,
}

/// Commit the per-evaluation `(e_z, e_x, [e_y])` eq tables via
/// limb decomposition.
///
/// Each extension-field input vector is split into `E::DEGREE` base-
/// field component vectors (its limbs in the extension's binomial
/// basis). The limbs are concatenated into a single combined dense
/// polynomial laid out per [`SparseEvalLayout`] and committed via
/// the existing base-field [`whir_commit::<F>`].
///
/// # Panics
/// Panics if input length invariants are violated, if `arity` and
/// the presence/absence of `e_y` disagree, or if `nnz` is not a
/// power of two.
pub fn sparse_commit_eval_tables<F, E>(
    arity: SparseArity,
    e_z: &[E],
    e_x: &[E],
    e_y: Option<&[E]>,
) -> (SparseEvalCommitment, SparseEvalScratch<F>)
where
    F: FFTField + SimdField<Scalar = F>,
    E: ExtensionField<BaseField = F>,
{
    let nnz = e_z.len();
    assert!(nnz > 0, "sparse_commit_eval_tables: nnz must be > 0");
    assert!(
        nnz.is_power_of_two(),
        "sparse_commit_eval_tables: nnz must be a power of two; got {nnz}"
    );
    assert_eq!(e_x.len(), nnz, "e_x length mismatch");
    let three_axis = e_y.is_some();
    assert_eq!(
        three_axis,
        arity == SparseArity::Three,
        "e_y presence must match arity"
    );
    if let Some(e_y_slice) = e_y {
        assert_eq!(e_y_slice.len(), nnz, "e_y length mismatch");
    }

    let degree = E::DEGREE;
    let layout = SparseEvalLayout::compute(arity, degree, nnz);

    let mut combined: Vec<F> = vec![F::ZERO; layout.combined_len()];

    write_constituent_limbs::<F, E>(&mut combined, &layout, EvalConstituent::Ez, e_z);
    write_constituent_limbs::<F, E>(&mut combined, &layout, EvalConstituent::Ex, e_x);
    if let Some(e_y_slice) = e_y {
        write_constituent_limbs::<F, E>(&mut combined, &layout, EvalConstituent::Ey, e_y_slice);
    }

    let (whir_commitment, tree, codeword) = whir_commit::<F>(&combined);
    debug_assert_eq!(whir_commitment.num_vars, layout.total_vars);

    let commitment = SparseEvalCommitment {
        arity,
        degree,
        mu_eval: layout.mu_eval,
        batched_num_vars: layout.total_vars,
        root: whir_commitment.root,
    };
    let scratch = SparseEvalScratch {
        layout,
        combined,
        tree,
        codeword,
    };
    (commitment, scratch)
}

/// Decompose an `E` vector into its `degree` base-field limb vectors
/// and write each limb into the corresponding slot of the combined
/// dense polynomial.
fn write_constituent_limbs<F, E>(
    combined: &mut [F],
    layout: &SparseEvalLayout,
    constituent: EvalConstituent,
    values: &[E],
) where
    F: Field,
    E: ExtensionField<BaseField = F>,
{
    let nnz = values.len();
    let slot_len = layout.slot_len();
    debug_assert!(nnz <= slot_len, "constituent longer than slot");
    for limb in 0..layout.degree {
        let slot = SparseEvalSlot { constituent, limb };
        let offset = layout.slot_offset(slot);
        for (k, v) in values.iter().enumerate() {
            let limbs = v.to_limbs();
            // `to_limbs` is documented to return at least DEGREE
            // entries; if a particular field returns fewer, the
            // missing high limbs are zero by definition.
            let lv = if limb < limbs.len() {
                limbs[limb]
            } else {
                F::ZERO
            };
            combined[offset + k] = lv;
        }
    }
}

/// Build the combined-polynomial eval point for opening `slot` at
/// sub-point `sub_point ∈ E^μ_eval`.
///
/// Mirrors [`super::combined_point::combined_eval_point`] for the
/// per-eval layout. The high `log_k_pad` bits encode the slot's
/// flat index in little-endian binary.
///
/// # Panics
/// Panics if `sub_point.len() != layout.mu_eval`.
#[must_use]
pub fn eval_combined_point<E: Field>(
    layout: &SparseEvalLayout,
    slot: SparseEvalSlot,
    sub_point: &[E],
) -> Vec<E> {
    assert_eq!(
        sub_point.len(),
        layout.mu_eval,
        "eval_combined_point: sub_point length {} does not match layout.mu_eval {}",
        sub_point.len(),
        layout.mu_eval
    );
    let mut point = Vec::with_capacity(layout.total_vars);
    point.extend_from_slice(sub_point);
    let idx = slot.flat_index(layout.degree);
    for bit_pos in 0..layout.log_k_pad {
        let bit = (idx >> bit_pos) & 1;
        if bit == 1 {
            point.push(E::ONE);
        } else {
            point.push(E::ZERO);
        }
    }
    point
}

/// Reconstruct an extension-field MLE evaluation from its per-limb
/// MLE evaluations at the same point.
///
/// Given `limb_evals[i] = ã_i(r)` where `a = a_0 + a_1·X + a_2·X² +
/// …` is the limb decomposition of an extension-field constituent,
/// returns
///
/// ```text
///     ã(r)  =  ã_0(r) + ã_1(r) · X + ã_2(r) · X² + … + ã_{d-1}(r) · X^{d-1}
/// ```
///
/// computed via the closed-form generator multiplication
/// `mul_by_x` so the caller does not need to know the canonical
/// `E::X` constant.
///
/// # Panics
/// Panics if `limb_evals.len() != E::DEGREE`.
#[must_use]
pub fn reconstruct_ext_eval<E: ExtensionField>(limb_evals: &[E]) -> E {
    assert_eq!(
        limb_evals.len(),
        E::DEGREE,
        "reconstruct_ext_eval: expected {} limb evaluations, got {}",
        E::DEGREE,
        limb_evals.len()
    );
    // Horner's method on E::X: result = ((limb_{d-1}·X + limb_{d-2})·X
    // + …)·X + limb_0. Slightly more arithmetic-efficient than the
    // direct sum and avoids any explicit X^k accumulator.
    let mut acc = E::ZERO;
    for i in (0..E::DEGREE).rev() {
        acc = acc.mul_by_x() + limb_evals[i];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::types::eval_eq_at_index;
    use arith::{ExtensionField, Field};
    use goldilocks::{Goldilocks, GoldilocksExt4};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use serdes::ExpSerde;

    fn rng_for(label: &str) -> ChaCha20Rng {
        let mut seed = [0u8; 32];
        let bytes = label.as_bytes();
        let n = bytes.len().min(32);
        seed[..n].copy_from_slice(&bytes[..n]);
        ChaCha20Rng::from_seed(seed)
    }

    #[test]
    fn layout_two_axis_pads_to_eight_slots() {
        // 2 constituents × 4 limbs = 8 slots → k_pad = 8, log_k_pad = 3
        let layout = SparseEvalLayout::compute(SparseArity::Two, 4, 8);
        assert_eq!(layout.degree, 4);
        assert_eq!(layout.num_slots, 8);
        assert_eq!(layout.k_pad, 8);
        assert_eq!(layout.log_k_pad, 3);
        assert_eq!(layout.mu_eval, 3);
        assert_eq!(layout.total_vars, 6);
        assert_eq!(layout.combined_len(), 64);
        assert_eq!(layout.slot_len(), 8);
    }

    #[test]
    fn layout_three_axis_pads_to_sixteen_slots() {
        // 3 constituents × 4 limbs = 12 slots → k_pad = 16, log_k_pad = 4
        let layout = SparseEvalLayout::compute(SparseArity::Three, 4, 16);
        assert_eq!(layout.num_slots, 12);
        assert_eq!(layout.k_pad, 16);
        assert_eq!(layout.log_k_pad, 4);
        assert_eq!(layout.mu_eval, 4);
        assert_eq!(layout.total_vars, 8);
        assert_eq!(layout.combined_len(), 256);
    }

    #[test]
    #[should_panic(expected = "nnz must be a power of two")]
    fn layout_rejects_non_power_of_two_nnz() {
        let _ = SparseEvalLayout::compute(SparseArity::Two, 4, 7);
    }

    #[test]
    fn slot_flat_index_is_constituent_major() {
        // For degree = 4: Ez occupies flat slots 0..4, Ex 4..8, Ey 8..12.
        let degree = 4;
        for limb in 0..4 {
            assert_eq!(
                SparseEvalSlot {
                    constituent: EvalConstituent::Ez,
                    limb,
                }
                .flat_index(degree),
                limb,
            );
            assert_eq!(
                SparseEvalSlot {
                    constituent: EvalConstituent::Ex,
                    limb,
                }
                .flat_index(degree),
                4 + limb,
            );
            assert_eq!(
                SparseEvalSlot {
                    constituent: EvalConstituent::Ey,
                    limb,
                }
                .flat_index(degree),
                8 + limb,
            );
        }
    }

    #[test]
    fn commit_two_axis_round_trips_layout() {
        let mut rng = rng_for("eval_commit_two_axis");
        let nnz = 8;
        let e_z: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_x: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (commitment, scratch) = sparse_commit_eval_tables::<Goldilocks, GoldilocksExt4>(
            SparseArity::Two,
            &e_z,
            &e_x,
            None,
        );
        assert_eq!(commitment.arity, SparseArity::Two);
        assert_eq!(commitment.degree, 4);
        assert_eq!(commitment.mu_eval, 3);
        assert_eq!(commitment.batched_num_vars, 6);
        assert_eq!(scratch.combined.len(), 64);

        // For each constituent, each limb's slot must contain the
        // base-field coefficients of the original ext-field input
        // at that limb position.
        for limb in 0..4 {
            let ez_slot = SparseEvalSlot {
                constituent: EvalConstituent::Ez,
                limb,
            };
            let ez_offset = scratch.layout.slot_offset(ez_slot);
            for k in 0..nnz {
                assert_eq!(scratch.combined[ez_offset + k], e_z[k].to_limbs()[limb]);
            }
            let ex_slot = SparseEvalSlot {
                constituent: EvalConstituent::Ex,
                limb,
            };
            let ex_offset = scratch.layout.slot_offset(ex_slot);
            for k in 0..nnz {
                assert_eq!(scratch.combined[ex_offset + k], e_x[k].to_limbs()[limb]);
            }
        }

        assert!(scratch.codeword.len() >= scratch.combined.len());
        assert!(scratch.codeword.len().is_power_of_two());
    }

    #[test]
    fn commit_three_axis_round_trips_layout() {
        let mut rng = rng_for("eval_commit_three_axis");
        let nnz = 16;
        let e_z: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_x: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_y: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (commitment, scratch) = sparse_commit_eval_tables::<Goldilocks, GoldilocksExt4>(
            SparseArity::Three,
            &e_z,
            &e_x,
            Some(&e_y),
        );
        assert_eq!(commitment.degree, 4);
        assert_eq!(commitment.mu_eval, 4);
        assert_eq!(commitment.batched_num_vars, 8);
        assert_eq!(scratch.combined.len(), 256);
        // num_slots = 12, k_pad = 16, so slots 12..15 are zero padding.
        for slot_idx in 12..16 {
            let off = slot_idx * scratch.layout.slot_len();
            for k in 0..scratch.layout.slot_len() {
                assert_eq!(scratch.combined[off + k], Goldilocks::ZERO);
            }
        }
    }

    #[test]
    #[should_panic(expected = "e_y presence must match arity")]
    fn commit_rejects_arity_e_y_mismatch() {
        let e: Vec<GoldilocksExt4> = vec![GoldilocksExt4::ZERO; 8];
        let _ = sparse_commit_eval_tables::<Goldilocks, GoldilocksExt4>(
            SparseArity::Two,
            &e,
            &e,
            Some(&e),
        );
    }

    #[test]
    fn commit_distinct_inputs_distinct_roots() {
        let mut rng = rng_for("eval_commit_distinct");
        let nnz = 8;
        let e_z_a: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_x_a: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_z_b: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (ca, _) = sparse_commit_eval_tables::<Goldilocks, GoldilocksExt4>(
            SparseArity::Two,
            &e_z_a,
            &e_x_a,
            None,
        );
        let (cb, _) = sparse_commit_eval_tables::<Goldilocks, GoldilocksExt4>(
            SparseArity::Two,
            &e_z_b,
            &e_x_a,
            None,
        );
        assert_ne!(ca.root, cb.root);
    }

    #[test]
    fn commitment_serialization_round_trip() {
        let mut rng = rng_for("eval_commit_serde");
        let e: Vec<GoldilocksExt4> = (0..8)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (commitment, _) =
            sparse_commit_eval_tables::<Goldilocks, GoldilocksExt4>(SparseArity::Two, &e, &e, None);
        let mut bytes = Vec::new();
        commitment.serialize_into(&mut bytes).unwrap();
        let decoded = SparseEvalCommitment::deserialize_from(&bytes[..]).unwrap();
        assert_eq!(decoded.arity, commitment.arity);
        assert_eq!(decoded.degree, commitment.degree);
        assert_eq!(decoded.mu_eval, commitment.mu_eval);
        assert_eq!(decoded.batched_num_vars, commitment.batched_num_vars);
    }

    #[test]
    fn eval_combined_point_layout() {
        let layout = SparseEvalLayout::compute(SparseArity::Three, 4, 8);
        // mu_eval = 3, num_slots = 12, k_pad = 16, log_k_pad = 4, total_vars = 7
        let sub_point: Vec<GoldilocksExt4> = vec![
            GoldilocksExt4::from(2u64),
            GoldilocksExt4::from(3u64),
            GoldilocksExt4::from(5u64),
        ];
        // Slot Ez limb 0 → flat index 0 → idx_bits LE = [0, 0, 0, 0]
        let p = eval_combined_point(
            &layout,
            SparseEvalSlot {
                constituent: EvalConstituent::Ez,
                limb: 0,
            },
            &sub_point,
        );
        assert_eq!(p.len(), 7);
        assert_eq!(&p[..3], sub_point.as_slice());
        for i in 3..7 {
            assert_eq!(p[i], GoldilocksExt4::ZERO);
        }
        // Slot Ey limb 2 → flat index 10 → idx_bits LE = [0, 1, 0, 1]
        let p = eval_combined_point(
            &layout,
            SparseEvalSlot {
                constituent: EvalConstituent::Ey,
                limb: 2,
            },
            &sub_point,
        );
        assert_eq!(p[3], GoldilocksExt4::ZERO);
        assert_eq!(p[4], GoldilocksExt4::ONE);
        assert_eq!(p[5], GoldilocksExt4::ZERO);
        assert_eq!(p[6], GoldilocksExt4::ONE);
    }

    #[test]
    fn reconstruct_ext_eval_round_trips_dense_oracle() {
        // Build a small Ez vector, evaluate it directly as an ext-field
        // MLE at a random point, then evaluate each of its limbs as
        // base-field MLEs at the same point and reconstruct via
        // reconstruct_ext_eval. The two values must agree.
        let mut rng = rng_for("reconstruct_ext_eval");
        let nnz: usize = 16;
        let e_z: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let mu = nnz.trailing_zeros() as usize;
        let r: Vec<GoldilocksExt4> = (0..mu)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();

        // Direct ext-field MLE evaluation
        let mut direct = GoldilocksExt4::ZERO;
        for (k, v) in e_z.iter().enumerate() {
            direct += eval_eq_at_index(&r, k) * *v;
        }

        // Per-limb base-field MLE evaluations
        let mut limb_evals = Vec::with_capacity(4);
        for limb in 0..4 {
            let mut sum = GoldilocksExt4::ZERO;
            for (k, v) in e_z.iter().enumerate() {
                let limbs = v.to_limbs();
                sum += eval_eq_at_index(&r, k) * GoldilocksExt4::from(limbs[limb]);
            }
            limb_evals.push(sum);
        }
        let reconstructed = reconstruct_ext_eval(&limb_evals);
        assert_eq!(reconstructed, direct);
    }
}
