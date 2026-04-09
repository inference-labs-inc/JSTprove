//! Per-evaluation extension-field WHIR commitment for the
//! `(e_z, e_x, [e_y])` eq tables.
//!
//! Phase 1d-4b: at sparse-MLE open time, the prover materializes the
//! per-axis eq lookup tables `e_z[k] = ẽq(row[k], z)`,
//! `e_x[k] = ẽq(col_x[k], x)`, and (for arity Three) `e_y[k] =
//! ẽq(col_y[k], y)`. These vectors live in the extension field `E`
//! because the outer eval point `(z, x, y)` is in `E`, so they
//! cannot be folded into the base-field commitment produced by Phase
//! 1d-1 (which lives in the VK and depends only on the circuit, not
//! on any per-inference eval point).
//!
//! This module provides a fresh extension-field WHIR commitment for
//! those tables: it lays them out into a combined dense polynomial
//! over `E`, runs `whir_commit::<E>` on the result, and returns a
//! `SparseEvalCommitment` carrying the Merkle root + the scratch
//! the prover needs to open the per-eval commitment at the points
//! produced by the eval-claim sumcheck and the per-axis multiset
//! arguments.
//!
//! The per-eval commitment root is included in the
//! `SparseMle3Opening` (Phase 1d-4c) so the verifier can replay it
//! as part of the proof transcript. It is *not* part of the
//! verifying key — it's freshly produced per opening, just like
//! Spartan §7.2's commitment to `e_row` and `e_col` at evaluation
//! time.
//!
//! Layout. The combined eval polynomial uses the same
//! [`SparseLayout`] geometry as the setup commitment, but with only
//! three slots active (`EZ`, `EX`, `EY`) and `μ_eval = ⌈log₂ nnz⌉`
//! since the eq tables are length-`nnz` only. Slot identifiers are
//! defined separately from the setup `SparseConstituentSlot` so the
//! per-eval layout can evolve independently of the VK.
//!
//! Field constraint. The underlying [`whir_commit`] requires
//! `LEAF_BYTES (= 64) % F::SIZE == 0` so that field elements pack
//! cleanly into Merkle leaves without alignment padding. For the
//! Goldilocks family this is satisfied by `Goldilocks` (8 B),
//! `GoldilocksExt2` (16 B), and `GoldilocksExt4` (32 B), but not by
//! `GoldilocksExt4` (24 B). Phase 1d-4b therefore uses
//! `GoldilocksExt4` in its tests; the Ext3 path will require a
//! WHIR leaf-packing extension that supports non-divisible element
//! sizes via padding (the open / verify halves of WHIR already
//! handle this through `build_tree_from_ext_codeword`'s padded
//! branch — only the commit path needs to be brought into line).
//! Tracked as a follow-up in the cross-repo impact summary.

use arith::{ExtensionField, FFTField, Field, SimdField};
use serdes::{ExpSerde, SerdeError, SerdeResult};
use tree::{Node, Tree};

use crate::whir::pcs_trait_impl::whir_commit;

use super::types::SparseArity;

/// Slot identifier inside the per-eval extension-field commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum SparseEvalSlot {
    Ez = 0,
    Ex = 1,
    /// 3-arity only.
    Ey = 2,
}

impl SparseEvalSlot {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Layout descriptor for the per-eval combined polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseEvalLayout {
    pub arity: SparseArity,
    /// Log size of each slot, equal to `⌈log₂ nnz⌉`.
    pub mu_eval: usize,
    /// Number of slots after padding to a power of two. For arity
    /// Two: 2 (Ez, Ex). For arity Three: 4 (Ez, Ex, Ey + 1 padding).
    pub k_pad: usize,
    pub log_k_pad: usize,
    /// Total number of variables in the combined polynomial.
    pub total_vars: usize,
}

impl SparseEvalLayout {
    /// Compute the layout for a sparse-MLE opening with the given
    /// arity and `nnz`.
    ///
    /// # Panics
    /// Panics if `nnz == 0` or `nnz` is not a power of two.
    #[must_use]
    pub fn compute(arity: SparseArity, nnz: usize) -> Self {
        assert!(nnz > 0, "SparseEvalLayout::compute: nnz must be > 0");
        assert!(
            nnz.is_power_of_two(),
            "SparseEvalLayout::compute: nnz must be a power of two; got {nnz}"
        );
        let mu_eval = nnz.trailing_zeros() as usize;
        let num_slots: usize = match arity {
            SparseArity::Two => 2,
            SparseArity::Three => 3,
        };
        let k_pad = num_slots.next_power_of_two();
        let log_k_pad = k_pad.trailing_zeros() as usize;
        Self {
            arity,
            mu_eval,
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

    #[inline]
    #[must_use]
    pub fn slot_offset(&self, slot: SparseEvalSlot) -> usize {
        slot.index() * self.slot_len()
    }
}

/// Public commitment to the per-eval `(e_z, e_x, [e_y])` tables.
/// Carries the WHIR Merkle root, the layout metadata, and the
/// arity. Lives inside [`super::types::SparseMle3Opening`] (or its
/// successor) — never in the verifying key.
#[derive(Debug, Clone, Default)]
pub struct SparseEvalCommitment {
    pub arity: SparseArity,
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
        let mu_eval = u64::deserialize_from(&mut reader)? as usize;
        let batched_num_vars = u64::deserialize_from(&mut reader)? as usize;
        let root = Node::deserialize_from(&mut reader)?;
        Ok(Self {
            arity,
            mu_eval,
            batched_num_vars,
            root,
        })
    }
}

/// Prover scratch for opening a [`SparseEvalCommitment`]. Holds the
/// underlying WHIR Merkle tree + codeword + the dense combined
/// vector so the open phase can call `whir_open` against it without
/// recomputing.
pub struct SparseEvalScratch<E: Field> {
    pub layout: SparseEvalLayout,
    pub combined: Vec<E>,
    pub tree: Tree,
    pub codeword: Vec<E>,
}

/// Commit the per-eval `(e_z, e_x, [e_y])` tables.
///
/// `e_y` must be `Some(_)` iff `arity == SparseArity::Three`. All
/// three input slices must have length `nnz` and `nnz` must be a
/// power of two (the open phase enforces this for `sparse_open` as
/// a whole).
///
/// Returns the public commitment + the prover scratch the open
/// routine consumes when calling `whir_open` against this
/// commitment at the post-sumcheck random points.
///
/// # Panics
/// Panics if input length invariants are violated or if `arity` and
/// the presence/absence of `e_y` disagree.
pub fn sparse_commit_eval_tables<E>(
    arity: SparseArity,
    e_z: &[E],
    e_x: &[E],
    e_y: Option<&[E]>,
) -> (SparseEvalCommitment, SparseEvalScratch<E>)
where
    E: FFTField + SimdField<Scalar = E> + ExtensionField,
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

    let layout = SparseEvalLayout::compute(arity, nnz);
    let mut combined: Vec<E> = vec![E::ZERO; layout.combined_len()];

    let ez_offset = layout.slot_offset(SparseEvalSlot::Ez);
    combined[ez_offset..ez_offset + nnz].copy_from_slice(e_z);
    let ex_offset = layout.slot_offset(SparseEvalSlot::Ex);
    combined[ex_offset..ex_offset + nnz].copy_from_slice(e_x);
    if let Some(e_y_slice) = e_y {
        let ey_offset = layout.slot_offset(SparseEvalSlot::Ey);
        combined[ey_offset..ey_offset + nnz].copy_from_slice(e_y_slice);
    }

    let (whir_commitment, tree, codeword) = whir_commit::<E>(&combined);
    debug_assert_eq!(whir_commitment.num_vars, layout.total_vars);

    let commitment = SparseEvalCommitment {
        arity,
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

/// Build the combined-polynomial eval point inside the per-eval
/// commitment for opening `slot` at sub-point `sub_point ∈ E^μ_eval`.
///
/// Mirrors [`super::combined_point::combined_eval_point`] but for
/// the per-eval layout. The two layouts are intentionally distinct
/// so the per-eval commitment slot count and slot identifiers can
/// evolve independently of the VK layout.
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
    let idx = slot.index();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::types::eval_eq_at_index;
    use arith::Field;
    use goldilocks::GoldilocksExt4;
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
    fn layout_two_axis_pads_to_two_slots() {
        let layout = SparseEvalLayout::compute(SparseArity::Two, 8);
        assert_eq!(layout.k_pad, 2);
        assert_eq!(layout.log_k_pad, 1);
        assert_eq!(layout.mu_eval, 3);
        assert_eq!(layout.total_vars, 4);
        assert_eq!(layout.combined_len(), 16);
        assert_eq!(layout.slot_len(), 8);
    }

    #[test]
    fn layout_three_axis_pads_to_four_slots() {
        let layout = SparseEvalLayout::compute(SparseArity::Three, 16);
        assert_eq!(layout.k_pad, 4);
        assert_eq!(layout.log_k_pad, 2);
        assert_eq!(layout.mu_eval, 4);
        assert_eq!(layout.total_vars, 6);
        assert_eq!(layout.combined_len(), 64);
    }

    #[test]
    #[should_panic(expected = "nnz must be a power of two")]
    fn layout_rejects_non_power_of_two_nnz() {
        let _ = SparseEvalLayout::compute(SparseArity::Two, 7);
    }

    #[test]
    fn commit_eval_tables_two_axis_round_trips_layout() {
        let mut rng = rng_for("eval_commit_two_axis");
        let nnz = 8;
        let e_z: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_x: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (commitment, scratch) =
            sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Two, &e_z, &e_x, None);
        assert_eq!(commitment.arity, SparseArity::Two);
        assert_eq!(commitment.mu_eval, 3);
        assert_eq!(commitment.batched_num_vars, 4);
        assert_eq!(scratch.layout.total_vars, 4);
        assert_eq!(scratch.combined.len(), 16);
        // Slots populated correctly
        assert_eq!(&scratch.combined[..nnz], &e_z[..]);
        assert_eq!(&scratch.combined[nnz..2 * nnz], &e_x[..]);
        // Codeword is at least the combined dense length
        assert!(scratch.codeword.len() >= scratch.combined.len());
        assert!(scratch.codeword.len().is_power_of_two());
    }

    #[test]
    fn commit_eval_tables_three_axis_round_trips_layout() {
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
        let (commitment, scratch) =
            sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Three, &e_z, &e_x, Some(&e_y));
        assert_eq!(commitment.arity, SparseArity::Three);
        assert_eq!(commitment.mu_eval, 4);
        assert_eq!(commitment.batched_num_vars, 6);
        assert_eq!(scratch.combined.len(), 64);
        assert_eq!(&scratch.combined[..nnz], &e_z[..]);
        assert_eq!(&scratch.combined[nnz..2 * nnz], &e_x[..]);
        assert_eq!(&scratch.combined[2 * nnz..3 * nnz], &e_y[..]);
        // Slot 3 (padding) is all zeros
        for v in &scratch.combined[3 * nnz..4 * nnz] {
            assert_eq!(*v, GoldilocksExt4::ZERO);
        }
    }

    #[test]
    #[should_panic(expected = "e_y presence must match arity")]
    fn commit_eval_tables_rejects_arity_e_y_mismatch() {
        let e: Vec<GoldilocksExt4> = vec![GoldilocksExt4::ZERO; 8];
        let _ = sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Two, &e, &e, Some(&e));
    }

    #[test]
    fn commit_eval_tables_distinct_inputs_distinct_roots() {
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
        let (ca, _) =
            sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Two, &e_z_a, &e_x_a, None);
        let (cb, _) =
            sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Two, &e_z_b, &e_x_a, None);
        assert_ne!(ca.root, cb.root);
    }

    #[test]
    fn commitment_serialization_round_trip() {
        let mut rng = rng_for("eval_commit_serde");
        let e: Vec<GoldilocksExt4> = (0..8)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (commitment, _) =
            sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Two, &e, &e, None);
        let mut bytes = Vec::new();
        commitment.serialize_into(&mut bytes).unwrap();
        let decoded = SparseEvalCommitment::deserialize_from(&bytes[..]).unwrap();
        assert_eq!(decoded.arity, commitment.arity);
        assert_eq!(decoded.mu_eval, commitment.mu_eval);
        assert_eq!(decoded.batched_num_vars, commitment.batched_num_vars);
    }

    #[test]
    fn eval_combined_point_layout() {
        let layout = SparseEvalLayout::compute(SparseArity::Three, 8);
        // mu_eval = 3, k_pad = 4, log_k_pad = 2, total_vars = 5
        let sub_point: Vec<GoldilocksExt4> = vec![
            GoldilocksExt4::from(2u64),
            GoldilocksExt4::from(3u64),
            GoldilocksExt4::from(5u64),
        ];
        let p_ez = eval_combined_point(&layout, SparseEvalSlot::Ez, &sub_point);
        assert_eq!(p_ez.len(), 5);
        assert_eq!(&p_ez[..3], sub_point.as_slice());
        assert_eq!(p_ez[3], GoldilocksExt4::ZERO);
        assert_eq!(p_ez[4], GoldilocksExt4::ZERO);

        let p_ey = eval_combined_point(&layout, SparseEvalSlot::Ey, &sub_point);
        // Slot 2 → idx_bits LE = [0, 1]
        assert_eq!(p_ey[3], GoldilocksExt4::ZERO);
        assert_eq!(p_ey[4], GoldilocksExt4::ONE);
    }

    #[test]
    fn eval_combined_point_evaluation_matches_constituent() {
        let mut rng = rng_for("eval_combined_point_oracle");
        let nnz = 8;
        let e_z: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let e_x: Vec<GoldilocksExt4> = (0..nnz)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        let (_commitment, scratch) =
            sparse_commit_eval_tables::<GoldilocksExt4>(SparseArity::Two, &e_z, &e_x, None);
        let sub_point: Vec<GoldilocksExt4> = (0..scratch.layout.mu_eval)
            .map(|_| GoldilocksExt4::random_unsafe(&mut rng))
            .collect();
        // Evaluate e_z (the constituent) at sub_point directly
        let mut expected = GoldilocksExt4::ZERO;
        for (i, v) in e_z.iter().enumerate() {
            expected += eval_eq_at_index(&sub_point, i) * *v;
        }
        // Evaluate combined at eval_combined_point(Ez, sub_point)
        let combined_point = eval_combined_point(&scratch.layout, SparseEvalSlot::Ez, &sub_point);
        let mut combined_eval = GoldilocksExt4::ZERO;
        for (b, v) in scratch.combined.iter().enumerate() {
            combined_eval += eval_eq_at_index(&combined_point, b) * *v;
        }
        assert_eq!(expected, combined_eval);
    }
}
