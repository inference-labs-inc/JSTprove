//! Combined-polynomial eval-point arithmetic for the sparse-MLE
//! WHIR commitment.
//!
//! Phase 1d-1 packs the constituent vectors `(val, row, col_x, …,
//! audit_ts_y)` into a single dense polynomial via Spartan §7.2.3
//! optimization (5): each constituent occupies a fixed slice of the
//! combined dense vector indexed by a power-of-two slot identifier
//! `i ∈ [k_pad]`. The low `μ = combined.batched_num_vars −
//! log₂ k_pad` variables index *into* a constituent, the high `log₂
//! k_pad` variables select *which* constituent.
//!
//! Opening constituent `i` at a sub-point `r ∈ E^μ` is therefore
//! equivalent to opening the combined polynomial at the extended
//! point
//!
//! ```text
//!     point = (r ‖ idx_bits(i))
//! ```
//!
//! where `idx_bits(i)` is the little-endian binary expansion of `i`
//! in `log₂ k_pad` bits. This module defines that translation as a
//! pure helper so the open / verify phases share a single source of
//! truth.

use arith::Field;

use super::commit::{SparseConstituentSlot, SparseLayout};

/// Build the combined-polynomial eval point for opening
/// `constituent` at the sub-point `sub_point ∈ E^μ`.
///
/// Returns a vector of length `layout.total_vars = μ + log₂ k_pad`
/// where the first `μ` entries are `sub_point` (in order, low bits
/// first) and the trailing `log₂ k_pad` entries encode the
/// constituent slot index in little-endian binary as `0`s and `1`s.
///
/// # Panics
/// Panics if `sub_point.len() != layout.mu`.
#[must_use]
pub fn combined_eval_point<E: Field>(
    layout: &SparseLayout,
    constituent: SparseConstituentSlot,
    sub_point: &[E],
) -> Vec<E> {
    assert_eq!(
        sub_point.len(),
        layout.mu,
        "combined_eval_point: sub_point length {} does not match layout.mu {}",
        sub_point.len(),
        layout.mu
    );
    let mut point = Vec::with_capacity(layout.total_vars);
    point.extend_from_slice(sub_point);
    let idx = constituent.index();
    assert!(
        idx < (1 << layout.log_k_pad),
        "combined_eval_point: constituent index {idx} out of range for k_pad {} (log_k_pad {})",
        1usize << layout.log_k_pad,
        layout.log_k_pad
    );
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

/// Decode the trailing slot bits of a `combined_eval_point` back
/// into the constituent slot index. Used by tests; the verifier
/// never has to do this in normal operation since it constructs the
/// point from a known slot, not the other way round.
#[cfg(test)]
#[must_use]
pub fn decode_constituent_index<E: Field>(layout: &SparseLayout, point: &[E]) -> Option<usize> {
    if point.len() != layout.total_vars {
        return None;
    }
    let mut idx = 0usize;
    for bit_pos in 0..layout.log_k_pad {
        let v = point[layout.mu + bit_pos];
        if v == E::ZERO {
            // bit is 0
        } else if v == E::ONE {
            idx |= 1 << bit_pos;
        } else {
            return None;
        }
    }
    Some(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::commit::{sparse_commit, SparseLayout};
    use crate::whir::sparse::types::eval_eq_at_index;
    use crate::whir::sparse::{SparseArity, SparseMle3};
    use arith::Field;
    use goldilocks::{Goldilocks, GoldilocksExt3};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

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

    #[test]
    fn combined_point_layout_two_axis() {
        let layout = SparseLayout::compute(SparseArity::Two, 8, 4, 4, 1);
        // mu = 3, k_pad = 8, log_k_pad = 3, total_vars = 6
        assert_eq!(layout.total_vars, 6);
        let sub_point: Vec<GoldilocksExt3> = vec![
            GoldilocksExt3::from(2u64),
            GoldilocksExt3::from(3u64),
            GoldilocksExt3::from(5u64),
        ];
        // Slot 0 (Val) → idx_bits = [0, 0, 0]
        let p0 = combined_eval_point(&layout, SparseConstituentSlot::Val, &sub_point);
        assert_eq!(p0.len(), 6);
        assert_eq!(&p0[..3], sub_point.as_slice());
        assert_eq!(p0[3], GoldilocksExt3::ZERO);
        assert_eq!(p0[4], GoldilocksExt3::ZERO);
        assert_eq!(p0[5], GoldilocksExt3::ZERO);

        // Slot 6 (AuditTsX) → idx_bits = [0, 1, 1] (LE binary of 6)
        let p6 = combined_eval_point(&layout, SparseConstituentSlot::AuditTsX, &sub_point);
        assert_eq!(p6[3], GoldilocksExt3::ZERO);
        assert_eq!(p6[4], GoldilocksExt3::ONE);
        assert_eq!(p6[5], GoldilocksExt3::ONE);
    }

    #[test]
    fn combined_point_round_trips_through_decoder() {
        let layout = SparseLayout::compute(SparseArity::Three, 16, 8, 8, 8);
        let sub_point: Vec<GoldilocksExt3> = (0..layout.mu)
            .map(|i| GoldilocksExt3::from((i as u64) + 1))
            .collect();
        for slot in &[
            SparseConstituentSlot::Val,
            SparseConstituentSlot::Row,
            SparseConstituentSlot::ColX,
            SparseConstituentSlot::ColY,
            SparseConstituentSlot::ReadTsZ,
            SparseConstituentSlot::AuditTsZ,
            SparseConstituentSlot::ReadTsX,
            SparseConstituentSlot::AuditTsX,
            SparseConstituentSlot::ReadTsY,
            SparseConstituentSlot::AuditTsY,
        ] {
            let point = combined_eval_point(&layout, *slot, &sub_point);
            assert_eq!(
                decode_constituent_index(&layout, &point),
                Some(slot.index())
            );
        }
    }

    #[test]
    #[should_panic(expected = "sub_point length")]
    fn combined_point_rejects_wrong_sub_point_length() {
        let layout = SparseLayout::compute(SparseArity::Two, 8, 4, 4, 1);
        let bad_sub_point: Vec<GoldilocksExt3> = vec![GoldilocksExt3::ZERO; 2]; // mu = 3
        let _ = combined_eval_point(&layout, SparseConstituentSlot::Val, &bad_sub_point);
    }

    #[test]
    fn combined_point_evaluation_matches_constituent_evaluation_via_dense_oracle() {
        // Sanity check that evaluating the combined dense polynomial
        // at combined_eval_point(slot, r) yields the same value as
        // evaluating the constituent's MLE at r directly. This pins
        // the layout interpretation: the verifier's WHIR open at
        // combined_eval_point IS an opening of the constituent at r.
        let mut rng = rng_for("combined_point_dense_oracle");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (commitment, scratch, _tree, _codeword) = sparse_commit::<Goldilocks>(&poly).unwrap();

        let layout = SparseLayout::compute(
            SparseArity::Two,
            scratch.nnz,
            1usize << scratch.n_z,
            1usize << scratch.n_x,
            1,
        );
        assert_eq!(layout.total_vars, commitment.batched_num_vars);

        // Build a dense version of the val constituent for the
        // oracle check. The val slot in the combined polynomial has
        // length slot_len, padded with zeros past nnz.
        let mut val_dense: Vec<Goldilocks> = vec![Goldilocks::ZERO; layout.slot_len()];
        val_dense[..scratch.val.len()].copy_from_slice(&scratch.val);

        // Pick a random sub-point in E^μ
        let sub_point: Vec<GoldilocksExt3> = (0..layout.mu)
            .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
            .collect();
        let combined_point = combined_eval_point(&layout, SparseConstituentSlot::Val, &sub_point);

        // Evaluate val_dense MLE at sub_point directly
        let mut expected_val_eval = GoldilocksExt3::ZERO;
        for (i, v) in val_dense.iter().enumerate() {
            expected_val_eval += eval_eq_at_index(&sub_point, i) * GoldilocksExt3::from(*v);
        }

        // Build the combined dense vector exactly as sparse_commit
        // would, then evaluate it at combined_point. They must match.
        let combined_len = layout.combined_len();
        let mut combined: Vec<Goldilocks> = vec![Goldilocks::ZERO; combined_len];
        let val_offset = layout.slot_offset(SparseConstituentSlot::Val);
        combined[val_offset..val_offset + scratch.val.len()].copy_from_slice(&scratch.val);

        let mut combined_eval = GoldilocksExt3::ZERO;
        for (b, v) in combined.iter().enumerate() {
            combined_eval += eval_eq_at_index(&combined_point, b) * GoldilocksExt3::from(*v);
        }
        assert_eq!(
            expected_val_eval, combined_eval,
            "combined-point evaluation must equal constituent-MLE evaluation"
        );
    }
}
