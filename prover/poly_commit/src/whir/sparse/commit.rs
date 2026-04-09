//! Commit phase for the sparse-MLE WHIR commitment.
//!
//! Materializes the constituent dense polynomials of a sparse
//! multilinear polynomial — the address vectors (`row`, `col_x`,
//! `col_y`), the value vector `val`, and the per-axis offline
//! memory-checking tracks (`read_ts`, `audit_ts`) — packs them into a
//! single combined dense polynomial via Spartan §7.2.3 optimization
//! (5), and produces a single WHIR commitment whose Merkle root lives
//! in [`SparseMle3Commitment`].
//!
//! Layout of the combined polynomial. Each constituent is padded to a
//! uniform length `2^μ`, where
//!
//! ```text
//!     μ = ⌈log₂ max(nnz, m_z, m_x[, m_y])⌉
//! ```
//!
//! and the constituent count is padded to a power of two `k_pad`.
//! The combined dense vector has length `k_pad · 2^μ` and is laid
//! out so that constituent `i ∈ [k_pad]` occupies the slice
//! `combined[i · 2^μ .. (i + 1) · 2^μ]`. Under the little-endian MLE
//! convention used by the rest of the sparse module, the low `μ`
//! variables index *into* a constituent and the high `log₂ k_pad`
//! variables select *which* constituent is being addressed.
//!
//! With this layout, opening constituent `i` at sumcheck point
//! `r ∈ Eᵘ` is equivalent to opening the combined polynomial at the
//! extended point `(r, idx_bits(i))`, where `idx_bits(i)` is the
//! binary expansion of `i` in `log₂ k_pad` little-endian bits. The
//! corresponding eq factor for the high bits is exactly
//! `eval_eq_at_index(idx_bits, i)`, so the verifier need not recover
//! the layout from the commitment header.
//!
//! Constituent ordering. We commit the constituents in a fixed order
//! given by [`SparseConstituentSlot`] so the prover and verifier
//! agree on which slice of the combined polynomial holds which
//! sub-vector. The order is intentional and enumerated by a public
//! enum so any future addition forces a deliberate slot assignment.

use arith::{FFTField, Field, SimdField};
use tree::Tree;

use super::memcheck::AddrTimestamps;
use super::types::{
    SparseArity, SparseMle3, SparseMle3Commitment, SparseMleError, SparseMleScratchPad,
};
use crate::whir::pcs_trait_impl::whir_commit;

/// Stable identifier for a constituent's slot in the combined dense
/// polynomial. The numeric values are used both as indices into the
/// combined layout and as the high-bit selector when opening a
/// constituent at the verifier.
///
/// Slot order is fixed for forwards / backwards compatibility of the
/// `SparseMle3Commitment` wire format. Adding a new constituent
/// requires extending this enum and bumping the commitment
/// serialization version (currently encoded implicitly via the
/// constituent count derived from arity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum SparseConstituentSlot {
    Val = 0,
    Row = 1,
    ColX = 2,
    ReadTsZ = 3,
    AuditTsZ = 4,
    ReadTsX = 5,
    AuditTsX = 6,
    /// 3-arity only — col_y address vector.
    ColY = 7,
    /// 3-arity only — y-axis read timestamps.
    ReadTsY = 8,
    /// 3-arity only — y-axis audit timestamps.
    AuditTsY = 9,
}

impl SparseConstituentSlot {
    /// Slot index in the combined dense polynomial.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// All slots active for a given arity, in canonical order.
    #[must_use]
    pub fn slots_for(arity: SparseArity) -> &'static [Self] {
        const TWO_AXIS: &[SparseConstituentSlot] = &[
            SparseConstituentSlot::Val,
            SparseConstituentSlot::Row,
            SparseConstituentSlot::ColX,
            SparseConstituentSlot::ReadTsZ,
            SparseConstituentSlot::AuditTsZ,
            SparseConstituentSlot::ReadTsX,
            SparseConstituentSlot::AuditTsX,
        ];
        const THREE_AXIS: &[SparseConstituentSlot] = &[
            SparseConstituentSlot::Val,
            SparseConstituentSlot::Row,
            SparseConstituentSlot::ColX,
            SparseConstituentSlot::ReadTsZ,
            SparseConstituentSlot::AuditTsZ,
            SparseConstituentSlot::ReadTsX,
            SparseConstituentSlot::AuditTsX,
            SparseConstituentSlot::ColY,
            SparseConstituentSlot::ReadTsY,
            SparseConstituentSlot::AuditTsY,
        ];
        match arity {
            SparseArity::Two => TWO_AXIS,
            SparseArity::Three => THREE_AXIS,
        }
    }

    /// Number of constituents active for a given arity.
    #[inline]
    #[must_use]
    pub fn count_for(arity: SparseArity) -> usize {
        Self::slots_for(arity).len()
    }
}

/// Layout descriptor for the combined dense polynomial held by a
/// [`SparseMle3Commitment`]. The descriptor is fully recoverable from
/// the public `SparseMle3Commitment` fields, but pre-computing it
/// once and caching it in [`SparseMleScratchPad`] avoids repeating
/// the bookkeeping at every open / verify call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseLayout {
    pub arity: SparseArity,
    /// Log size of each constituent slot before slot-index padding.
    pub mu: usize,
    /// Constituent count after padding to a power of two.
    pub k_pad: usize,
    /// Log of `k_pad`, i.e. number of high-order variables that
    /// select which constituent is being addressed.
    pub log_k_pad: usize,
    /// Total number of variables in the combined polynomial,
    /// `mu + log_k_pad`.
    pub total_vars: usize,
}

impl SparseLayout {
    /// Compute the layout descriptor for the given arity, sparse
    /// dimensions, and per-axis cell counts.
    ///
    /// `mu` is taken to be `⌈log₂ max(nnz, m_z, m_x[, m_y])⌉` so the
    /// largest constituent fits in `2^μ`. Smaller constituents are
    /// zero-padded to that length at commit time.
    ///
    /// # Panics
    /// Panics if `nnz` is `0`. Empty sparse polynomials have no
    /// commitment to make.
    #[must_use]
    pub fn compute(arity: SparseArity, nnz: usize, m_z: usize, m_x: usize, m_y: usize) -> Self {
        assert!(nnz > 0, "SparseLayout::compute: nnz must be non-zero");
        let max_len = match arity {
            SparseArity::Two => nnz.max(m_z).max(m_x),
            SparseArity::Three => nnz.max(m_z).max(m_x).max(m_y),
        };
        let mu = log2_ceil(max_len);
        let num_constituents = SparseConstituentSlot::count_for(arity);
        let k_pad = num_constituents.next_power_of_two();
        let log_k_pad = k_pad.trailing_zeros() as usize;
        Self {
            arity,
            mu,
            k_pad,
            log_k_pad,
            total_vars: mu + log_k_pad,
        }
    }

    /// Length of the combined dense polynomial, `2^total_vars`.
    #[inline]
    #[must_use]
    pub fn combined_len(&self) -> usize {
        1usize << self.total_vars
    }

    /// Length of each constituent slot, `2^μ`.
    #[inline]
    #[must_use]
    pub fn slot_len(&self) -> usize {
        1usize << self.mu
    }

    /// Byte offset (in field elements) of slot `s` in the combined
    /// dense vector.
    #[inline]
    #[must_use]
    pub fn slot_offset(&self, slot: SparseConstituentSlot) -> usize {
        slot.index() * self.slot_len()
    }
}

/// Commit a sparse multilinear polynomial.
///
/// Returns the public commitment, the prover scratch pad (which
/// holds the dense constituent vectors and the WHIR Merkle tree /
/// codeword for later opening calls), and the field-element layout
/// descriptor.
///
/// `F` must be a finite field for which WHIR is defined: it must
/// implement `FFTField + SimdField<Scalar = F>`. For our production
/// configurations this is `Goldilocks` (commit constituents in the
/// base field; the open phase lifts to the extension on demand) or
/// any of `GoldilocksExt2/Ext3/Ext4` (commit directly in the
/// extension when the caller already has lifted constituents).
///
/// # Errors
/// Returns [`SparseMleError`] if `poly.validate()` rejects the input,
/// e.g. due to a dimension mismatch or an out-of-range address.
pub fn sparse_commit<F: FFTField + SimdField<Scalar = F>>(
    poly: &SparseMle3<F>,
) -> Result<(SparseMle3Commitment, SparseMleScratchPad<F>, Tree, Vec<F>), SparseMleError> {
    poly.validate()?;
    let nnz = poly.nnz();
    let m_z = 1usize << poly.n_z;
    let m_x = 1usize << poly.n_x;
    let m_y = if poly.arity == SparseArity::Two {
        1
    } else {
        1usize << poly.n_y
    };

    let layout = SparseLayout::compute(poly.arity, nnz, m_z, m_x, m_y);

    // Compute per-axis timestamps. For 2-axis polynomials the y-axis
    // is unused (col_y is all zeros and y is omitted from the
    // multiset checks); we still construct an empty `AddrTimestamps`
    // placeholder for the scratch pad slot so its layout is uniform.
    let ts_z = AddrTimestamps::<F>::memory_in_the_head(m_z, &poly.row);
    let ts_x = AddrTimestamps::<F>::memory_in_the_head(m_x, &poly.col_x);
    let ts_y = if poly.arity == SparseArity::Three {
        Some(AddrTimestamps::<F>::memory_in_the_head(m_y, &poly.col_y))
    } else {
        None
    };

    // Materialize the combined dense polynomial. Each slot is
    // populated by `populate_slot` and zero-padded automatically by
    // the initial `vec![F::ZERO; combined_len]`.
    let combined_len = layout.combined_len();
    let mut combined: Vec<F> = vec![F::ZERO; combined_len];

    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::Val,
        &poly.val,
    );

    let row_field: Vec<F> = lift_addresses(&poly.row);
    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::Row,
        &row_field,
    );

    let col_x_field: Vec<F> = lift_addresses(&poly.col_x);
    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::ColX,
        &col_x_field,
    );

    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::ReadTsZ,
        &ts_z.read_ts,
    );
    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::AuditTsZ,
        &ts_z.audit_ts,
    );

    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::ReadTsX,
        &ts_x.read_ts,
    );
    populate_slot(
        &mut combined,
        &layout,
        SparseConstituentSlot::AuditTsX,
        &ts_x.audit_ts,
    );

    if poly.arity == SparseArity::Three {
        let col_y_field: Vec<F> = lift_addresses(&poly.col_y);
        populate_slot(
            &mut combined,
            &layout,
            SparseConstituentSlot::ColY,
            &col_y_field,
        );

        let ts_y_ref = ts_y.as_ref().expect("ts_y must be present for arity Three");
        populate_slot(
            &mut combined,
            &layout,
            SparseConstituentSlot::ReadTsY,
            &ts_y_ref.read_ts,
        );
        populate_slot(
            &mut combined,
            &layout,
            SparseConstituentSlot::AuditTsY,
            &ts_y_ref.audit_ts,
        );
    }

    // Run the underlying WHIR commit on the combined polynomial.
    let (whir_commitment, tree, codeword) = whir_commit::<F>(&combined);
    debug_assert_eq!(whir_commitment.num_vars, layout.total_vars);

    let log_nnz = log2_ceil(nnz);

    let commitment = SparseMle3Commitment {
        arity: poly.arity,
        n_z: poly.n_z,
        n_x: poly.n_x,
        n_y: poly.n_y,
        nnz,
        log_nnz,
        batched_root: whir_commitment.root,
        batched_num_vars: layout.total_vars,
    };

    let scratch = SparseMleScratchPad {
        arity: poly.arity,
        n_z: poly.n_z,
        n_x: poly.n_x,
        n_y: poly.n_y,
        nnz,
        log_nnz,
        row: poly.row.clone(),
        col_x: poly.col_x.clone(),
        col_y: poly.col_y.clone(),
        val: poly.val.clone(),
        ts_z,
        ts_x,
        ts_y,
    };

    Ok((commitment, scratch, tree, codeword))
}

/// Copy a constituent vector into its slot in the combined dense
/// polynomial. Source vectors shorter than `slot_len` are zero-padded
/// at the high-index end (which is the natural multilinear-extension
/// padding semantics).
fn populate_slot<F: Field>(
    combined: &mut [F],
    layout: &SparseLayout,
    slot: SparseConstituentSlot,
    source: &[F],
) {
    let slot_len = layout.slot_len();
    debug_assert!(source.len() <= slot_len, "source longer than slot");
    let offset = layout.slot_offset(slot);
    combined[offset..offset + source.len()].copy_from_slice(source);
}

/// Lift a `&[usize]` address vector into the field by `F::from(u32)`.
/// Caller is responsible for ensuring all addresses fit in `u32`,
/// which the `SparseMle3::validate` precondition guarantees via
/// [`super::types::SPARSE_MLE_MAX_LOG_DOMAIN`].
fn lift_addresses<F: Field>(addrs: &[usize]) -> Vec<F> {
    addrs
        .iter()
        .map(|&a| {
            let a32 = u32::try_from(a)
                .expect("address exceeds u32::MAX — caught by SparseMle3::validate");
            F::from(a32)
        })
        .collect()
}

/// `⌈log₂ n⌉` with the convention `log2_ceil(0) = 0` and
/// `log2_ceil(1) = 0`. For `n ≥ 2`, this returns the smallest `k`
/// such that `n ≤ 2^k`.
#[inline]
fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (n - 1).ilog2() as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whir::sparse::types::{SparseArity, SparseMle3};
    use arith::Field;
    use goldilocks::Goldilocks;
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
    fn log2_ceil_matches_table() {
        let cases = [
            (0usize, 0usize),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (7, 3),
            (8, 3),
            (9, 4),
            (16, 4),
            (17, 5),
            (1024, 10),
            (1025, 11),
        ];
        for (n, expected) in cases {
            assert_eq!(log2_ceil(n), expected, "log2_ceil({n}) wrong");
        }
    }

    #[test]
    fn slot_count_two_axis_is_seven() {
        // val + row + col_x + (read_ts, audit_ts)·2 = 7
        assert_eq!(SparseConstituentSlot::count_for(SparseArity::Two), 7);
    }

    #[test]
    fn slot_count_three_axis_is_ten() {
        // val + row + col_x + col_y + (read_ts, audit_ts)·3 = 10
        assert_eq!(SparseConstituentSlot::count_for(SparseArity::Three), 10);
    }

    #[test]
    fn layout_two_axis_pads_to_eight_slots() {
        // 7 constituents → padded to 8 (next power of two), so
        // log_k_pad = 3.
        let layout = SparseLayout::compute(SparseArity::Two, 8, 4, 4, 1);
        assert_eq!(layout.k_pad, 8);
        assert_eq!(layout.log_k_pad, 3);
        assert_eq!(layout.mu, 3); // max(8, 4, 4) = 8 → log = 3
        assert_eq!(layout.total_vars, 6);
        assert_eq!(layout.combined_len(), 64);
        assert_eq!(layout.slot_len(), 8);
    }

    #[test]
    fn layout_three_axis_pads_to_sixteen_slots() {
        // 10 constituents → padded to 16, log_k_pad = 4.
        let layout = SparseLayout::compute(SparseArity::Three, 16, 8, 8, 8);
        assert_eq!(layout.k_pad, 16);
        assert_eq!(layout.log_k_pad, 4);
        assert_eq!(layout.mu, 4); // max(16, 8, 8, 8) = 16
        assert_eq!(layout.total_vars, 8);
        assert_eq!(layout.combined_len(), 256);
        assert_eq!(layout.slot_len(), 16);
    }

    #[test]
    fn layout_uses_largest_axis_for_mu() {
        // nnz < m_z → m_z dictates mu
        let layout = SparseLayout::compute(SparseArity::Two, 4, 32, 8, 1);
        assert_eq!(layout.mu, 5); // max(4, 32, 8) = 32 → log = 5
    }

    #[test]
    fn slot_offsets_increase_monotonically() {
        let layout = SparseLayout::compute(SparseArity::Three, 16, 8, 8, 8);
        let mut prev = 0usize;
        for slot in SparseConstituentSlot::slots_for(SparseArity::Three) {
            let offset = layout.slot_offset(*slot);
            assert!(offset >= prev, "slot offsets must be monotonic");
            prev = offset;
        }
    }

    #[test]
    fn commit_two_axis_round_trips_layout() {
        let mut rng = rng_for("commit_two_axis");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (commitment, scratch, _tree, codeword) =
            sparse_commit::<Goldilocks>(&poly).expect("valid commit");

        // Commitment metadata matches input.
        assert_eq!(commitment.arity, SparseArity::Two);
        assert_eq!(commitment.n_z, 3);
        assert_eq!(commitment.n_x, 3);
        assert_eq!(commitment.nnz, 8);
        assert_eq!(commitment.log_nnz, 3);
        // Layout: 7 slots → k_pad = 8, mu = 3 (max(8, 8, 8) = 8 → log = 3),
        // total_vars = 6, combined_len = 64.
        assert_eq!(commitment.batched_num_vars, 6);

        // Scratch holds the constituent vectors verbatim.
        assert_eq!(scratch.row, poly.row);
        assert_eq!(scratch.col_x, poly.col_x);
        assert_eq!(scratch.val, poly.val);
        assert!(scratch.ts_y.is_none());

        // Codeword length is 2^total_vars * 2^WHIR_RATE_LOG.
        // We don't pin the rate constant here, but we sanity-check
        // it is at least the combined dense length and is a power
        // of two.
        assert!(codeword.len() >= 64);
        assert!(codeword.len().is_power_of_two());
    }

    #[test]
    fn commit_three_axis_round_trips_layout() {
        let mut rng = rng_for("commit_three_axis");
        let poly = build_three_axis(&mut rng, 3, 3, 3, 8);
        let (commitment, scratch, _tree, codeword) =
            sparse_commit::<Goldilocks>(&poly).expect("valid commit");

        assert_eq!(commitment.arity, SparseArity::Three);
        assert_eq!(commitment.n_z, 3);
        assert_eq!(commitment.n_x, 3);
        assert_eq!(commitment.n_y, 3);
        assert_eq!(commitment.nnz, 8);
        // Layout: 10 slots → k_pad = 16, mu = 3 (max all = 8 → log = 3),
        // total_vars = 7, combined_len = 128.
        assert_eq!(commitment.batched_num_vars, 7);

        assert_eq!(scratch.row, poly.row);
        assert_eq!(scratch.col_x, poly.col_x);
        assert_eq!(scratch.col_y, poly.col_y);
        assert_eq!(scratch.val, poly.val);
        assert!(scratch.ts_y.is_some());
        let ts_y = scratch.ts_y.as_ref().unwrap();
        assert_eq!(ts_y.read_ts.len(), 8);
        assert_eq!(ts_y.audit_ts.len(), 8);

        assert!(codeword.len() >= 128);
        assert!(codeword.len().is_power_of_two());
    }

    #[test]
    fn commit_rejects_invalid_input() {
        // Address out of range — n_z = 2 means addresses must be < 4.
        let bad: SparseMle3<Goldilocks> = SparseMle3 {
            n_z: 2,
            n_x: 2,
            n_y: 0,
            arity: SparseArity::Two,
            row: vec![0, 4], // 4 ≥ 2² = 4
            col_x: vec![0, 1],
            col_y: vec![0, 0],
            val: vec![Goldilocks::ONE, Goldilocks::ONE],
        };
        let err = sparse_commit::<Goldilocks>(&bad).unwrap_err();
        assert!(matches!(err, SparseMleError::AddressOutOfRange { .. }));
    }

    #[test]
    fn commit_distinct_polynomials_yield_distinct_roots() {
        let mut rng = rng_for("commit_distinct_roots");
        let poly_a = build_two_axis(&mut rng, 3, 3, 8);
        let poly_b = build_two_axis(&mut rng, 3, 3, 8);
        let (ca, _, _, _) = sparse_commit::<Goldilocks>(&poly_a).unwrap();
        let (cb, _, _, _) = sparse_commit::<Goldilocks>(&poly_b).unwrap();
        assert_ne!(
            ca.batched_root, cb.batched_root,
            "distinct sparse polynomials must commit to distinct roots"
        );
    }

    #[test]
    fn commit_is_deterministic() {
        let mut rng = rng_for("commit_determinism");
        let poly = build_two_axis(&mut rng, 3, 3, 8);
        let (c1, _, _, _) = sparse_commit::<Goldilocks>(&poly).unwrap();
        let (c2, _, _, _) = sparse_commit::<Goldilocks>(&poly).unwrap();
        assert_eq!(
            c1.batched_root, c2.batched_root,
            "commit must be a deterministic function of the input polynomial"
        );
    }
}
