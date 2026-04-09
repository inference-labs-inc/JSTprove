//! Tests for the sparse-MLE WHIR foundation: data structures, the
//! `MemoryInTheHead` derivation, and the multiset hash subset check.
//!
//! Protocol-level tests (open / verify round trip, end-to-end against
//! WHIR) live alongside the open / verify modules and are gated on
//! the `goldilocks` feature so they only run when the field used by
//! production GoldilocksExt3Whir / GoldilocksExt4Whir paths is in
//! scope.

use goldilocks::{Goldilocks, GoldilocksExt3, GoldilocksExt4};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use super::memcheck::{build_memcheck_sets, multiset_hash, AddrTimestamps, MemoryHashParams};
use super::types::{
    eval_eq_at_index, AddressAxis, SparseArity, SparseMle3, SparseMle3Commitment, SparseMleError,
    SPARSE_MLE_MAX_LOG_DOMAIN,
};
use arith::Field;
use serdes::ExpSerde;

fn rng_for(label: &str) -> ChaCha20Rng {
    let mut seed = [0u8; 32];
    let bytes = label.as_bytes();
    let n = bytes.len().min(32);
    seed[..n].copy_from_slice(&bytes[..n]);
    ChaCha20Rng::from_seed(seed)
}

// ----------------------------------------------------------------------------
// MemoryInTheHead
// ----------------------------------------------------------------------------

#[test]
fn memcheck_per_cell_counters_are_monotonic() {
    // addrs = [2, 0, 2, 1, 2, 0]; m = 3
    // Cell-by-cell read counts:
    //   addr=2 occurs at i=0,2,4  -> read_ts = 0, 1, 2 ; audit_ts[2] = 3
    //   addr=0 occurs at i=1,5    -> read_ts = 0, 1    ; audit_ts[0] = 2
    //   addr=1 occurs at i=3      -> read_ts = 0       ; audit_ts[1] = 1
    let addrs = [2usize, 0, 2, 1, 2, 0];
    let ts = AddrTimestamps::<Goldilocks>::memory_in_the_head(3, &addrs);

    let g = |n: u64| Goldilocks::from(n);
    assert_eq!(
        ts.read_ts,
        vec![g(0), g(0), g(1), g(0), g(2), g(1)],
        "per-cell read timestamps must increment in occurrence order"
    );
    assert_eq!(
        ts.audit_ts,
        vec![g(2), g(1), g(3)],
        "audit timestamps must equal final per-cell read counts"
    );
}

#[test]
fn memcheck_empty_address_sequence_yields_zero_audit() {
    let ts = AddrTimestamps::<Goldilocks>::memory_in_the_head(8, &[]);
    assert!(ts.read_ts.is_empty());
    assert_eq!(ts.audit_ts, vec![Goldilocks::ZERO; 8]);
}

#[test]
#[should_panic(expected = "out of range")]
fn memcheck_rejects_address_above_m() {
    let _ = AddrTimestamps::<Goldilocks>::memory_in_the_head(4, &[0, 1, 4]);
}

// ----------------------------------------------------------------------------
// Multiset hash + subset check
// ----------------------------------------------------------------------------

#[test]
fn multiset_hash_is_permutation_invariant() {
    let mut rng = rng_for("multiset_hash_perm_invariance");
    let xs: Vec<GoldilocksExt3> = (0..32)
        .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
        .collect();
    let mut ys = xs.clone();
    // Apply a non-trivial permutation
    ys.swap(0, 17);
    ys.swap(3, 9);
    ys.swap(11, 31);

    let gamma = GoldilocksExt3::random_unsafe(&mut rng);
    let hx = multiset_hash(&xs, gamma);
    let hy = multiset_hash(&ys, gamma);
    assert_eq!(hx, hy, "H_γ must be invariant under multiset permutation");
}

#[test]
fn multiset_hash_distinguishes_different_multisets() {
    let mut rng = rng_for("multiset_hash_distinguishes");
    let xs: Vec<GoldilocksExt3> = (0..32)
        .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
        .collect();
    let mut ys = xs.clone();
    // Replace one element with a guaranteed-different value
    ys[5] = ys[5] + GoldilocksExt3::ONE;
    let gamma = GoldilocksExt3::random_unsafe(&mut rng);
    let hx = multiset_hash(&xs, gamma);
    let hy = multiset_hash(&ys, gamma);
    assert_ne!(
        hx, hy,
        "H_γ must (with high probability over γ) reject distinct multisets"
    );
}

#[test]
fn memcheck_subset_equation_holds_for_consistent_trace() {
    let mut rng = rng_for("memcheck_subset_eq_consistent");

    // Build a small but representative trace
    let m = 16;
    let n = 64;
    let addrs: Vec<usize> = (0..n).map(|_| rng.gen_range(0..m)).collect();

    // Memory cell values: arbitrary fixed values (would be ẽq evals in
    // the real protocol — the equation we are checking is independent
    // of which value function is chosen).
    let mem: Vec<GoldilocksExt3> = (0..m)
        .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
        .collect();

    // Per-op value is whatever is currently in the cell at read time.
    // Since reads are trusted in our setting, val_per_op[i] = mem[addrs[i]].
    let val_per_op: Vec<GoldilocksExt3> = addrs.iter().map(|&a| mem[a]).collect();

    let ts = AddrTimestamps::<GoldilocksExt3>::memory_in_the_head(m, &addrs);
    let params = MemoryHashParams {
        gamma_1: GoldilocksExt3::random_unsafe(&mut rng),
        gamma_2: GoldilocksExt3::random_unsafe(&mut rng),
    };

    let sets = build_memcheck_sets(
        &params,
        &addrs,
        &val_per_op,
        &ts.read_ts,
        &mem,
        &ts.audit_ts,
    );
    assert!(
        sets.check_subset_equation(params.gamma_2),
        "honest trace must satisfy H(Init)·H(WS) = H(RS)·H(Audit)"
    );
}

#[test]
fn memcheck_subset_equation_rejects_skipped_read_ts() {
    let mut rng = rng_for("memcheck_subset_eq_tampered");

    let m = 8;
    let addrs: Vec<usize> = vec![0, 0, 1, 0, 2, 1, 2, 2];
    let mem: Vec<GoldilocksExt3> = (0..m)
        .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
        .collect();
    let val_per_op: Vec<GoldilocksExt3> = addrs.iter().map(|&a| mem[a]).collect();

    let mut ts = AddrTimestamps::<GoldilocksExt3>::memory_in_the_head(m, &addrs);
    // Corrupt one read timestamp — claim a later read happened earlier.
    ts.read_ts[3] = ts.read_ts[3] + GoldilocksExt3::ONE;

    let params = MemoryHashParams {
        gamma_1: GoldilocksExt3::random_unsafe(&mut rng),
        gamma_2: GoldilocksExt3::random_unsafe(&mut rng),
    };
    let sets = build_memcheck_sets(
        &params,
        &addrs,
        &val_per_op,
        &ts.read_ts,
        &mem,
        &ts.audit_ts,
    );
    assert!(
        !sets.check_subset_equation(params.gamma_2),
        "tampered read_ts must break the subset equation w.h.p."
    );
}

#[test]
fn memcheck_subset_equation_rejects_tampered_audit_ts() {
    let mut rng = rng_for("memcheck_subset_eq_audit_tampered");

    let m = 8;
    let addrs: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let mem: Vec<GoldilocksExt3> = (0..m)
        .map(|_| GoldilocksExt3::random_unsafe(&mut rng))
        .collect();
    let val_per_op: Vec<GoldilocksExt3> = addrs.iter().map(|&a| mem[a]).collect();

    let mut ts = AddrTimestamps::<GoldilocksExt3>::memory_in_the_head(m, &addrs);
    ts.audit_ts[2] = ts.audit_ts[2] + GoldilocksExt3::ONE;

    let params = MemoryHashParams {
        gamma_1: GoldilocksExt3::random_unsafe(&mut rng),
        gamma_2: GoldilocksExt3::random_unsafe(&mut rng),
    };
    let sets = build_memcheck_sets(
        &params,
        &addrs,
        &val_per_op,
        &ts.read_ts,
        &mem,
        &ts.audit_ts,
    );
    assert!(
        !sets.check_subset_equation(params.gamma_2),
        "tampered audit_ts must break the subset equation w.h.p."
    );
}

// ----------------------------------------------------------------------------
// SparseMle3 dense evaluation oracle
// ----------------------------------------------------------------------------

#[test]
fn sparse_evaluate_matches_dense_for_two_axis() {
    // 2-axis sparse polynomial M(z, x) over n_z = 2, n_x = 2.
    // Three nonzero entries:
    //   (z=01, x=10) val = 3
    //   (z=10, x=11) val = 5
    //   (z=11, x=00) val = 7
    let m: SparseMle3<Goldilocks> = SparseMle3 {
        n_z: 2,
        n_x: 2,
        n_y: 0,
        arity: SparseArity::Two,
        row: vec![1, 2, 3],
        col_x: vec![2, 3, 0],
        col_y: vec![0, 0, 0],
        val: vec![
            Goldilocks::from(3u64),
            Goldilocks::from(5u64),
            Goldilocks::from(7u64),
        ],
    };
    m.validate().expect("valid sparse layout");

    // Pick a deterministic evaluation point
    let z = vec![GoldilocksExt3::from(2u64), GoldilocksExt3::from(3u64)];
    let x = vec![GoldilocksExt3::from(5u64), GoldilocksExt3::from(7u64)];
    let y: Vec<GoldilocksExt3> = vec![];

    let v_sparse = m.evaluate(&z, &x, &y);

    // Dense reconstruction: build the full evaluation table and
    // multilinearly extend it.
    let mut dense = vec![Goldilocks::ZERO; 1 << 4];
    for k in 0..m.val.len() {
        let idx = m.row[k] | (m.col_x[k] << 2);
        dense[idx] = dense[idx] + m.val[k];
    }
    // Evaluate dense MLE at (z||x) by direct sum: Σ_b dense[b] · ẽq(b, point)
    // where the variable order is (z_0, z_1, x_0, x_1).
    let point: Vec<GoldilocksExt3> = z.iter().chain(x.iter()).copied().collect();
    let mut v_dense = GoldilocksExt3::ZERO;
    for (b, val) in dense.iter().enumerate() {
        let eq = eval_eq_at_index(&point, b);
        v_dense += eq * GoldilocksExt3::from(*val);
    }

    assert_eq!(v_sparse, v_dense, "sparse evaluate must match dense MLE");
}

#[test]
fn sparse_evaluate_matches_dense_for_three_axis() {
    // 3-axis sparse polynomial M(z, x, y) over n_z = n_x = n_y = 2.
    let m: SparseMle3<Goldilocks> = SparseMle3 {
        n_z: 2,
        n_x: 2,
        n_y: 2,
        arity: SparseArity::Three,
        row: vec![0, 1, 2, 3],
        col_x: vec![3, 2, 1, 0],
        col_y: vec![1, 2, 3, 0],
        val: (1u64..=4u64).map(Goldilocks::from).collect(),
    };
    m.validate().expect("valid sparse layout");

    let z = vec![GoldilocksExt4::from(11u64), GoldilocksExt4::from(13u64)];
    let x = vec![GoldilocksExt4::from(17u64), GoldilocksExt4::from(19u64)];
    let y = vec![GoldilocksExt4::from(23u64), GoldilocksExt4::from(29u64)];

    let v_sparse = m.evaluate(&z, &x, &y);

    let mut dense = vec![Goldilocks::ZERO; 1 << 6];
    for k in 0..m.val.len() {
        let idx = m.row[k] | (m.col_x[k] << 2) | (m.col_y[k] << 4);
        dense[idx] = dense[idx] + m.val[k];
    }
    let point: Vec<GoldilocksExt4> = z.iter().chain(x.iter()).chain(y.iter()).copied().collect();
    let mut v_dense = GoldilocksExt4::ZERO;
    for (b, val) in dense.iter().enumerate() {
        let eq = eval_eq_at_index(&point, b);
        v_dense += eq * GoldilocksExt4::from(*val);
    }

    assert_eq!(v_sparse, v_dense, "sparse evaluate must match dense MLE");
}

// ----------------------------------------------------------------------------
// Validation
// ----------------------------------------------------------------------------

#[test]
fn validate_rejects_dimension_mismatch() {
    let m: SparseMle3<Goldilocks> = SparseMle3 {
        n_z: 2,
        n_x: 2,
        n_y: 0,
        arity: SparseArity::Two,
        row: vec![0, 1],
        col_x: vec![0, 1, 2],
        col_y: vec![0, 0],
        val: vec![Goldilocks::ONE, Goldilocks::ONE],
    };
    let err = m.validate().unwrap_err();
    assert!(matches!(err, SparseMleError::DimensionMismatch { .. }));
}

#[test]
fn validate_rejects_address_overflow() {
    let m: SparseMle3<Goldilocks> = SparseMle3 {
        n_z: 2,
        n_x: 2,
        n_y: 0,
        arity: SparseArity::Two,
        row: vec![0, 4],
        col_x: vec![0, 1],
        col_y: vec![0, 0],
        val: vec![Goldilocks::ONE, Goldilocks::ONE],
    };
    let err = m.validate().unwrap_err();
    match err {
        SparseMleError::AddressOutOfRange {
            axis,
            index,
            addr,
            bound,
        } => {
            assert_eq!(axis, AddressAxis::Z);
            assert_eq!(index, 1);
            assert_eq!(addr, 4);
            assert_eq!(bound, 4);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn validate_rejects_two_axis_with_nonzero_y_address() {
    let m: SparseMle3<Goldilocks> = SparseMle3 {
        n_z: 2,
        n_x: 2,
        n_y: 0,
        arity: SparseArity::Two,
        row: vec![0],
        col_x: vec![0],
        col_y: vec![1],
        val: vec![Goldilocks::ONE],
    };
    let err = m.validate().unwrap_err();
    assert!(matches!(
        err,
        SparseMleError::AddressOutOfRange {
            axis: AddressAxis::Y,
            ..
        }
    ));
}

#[test]
fn validate_rejects_log_domain_overflow() {
    let m: SparseMle3<Goldilocks> = SparseMle3 {
        n_z: SPARSE_MLE_MAX_LOG_DOMAIN + 1,
        n_x: 0,
        n_y: 0,
        arity: SparseArity::Two,
        row: vec![],
        col_x: vec![],
        col_y: vec![],
        val: vec![],
    };
    assert!(matches!(
        m.validate(),
        Err(SparseMleError::DomainTooLarge { .. })
    ));
}

// ----------------------------------------------------------------------------
// Commitment serialization round trip
// ----------------------------------------------------------------------------

#[test]
fn commitment_serialization_round_trip() {
    use tree::Node;
    let commitment = SparseMle3Commitment {
        arity: SparseArity::Three,
        n_z: 5,
        n_x: 6,
        n_y: 7,
        nnz: 100,
        log_nnz: 7,
        batched_root: Node::default(),
        batched_num_vars: 12,
    };
    let mut bytes = Vec::new();
    commitment.serialize_into(&mut bytes).unwrap();
    let decoded = SparseMle3Commitment::deserialize_from(&bytes[..]).unwrap();
    assert_eq!(decoded.arity, commitment.arity);
    assert_eq!(decoded.n_z, commitment.n_z);
    assert_eq!(decoded.n_x, commitment.n_x);
    assert_eq!(decoded.n_y, commitment.n_y);
    assert_eq!(decoded.nnz, commitment.nnz);
    assert_eq!(decoded.log_nnz, commitment.log_nnz);
    assert_eq!(decoded.batched_num_vars, commitment.batched_num_vars);
}
