use goldilocks::{Goldilocks, GoldilocksExt2};

use super::adapter;

fn eval_mle_lsb_first(evals: &[GoldilocksExt2], point: &[GoldilocksExt2]) -> GoldilocksExt2 {
    use arith::Field;
    let n = point.len();
    assert_eq!(evals.len(), 1 << n);
    let mut scratch = evals.to_vec();
    for (k, &r) in point.iter().enumerate() {
        let half = 1 << (n - 1 - k);
        for i in 0..half {
            scratch[i] = scratch[i * 2] * (GoldilocksExt2::ONE - r) + scratch[i * 2 + 1] * r;
        }
    }
    scratch[0]
}

#[test]
fn whir_field_conversion_roundtrip() {
    use ark_ff_05::PrimeField;
    for v in [0u64, 1, 42, 1000000007, 0xFFFFFFFF00000000] {
        let whir_f = adapter::goldilocks_u64_to_whir(v);
        let back: u64 = whir_f.into_bigint().0[0];
        assert_eq!(v, back, "roundtrip failed for {v}");
    }
}

#[test]
fn whir_ext2_conversion_roundtrip() {
    use ark_ff_05::PrimeField;
    let pairs = [(0u64, 0u64), (1, 0), (0, 1), (42, 999), (0xFFFFFFFE, 7)];
    for (c0, c1) in pairs {
        let whir_f = adapter::goldilocks_ext2_to_whir(c0, c1);
        let back0: u64 = whir_f.c0.into_bigint().0[0];
        let back1: u64 = whir_f.c1.into_bigint().0[0];
        assert_eq!((c0, c1), (back0, back1), "ext2 roundtrip failed");
    }
}

#[test]
fn whir_commit_open_verify_roundtrip() {
    let num_vars = 6;
    let n = 1 << num_vars;

    let base_evals: Vec<u64> = (0..n).map(|i| (i as u64) + 1).collect();

    let point_pairs: Vec<(u64, u64)> = (0..num_vars).map(|i| ((i as u64) + 10, 0u64)).collect();

    let proof = adapter::whir_commit_and_open(&base_evals, &point_pairs, num_vars);

    let ext_evals: Vec<GoldilocksExt2> = base_evals
        .iter()
        .map(|&x| GoldilocksExt2::from(Goldilocks { v: x }))
        .collect();
    let point: Vec<GoldilocksExt2> = point_pairs
        .iter()
        .map(|&(c0, _)| GoldilocksExt2::from(Goldilocks { v: c0 }))
        .collect();
    let claimed_eval = eval_mle_lsb_first(&ext_evals, &point);

    let mut eval_buf = [0u8; 16];
    serdes::ExpSerde::serialize_into(&claimed_eval, &mut eval_buf[..]).unwrap();
    let eval_c0 = u64::from_le_bytes(eval_buf[..8].try_into().unwrap());
    let eval_c1 = u64::from_le_bytes(eval_buf[8..16].try_into().unwrap());

    let result = adapter::whir_verify(&point_pairs, (eval_c0, eval_c1), num_vars, &proof);

    assert!(result, "WHIR verification failed");
}

#[test]
fn whir_rejects_wrong_evaluation() {
    let num_vars = 4;
    let n = 1 << num_vars;

    let base_evals: Vec<u64> = (0..n).map(|i| (i as u64) + 1).collect();
    let point_pairs: Vec<(u64, u64)> = (0..num_vars).map(|i| ((i as u64) + 10, 0u64)).collect();

    let proof = adapter::whir_commit_and_open(&base_evals, &point_pairs, num_vars);

    let result = adapter::whir_verify(&point_pairs, (999, 0), num_vars, &proof);

    assert!(!result, "WHIR should reject wrong evaluation");
}
