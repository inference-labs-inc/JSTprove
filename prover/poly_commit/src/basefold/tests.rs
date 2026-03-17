use arith::Field;
use goldilocks::{Goldilocks, GoldilocksExt2};

use super::commit::basefold_commit;
use super::encoding::fold_evals;
use super::open::basefold_open;
use super::verify::basefold_verify;

fn eval_mle_lsb_first(evals: &[GoldilocksExt2], point: &[GoldilocksExt2]) -> GoldilocksExt2 {
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
fn multilinear_fold_matches_eval_lsb_convention() {
    let num_vars = 4;
    let n = 1 << num_vars;

    let ext_evals: Vec<GoldilocksExt2> = (0..n)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u64 + 1)))
        .collect();

    let point: Vec<GoldilocksExt2> = (0..num_vars)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u64 + 10)))
        .collect();

    let expected_eval = eval_mle_lsb_first(&ext_evals, &point);

    let mut current = ext_evals;
    for round in 0..num_vars {
        let challenge = point[num_vars - 1 - round];
        current = fold_evals(&current, challenge);
    }

    assert_eq!(current.len(), 1);
    assert_eq!(current[0], expected_eval);
}

#[test]
fn basefold_commit_open_verify_roundtrip() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 6;
    let n = 1 << num_vars;

    let base_evals: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::from(i as u64 + 1)).collect();

    let point: Vec<GoldilocksExt2> = (0..num_vars)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u64 + 10)))
        .collect();

    let ext_evals: Vec<GoldilocksExt2> = base_evals
        .iter()
        .map(|&x| GoldilocksExt2::from(x))
        .collect();
    let claimed_eval = eval_mle_lsb_first(&ext_evals, &point);

    let (commitment, tree) = basefold_commit(&base_evals);

    let mut prover_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let opening = basefold_open::<Goldilocks, GoldilocksExt2>(
        &base_evals,
        &tree,
        num_vars,
        &point,
        &mut prover_transcript,
    );

    assert!(!opening.final_poly.is_empty());

    let mut verifier_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let result = basefold_verify::<Goldilocks, GoldilocksExt2>(
        &commitment,
        &point,
        claimed_eval,
        &opening,
        &mut verifier_transcript,
    );

    assert!(result, "basefold verification failed");
}
