use arith::{FFTField, Field};
use goldilocks::{Goldilocks, GoldilocksExt2};

use super::commit::basefold_commit;
use super::encoding::{
    bit_reverse_slice, compute_twiddle_coeffs, fold_codeword, fold_codeword_first_round, rs_encode,
};
use super::open::basefold_open;
use super::types::RATE_LOG;
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
fn rs_encode_roundtrip_recovers_polynomial() {
    let num_vars = 4;
    let n = 1 << num_vars;

    let evals: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::from(i as u32 + 1)).collect();
    let codeword = rs_encode(&evals);

    assert_eq!(codeword.len(), n << RATE_LOG);

    let recovered = Goldilocks::ifft(&{
        let mut c = codeword.clone();
        bit_reverse_slice(&mut c);
        c
    });
    for i in 0..n {
        assert_eq!(recovered[i], evals[i]);
    }
}

#[test]
fn twiddle_decoded_fold_matches_multilinear_fold() {
    let num_vars = 4;
    let n = 1 << num_vars;

    let base_evals: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::from(i as u32 + 1)).collect();
    let ext_evals: Vec<GoldilocksExt2> = base_evals
        .iter()
        .map(|&x| GoldilocksExt2::from(x))
        .collect();

    let point: Vec<GoldilocksExt2> = (0..num_vars)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u32 + 10)))
        .collect();

    let expected_eval = eval_mle_lsb_first(&ext_evals, &point);

    let codeword = rs_encode(&base_evals);
    let codeword_log = num_vars + RATE_LOG;

    let level = codeword_log - 1;
    let twiddle_coeffs = compute_twiddle_coeffs::<Goldilocks>(level);
    let mut current = fold_codeword_first_round(&codeword, point[0], &twiddle_coeffs);

    for round in 1..num_vars {
        let level = codeword_log - 1 - round;
        let twiddle_coeffs = compute_twiddle_coeffs::<Goldilocks>(level);
        current = fold_codeword(&current, point[round], &twiddle_coeffs);
    }

    assert_eq!(current.len(), 1 << RATE_LOG);

    let coeffs_recovered = GoldilocksExt2::ifft(&{
        let mut c = current.clone();
        super::encoding::bit_reverse_slice(&mut c);
        c
    });
    assert_eq!(coeffs_recovered[0], expected_eval);
}

#[test]
fn basefold_commit_open_verify_roundtrip() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 6;
    let n = 1 << num_vars;

    let base_evals: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::from(i as u32 + 1)).collect();

    let point: Vec<GoldilocksExt2> = (0..num_vars)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u32 + 10)))
        .collect();

    let ext_evals: Vec<GoldilocksExt2> = base_evals
        .iter()
        .map(|&x| GoldilocksExt2::from(x))
        .collect();
    let claimed_eval = eval_mle_lsb_first(&ext_evals, &point);

    let (commitment, tree, codeword) = basefold_commit(&base_evals);

    let mut prover_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let opening = basefold_open::<Goldilocks, GoldilocksExt2>(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut prover_transcript,
    );

    assert!(!opening.final_poly.is_empty());
    assert_eq!(opening.sumcheck_messages.len(), num_vars);

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

#[test]
fn basefold_rejects_wrong_evaluation() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 4;
    let n = 1 << num_vars;

    let base_evals: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::from(i as u32 + 1)).collect();

    let point: Vec<GoldilocksExt2> = (0..num_vars)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u32 + 10)))
        .collect();

    let ext_evals: Vec<GoldilocksExt2> = base_evals
        .iter()
        .map(|&x| GoldilocksExt2::from(x))
        .collect();
    let real_eval = eval_mle_lsb_first(&ext_evals, &point);
    let wrong_eval = real_eval + GoldilocksExt2::ONE;

    let (commitment, tree, codeword) = basefold_commit(&base_evals);

    let mut prover_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let opening = basefold_open::<Goldilocks, GoldilocksExt2>(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut prover_transcript,
    );

    let mut verifier_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let result = basefold_verify::<Goldilocks, GoldilocksExt2>(
        &commitment,
        &point,
        wrong_eval,
        &opening,
        &mut verifier_transcript,
    );

    assert!(!result, "basefold should reject wrong evaluation");
}
