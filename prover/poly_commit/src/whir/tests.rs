use arith::Field;
use goldilocks::{Goldilocks, GoldilocksExt2};

use super::types::WhirCommitment;

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
fn whir_commit_open_verify_roundtrip() {
    use super::parameters::WHIR_RATE_LOG;
    use crate::basefold::encoding::rs_encode_with_rate;
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 8;
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

    let codeword = rs_encode_with_rate(&base_evals, WHIR_RATE_LOG);
    let tree = tree::Tree::compact_new_with_field_elems::<Goldilocks, Goldilocks>(codeword.clone());
    let root = tree.root();
    let commitment = WhirCommitment { root, num_vars };

    let mut prover_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let opening = super::pcs_trait_impl::whir_open_for_test(
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
    let result = super::pcs_trait_impl::whir_verify_for_test(
        &commitment,
        &point,
        claimed_eval,
        &opening,
        &mut verifier_transcript,
    );

    assert!(result, "WHIR verification failed");
}

#[test]
fn whir_rejects_wrong_evaluation() {
    use super::parameters::WHIR_RATE_LOG;
    use crate::basefold::encoding::rs_encode_with_rate;
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 8;
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

    let codeword = rs_encode_with_rate(&base_evals, WHIR_RATE_LOG);
    let tree = tree::Tree::compact_new_with_field_elems::<Goldilocks, Goldilocks>(codeword.clone());
    let root = tree.root();
    let commitment = WhirCommitment { root, num_vars };

    let mut prover_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let opening = super::pcs_trait_impl::whir_open_for_test(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut prover_transcript,
    );

    let mut verifier_transcript = BytesHashTranscript::<SHA256hasher>::new();
    let result = super::pcs_trait_impl::whir_verify_for_test(
        &commitment,
        &point,
        wrong_eval,
        &opening,
        &mut verifier_transcript,
    );

    assert!(!result, "WHIR should reject wrong evaluation");
}
