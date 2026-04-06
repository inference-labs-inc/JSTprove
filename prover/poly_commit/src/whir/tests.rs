use arith::Field;
use goldilocks::{Goldilocks, GoldilocksExt2};

use super::types::{WhirCommitment, WhirOpening};

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

fn make_test_data(
    num_vars: usize,
) -> (
    Vec<Goldilocks>,
    Vec<GoldilocksExt2>,
    GoldilocksExt2,
    Vec<Goldilocks>,
    tree::Tree,
    WhirCommitment,
) {
    use super::parameters::WHIR_RATE_LOG;
    use crate::basefold::encoding::rs_encode_with_rate;

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
    (base_evals, point, claimed_eval, codeword, tree, commitment)
}

fn prove_and_verify(
    base_evals: &[Goldilocks],
    codeword: &[Goldilocks],
    tree: &tree::Tree,
    num_vars: usize,
    point: &[GoldilocksExt2],
    commitment: &WhirCommitment,
    claimed_eval: GoldilocksExt2,
) -> bool {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let mut pt = BytesHashTranscript::<SHA256hasher>::new();
    let opening = super::pcs_trait_impl::whir_open_for_test(
        base_evals, codeword, tree, num_vars, point, &mut pt,
    );
    let mut vt = BytesHashTranscript::<SHA256hasher>::new();
    super::pcs_trait_impl::whir_verify_for_test(commitment, point, claimed_eval, &opening, &mut vt)
}

#[test]
fn whir_commit_open_verify_roundtrip() {
    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);
    assert!(prove_and_verify(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &commitment,
        claimed_eval,
    ));
}

#[test]
fn whir_rejects_wrong_evaluation() {
    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);
    let wrong_eval = claimed_eval + GoldilocksExt2::ONE;
    assert!(!prove_and_verify(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &commitment,
        wrong_eval,
    ));
}

#[test]
fn adversarial_wrong_commitment_root() {
    use super::parameters::WHIR_RATE_LOG;
    use crate::basefold::encoding::rs_encode_with_rate;

    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, _) = make_test_data(num_vars);

    let fake_evals: Vec<Goldilocks> = (0..1 << num_vars)
        .map(|i| Goldilocks::from(i as u32 + 999))
        .collect();
    let fake_cw = rs_encode_with_rate(&fake_evals, WHIR_RATE_LOG);
    let fake_tree =
        tree::Tree::compact_new_with_field_elems::<Goldilocks, Goldilocks>(fake_cw.clone());
    let fake_commitment = WhirCommitment {
        root: fake_tree.root(),
        num_vars,
    };

    assert!(
        !prove_and_verify(
            &base_evals,
            &codeword,
            &tree,
            num_vars,
            &point,
            &fake_commitment,
            claimed_eval,
        ),
        "must reject proof against wrong commitment root"
    );
}

#[test]
fn adversarial_tampered_sumcheck_message() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);

    let mut pt = BytesHashTranscript::<SHA256hasher>::new();
    let mut opening = super::pcs_trait_impl::whir_open_for_test(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut pt,
    );

    opening.sumcheck_messages[0].eval_at_1 += GoldilocksExt2::ONE;

    let mut vt = BytesHashTranscript::<SHA256hasher>::new();
    let result = super::pcs_trait_impl::whir_verify_for_test(
        &commitment,
        &point,
        claimed_eval,
        &opening,
        &mut vt,
    );
    assert!(!result, "must reject tampered sumcheck message");
}

#[test]
fn adversarial_tampered_final_poly() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);

    let mut pt = BytesHashTranscript::<SHA256hasher>::new();
    let mut opening = super::pcs_trait_impl::whir_open_for_test(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut pt,
    );

    if !opening.final_poly.is_empty() {
        opening.final_poly[0] += GoldilocksExt2::ONE;
    }

    let mut vt = BytesHashTranscript::<SHA256hasher>::new();
    let result = super::pcs_trait_impl::whir_verify_for_test(
        &commitment,
        &point,
        claimed_eval,
        &opening,
        &mut vt,
    );
    assert!(!result, "must reject tampered final polynomial");
}

#[test]
fn adversarial_swapped_query_proofs() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);

    let mut pt = BytesHashTranscript::<SHA256hasher>::new();
    let mut opening = super::pcs_trait_impl::whir_open_for_test(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut pt,
    );

    if !opening.round_query_proofs.is_empty() {
        let proofs = &mut opening.round_query_proofs[0];
        if proofs.len() >= 2 {
            proofs.swap(0, 1);
        }
    }

    let mut vt = BytesHashTranscript::<SHA256hasher>::new();
    let result = super::pcs_trait_impl::whir_verify_for_test(
        &commitment,
        &point,
        claimed_eval,
        &opening,
        &mut vt,
    );
    assert!(!result, "must reject swapped query proofs");
}

#[test]
fn adversarial_truncated_proof() {
    use gkr_engine::Transcript;
    use gkr_hashers::SHA256hasher;
    use transcript::BytesHashTranscript;

    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);

    let mut pt = BytesHashTranscript::<SHA256hasher>::new();
    let mut opening = super::pcs_trait_impl::whir_open_for_test(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &mut pt,
    );

    if !opening.round_query_proofs.is_empty() {
        opening.round_query_proofs[0].pop();
    }

    let mut vt = BytesHashTranscript::<SHA256hasher>::new();
    let result = super::pcs_trait_impl::whir_verify_for_test(
        &commitment,
        &point,
        claimed_eval,
        &opening,
        &mut vt,
    );
    assert!(!result, "must reject truncated query proofs");
}

#[test]
fn adversarial_wrong_num_vars() {
    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, mut commitment) =
        make_test_data(num_vars);

    commitment.num_vars = num_vars + 1;

    assert!(
        !prove_and_verify(
            &base_evals,
            &codeword,
            &tree,
            num_vars,
            &point,
            &commitment,
            claimed_eval,
        ),
        "must reject mismatched num_vars"
    );
}

#[test]
fn adversarial_zero_polynomial() {
    let num_vars = 8;
    let n = 1 << num_vars;
    let base_evals: Vec<Goldilocks> = vec![Goldilocks::ZERO; n];
    let point: Vec<GoldilocksExt2> = (0..num_vars)
        .map(|i| GoldilocksExt2::from(Goldilocks::from(i as u32 + 10)))
        .collect();
    let claimed_eval = GoldilocksExt2::ZERO;

    use super::parameters::WHIR_RATE_LOG;
    use crate::basefold::encoding::rs_encode_with_rate;
    let codeword = rs_encode_with_rate(&base_evals, WHIR_RATE_LOG);
    let tree = tree::Tree::compact_new_with_field_elems::<Goldilocks, Goldilocks>(codeword.clone());
    let commitment = WhirCommitment {
        root: tree.root(),
        num_vars,
    };

    assert!(prove_and_verify(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &commitment,
        claimed_eval,
    ));

    assert!(!prove_and_verify(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &commitment,
        GoldilocksExt2::ONE,
    ));
}

#[test]
fn fuzz_random_evaluations_all_rejected() {
    use rand::Rng;

    let num_vars = 8;
    let (base_evals, point, claimed_eval, codeword, tree, commitment) = make_test_data(num_vars);

    assert!(prove_and_verify(
        &base_evals,
        &codeword,
        &tree,
        num_vars,
        &point,
        &commitment,
        claimed_eval,
    ));

    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let random_eval = GoldilocksExt2::random_unsafe(&mut rng);
        if random_eval == claimed_eval {
            continue;
        }
        assert!(
            !prove_and_verify(
                &base_evals,
                &codeword,
                &tree,
                num_vars,
                &point,
                &commitment,
                random_eval,
            ),
            "must reject random incorrect evaluation"
        );
    }
}

#[test]
fn fuzz_random_polynomials_verify() {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    for _ in 0..5 {
        let num_vars = 8;
        let n = 1 << num_vars;
        let base_evals: Vec<Goldilocks> = (0..n)
            .map(|_| Goldilocks::random_unsafe(&mut rng))
            .collect();
        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|_| GoldilocksExt2::random_unsafe(&mut rng))
            .collect();
        let ext_evals: Vec<GoldilocksExt2> = base_evals
            .iter()
            .map(|&x| GoldilocksExt2::from(x))
            .collect();
        let claimed_eval = eval_mle_lsb_first(&ext_evals, &point);

        use super::parameters::WHIR_RATE_LOG;
        use crate::basefold::encoding::rs_encode_with_rate;
        let codeword = rs_encode_with_rate(&base_evals, WHIR_RATE_LOG);
        let tree =
            tree::Tree::compact_new_with_field_elems::<Goldilocks, Goldilocks>(codeword.clone());
        let commitment = WhirCommitment {
            root: tree.root(),
            num_vars,
        };

        assert!(
            prove_and_verify(
                &base_evals,
                &codeword,
                &tree,
                num_vars,
                &point,
                &commitment,
                claimed_eval,
            ),
            "random polynomial should verify"
        );
    }
}
