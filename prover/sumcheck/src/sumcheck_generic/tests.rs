use super::*;

use arith::Fr;
use ark_std::{rand::thread_rng, test_rng};
use gkr_hashers::{Keccak256hasher, SHA256hasher};
use polynomials::{MultiLinearPoly, MutableMultilinearExtension};
use transcript::BytesHashTranscript;

#[test]
fn test_sumcheck_subroutine() {
    for num_vars in 1..10 {
        let size = 1 << num_vars;

        for num_poly in 1..10 {
            let f_and_g_pairs = (0..num_poly)
                .map(|j| {
                    let f_coeffs = (1..=size)
                        .map(|i| Fr::from(j * 10 + i as u32))
                        .collect::<Vec<_>>();
                    let f = MultiLinearPoly::<Fr>::new(f_coeffs);

                    let g_coeffs = (1..=size)
                        .map(|i| Fr::from(j * 100 + i as u32))
                        .collect::<Vec<_>>();
                    let g = MultiLinearPoly::<Fr>::new(g_coeffs);

                    (f, g)
                })
                .collect::<Vec<_>>();

            let mle_list = SumOfProductsPoly::<Fr> { f_and_g_pairs };

            let asserted_sum = mle_list.sum();

            let mut transcript = BytesHashTranscript::<Keccak256hasher>::new();

            let mut prover = IOPProverState::prover_init(&mle_list);
            let mut verifier = IOPVerifierState::verifier_init(prover.mle_list.num_vars());

            let mut challenge = None;

            for _ in 0..prover.mle_list.num_vars() {
                let prover_msg =
                    IOPProverState::prove_round_and_update_state(&mut prover, &challenge);

                challenge = Some(IOPVerifierState::verify_round_and_update_state(
                    &mut verifier,
                    &prover_msg,
                    &mut transcript,
                ));
            }

            let (verified, subclaim) =
                IOPVerifierState::check_and_generate_subclaim(&verifier, &asserted_sum);
            assert!(verified, "sumcheck verification failed");
            let evals = mle_list.evaluate(&subclaim.point);
            assert!(evals == subclaim.expected_evaluation, "wrong subclaim");
        }
    }
}

#[test]
fn test_sumcheck_e2e() {
    let mut rng = test_rng();

    for num_vars in 1..10 {
        let size = 1 << num_vars;

        for num_poly in 1..10 {
            let f_and_g_pairs = (0..num_poly)
                .map(|_| {
                    let f_coeffs = (1..=size)
                        .map(|_| Fr::random_unsafe(&mut rng))
                        .collect::<Vec<_>>();
                    let f = MultiLinearPoly::<Fr>::new(f_coeffs);

                    let g_coeffs = (1..=size)
                        .map(|_| Fr::random_unsafe(&mut rng))
                        .collect::<Vec<_>>();
                    let g = MultiLinearPoly::<Fr>::new(g_coeffs);

                    (f, g)
                })
                .collect::<Vec<_>>();

            let mle_list = SumOfProductsPoly::<Fr> { f_and_g_pairs };

            let asserted_sum = mle_list.sum();

            // prover
            let mut transcript = BytesHashTranscript::<Keccak256hasher>::new();
            let proof = SumCheck::<Fr>::prove(&mle_list, &mut transcript);

            // verifier
            let mut transcript = BytesHashTranscript::<Keccak256hasher>::new();
            let (verified, subclaim) =
                SumCheck::<Fr>::verify(asserted_sum, &proof, num_vars, &mut transcript);
            assert!(verified, "sumcheck verification failed");
            let evals = mle_list.evaluate(&subclaim.point);
            assert!(evals == subclaim.expected_evaluation, "wrong subclaim");
        }
    }
}

fn test_sumcheck_generic_padding_helper<F: Field, T: Transcript>() {
    let num_polys = 16;
    let max_num_vars = 10;
    let mut rng = thread_rng();

    let mle_list = SumOfProductsPoly {
        f_and_g_pairs: (0..num_polys)
            .map(|i| {
                let num_vars = i % (max_num_vars + 1);
                let poly0 = MultiLinearPoly::<F>::random(num_vars, &mut rng);
                let poly1 = MultiLinearPoly::<F>::random(num_vars, &mut rng);
                (poly0, poly1)
            })
            .collect(),
    };
    let claimed_sum = mle_list.sum();

    let proof = SumCheck::prove(&mle_list, &mut T::new());

    let padded_mle_list = SumOfProductsPoly {
        f_and_g_pairs: mle_list
            .f_and_g_pairs
            .iter()
            .map(|(f, g)| {
                let mut f_padded = f.clone();
                f_padded.lift_to_n_vars(max_num_vars);
                let mut g_padded = g.clone();
                g_padded.lift_to_n_vars(max_num_vars);
                (f_padded, g_padded)
            })
            .collect(),
    };

    let proof_with_padded_mle_list = SumCheck::prove(&padded_mle_list, &mut T::new());

    assert_eq!(proof, proof_with_padded_mle_list);

    let (verified, subclaim) = SumCheck::verify(
        claimed_sum,
        &proof_with_padded_mle_list,
        max_num_vars,
        &mut T::new(),
    );
    assert!(verified, "sumcheck verification failed");
    let evals = mle_list.evaluate(&subclaim.point);
    assert!(evals == subclaim.expected_evaluation, "wrong subclaim");
}

#[test]
fn test_sumcheck_generic_padding() {
    test_sumcheck_generic_padding_helper::<Fr, BytesHashTranscript<SHA256hasher>>();
}
