// Copyright 2025 Irreducible Inc.

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
use binius_transcript::ProverTranscript;
use binius_verifier::{
    config::{LOG_WORD_SIZE_BITS, StdChallenger},
    protocols::intmul::{common::IntMulOutput, verify},
};
use itertools::izip;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{prove::IntMulProver, witness::Witness};
use crate::fold_word::fold_words;

type F = BinaryField128bGhash;
type P = PackedBinaryGhash2x128b;

pub fn evaluate_witness(words: &[u64], eval_point: &[F]) -> F {
    let words = words
        .iter()
        .map(|&word| Word::from_u64(word))
        .collect::<Vec<_>>();

    let (prefix, suffix) = eval_point.split_at(LOG_WORD_SIZE_BITS);
    let prefix_tensor = eq_ind_partial_eval::<F>(prefix);
    let suffix_tensor = eq_ind_partial_eval::<F>(suffix);

    let partially_folded_witness = fold_words::<_, F>(&words, prefix_tensor.as_ref());

    inner_product_buffers(&partially_folded_witness, &suffix_tensor)
}

#[test]
fn prove_and_verify() {
    let mut rng = StdRng::seed_from_u64(0);

    const LOG_EXPONENTS: usize = 5;
    const NUM_EXPONENTS: usize = 1 << LOG_EXPONENTS;
    let mut a = Vec::with_capacity(NUM_EXPONENTS);
    let mut b = Vec::with_capacity(NUM_EXPONENTS);
    let mut c_lo = Vec::with_capacity(NUM_EXPONENTS);
    let mut c_hi = Vec::with_capacity(NUM_EXPONENTS);

    for _ in 0..NUM_EXPONENTS {
        let a_i = rng.random_range(1..u64::MAX);
        let b_i = rng.random_range(1..u64::MAX);

        let full_result = (a_i as u128) * (b_i as u128);

        let c_lo_i = full_result as u64;
        let c_hi_i = (full_result >> 64) as u64;

        a.push(a_i);
        b.push(b_i);
        c_lo.push(c_lo_i);
        c_hi.push(c_hi_i);
    }

    let witness = Witness::<P, _, _>::new(LOG_WORD_SIZE_BITS, &a, &b, &c_lo, &c_hi).unwrap();
    // Run prover
    let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
    let mut prover = IntMulProver::new(0, &mut prover_transcript);
    let prove_output = prover.prove(witness).unwrap();

    let IntMulOutput {
        eval_point,
        a_evals,
        b_evals,
        c_lo_evals,
        c_hi_evals,
    } = prove_output.clone();

    // Instead of evaluating each exponent bit column
    // separately, we batch them together with a `z_challenge`
    // and check consistency by evaluating at a single point `consistency_check_eval_point`.
    let z_challenge: Vec<F> = (0..LOG_WORD_SIZE_BITS)
        .map(|_| F::random(&mut rng))
        .collect();
    let z_tensor = eq_ind_partial_eval::<F>(&z_challenge);
    let consistency_check_eval_point = [z_challenge, eval_point].concat();
    let get_consistency_check_eval =
        |evals| izip!(evals, z_tensor.as_ref()).map(|(x, y)| x * y).sum();

    let test_cases = [
        (a, a_evals),
        (b, b_evals),
        (c_lo, c_lo_evals),
        (c_hi, c_hi_evals),
    ];

    for (exponents, evals) in test_cases {
        let expected_eval = evaluate_witness(&exponents, &consistency_check_eval_point);
        let given_eval = get_consistency_check_eval(evals);
        assert_eq!(expected_eval, given_eval);
    }
    // Run verifier
    let mut verifier_transcript = prover_transcript.into_verifier();
    let verify_output =
        verify(LOG_WORD_SIZE_BITS, LOG_EXPONENTS, &mut verifier_transcript).unwrap();

    // Check verifier output is consistent with prover output
    assert_eq!(prove_output, verify_output);
}
