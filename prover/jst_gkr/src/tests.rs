use arith::Field;
use goldilocks::Goldilocks;
use jst_gkr_engine::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::mle::{build_eq_table, eq_eval, evaluate_mle};
use crate::prover::prove;
use crate::sumcheck::{prove_sumcheck, verify_sumcheck};
use crate::transcript::Sha256Transcript;
use crate::verifier::verify;

type F = Goldilocks;

fn rand_field_vec(rng: &mut ChaCha20Rng, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(F::random_unsafe(&mut *rng));
    }
    v
}

#[test]
fn test_eq_eval_boolean() {
    let one = F::ONE;
    let zero = F::ZERO;
    let r = vec![one, zero, one];
    let x = vec![one, zero, one];
    let result = eq_eval(&r, &x);
    assert_eq!(result, F::ONE);

    let x2 = vec![one, one, one];
    let result2 = eq_eval(&r, &x2);
    assert_eq!(result2, F::ZERO);
}

#[test]
fn test_eq_table_consistency() {
    let mut rng = ChaCha20Rng::seed_from_u64(43);
    let n = 3;
    let r = rand_field_vec(&mut rng, n);
    let table = build_eq_table(&r);

    for (idx, entry) in table.iter().enumerate() {
        let bits: Vec<F> = (0..n)
            .map(|j| if (idx >> j) & 1 == 1 { F::ONE } else { F::ZERO })
            .collect();
        let expected = eq_eval(&r, &bits);
        assert_eq!(*entry, expected, "mismatch at index {idx}");
    }
}

#[test]
fn test_evaluate_mle_boolean() {
    let evals: Vec<F> = vec![F::from(3u32), F::from(5u32), F::from(7u32), F::from(11u32)];
    let point = vec![F::ZERO, F::ONE];
    let result = evaluate_mle(&evals, &point);
    assert_eq!(result, F::from(7u32));
}

#[test]
fn test_evaluate_mle_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(44);
    let n = 3;
    let size = 1 << n;
    let evals = rand_field_vec(&mut rng, size);
    let point = rand_field_vec(&mut rng, n);

    let eq_table = build_eq_table(&point);
    let mut expected = F::ZERO;
    for i in 0..size {
        expected += evals[i] * eq_table[i];
    }

    let result = evaluate_mle(&evals, &point);
    assert_eq!(result, expected);
}

#[test]
fn test_sumcheck_known_polynomial() {
    let mut rng = ChaCha20Rng::seed_from_u64(45);
    let n = 3;
    let size = 1 << n;

    let f_vals = rand_field_vec(&mut rng, size);
    let hg_vals = rand_field_vec(&mut rng, size);

    let mut expected_sum = F::ZERO;
    for i in 0..size {
        expected_sum += f_vals[i] * hg_vals[i];
    }

    let mut prover_transcript = Sha256Transcript::default();
    let mut bk_f = f_vals.clone();
    let mut bk_hg = hg_vals.clone();

    prover_transcript.append_field_element(&expected_sum);

    let (sc_proof, challenges) = prove_sumcheck(&mut bk_f, &mut bk_hg, n, &mut prover_transcript);

    assert_eq!(sc_proof.round_polys.len(), n);

    let mut verifier_transcript = Sha256Transcript::default();
    verifier_transcript.append_field_element(&expected_sum);

    let result = verify_sumcheck(expected_sum, &sc_proof, n, &mut verifier_transcript);
    assert!(result.is_some());

    let (final_val, v_challenges) = result.unwrap();
    assert_eq!(challenges, v_challenges);

    let f_at_r = evaluate_mle(&f_vals, &challenges);
    let hg_at_r = evaluate_mle(&hg_vals, &challenges);
    assert_eq!(final_val, f_at_r * hg_at_r);
}

#[test]
fn test_gkr_single_layer_mul_add() {
    let layer = CircuitLayer {
        input_var_num: 1,
        output_var_num: 1,
        mul_gates: vec![MulGate {
            o_id: 0,
            i_ids: [0, 1],
            coef: F::ONE,
        }],
        add_gates: vec![AddGate {
            o_id: 1,
            i_id: 0,
            coef: F::ONE,
        }],
        const_gates: vec![],
    };

    let circuit = LayeredCircuit {
        layers: vec![layer],
    };

    let witness = vec![F::from(3u32), F::from(5u32)];
    let layer_vals = circuit.evaluate(&witness);

    assert_eq!(layer_vals[0], witness);
    assert_eq!(layer_vals[1][0], F::from(15u32));
    assert_eq!(layer_vals[1][1], F::from(3u32));

    let mut prover_transcript = Sha256Transcript::default();
    let proof = prove::<F, Sha256Transcript>(&circuit, &witness, &mut prover_transcript);

    let mut verifier_transcript = Sha256Transcript::default();
    assert!(verify::<F, Sha256Transcript>(
        &circuit,
        &witness,
        &proof,
        &mut verifier_transcript
    ));
}

#[test]
fn test_gkr_addition_only() {
    let layer = CircuitLayer {
        input_var_num: 1,
        output_var_num: 0,
        mul_gates: vec![],
        add_gates: vec![
            AddGate {
                o_id: 0,
                i_id: 0,
                coef: F::ONE,
            },
            AddGate {
                o_id: 0,
                i_id: 1,
                coef: F::ONE,
            },
        ],
        const_gates: vec![],
    };

    let circuit = LayeredCircuit {
        layers: vec![layer],
    };

    let witness = vec![F::from(7u32), F::from(11u32)];
    let layer_vals = circuit.evaluate(&witness);
    assert_eq!(layer_vals[1][0], F::from(18u32));

    let mut prover_transcript = Sha256Transcript::default();
    let proof = prove::<F, Sha256Transcript>(&circuit, &witness, &mut prover_transcript);

    let mut verifier_transcript = Sha256Transcript::default();
    assert!(verify::<F, Sha256Transcript>(
        &circuit,
        &witness,
        &proof,
        &mut verifier_transcript
    ));
}

#[test]
fn test_gkr_two_layer_circuit() {
    let layer_0 = CircuitLayer {
        input_var_num: 2,
        output_var_num: 1,
        mul_gates: vec![
            MulGate {
                o_id: 0,
                i_ids: [0, 1],
                coef: F::ONE,
            },
            MulGate {
                o_id: 1,
                i_ids: [2, 3],
                coef: F::ONE,
            },
        ],
        add_gates: vec![],
        const_gates: vec![],
    };

    let layer_1 = CircuitLayer {
        input_var_num: 1,
        output_var_num: 1,
        mul_gates: vec![],
        add_gates: vec![
            AddGate {
                o_id: 0,
                i_id: 0,
                coef: F::ONE,
            },
            AddGate {
                o_id: 0,
                i_id: 1,
                coef: F::ONE,
            },
        ],
        const_gates: vec![],
    };

    let circuit = LayeredCircuit {
        layers: vec![layer_1, layer_0],
    };

    let witness = vec![F::from(2u32), F::from(3u32), F::from(4u32), F::from(5u32)];
    let layer_vals = circuit.evaluate(&witness);
    assert_eq!(layer_vals[1][0], F::from(6u32));
    assert_eq!(layer_vals[1][1], F::from(20u32));
    assert_eq!(layer_vals[2][0], F::from(26u32));

    let mut prover_transcript = Sha256Transcript::default();
    let proof = prove::<F, Sha256Transcript>(&circuit, &witness, &mut prover_transcript);

    let mut verifier_transcript = Sha256Transcript::default();
    assert!(verify::<F, Sha256Transcript>(
        &circuit,
        &witness,
        &proof,
        &mut verifier_transcript
    ));
}

#[test]
fn test_gkr_const_gates() {
    let layer = CircuitLayer {
        input_var_num: 1,
        output_var_num: 1,
        mul_gates: vec![],
        add_gates: vec![AddGate {
            o_id: 0,
            i_id: 0,
            coef: F::ONE,
        }],
        const_gates: vec![ConstGate {
            o_id: 1,
            coef: F::from(42u32),
        }],
    };

    let circuit = LayeredCircuit {
        layers: vec![layer],
    };

    let witness = vec![F::from(10u32), F::from(0u32)];
    let layer_vals = circuit.evaluate(&witness);
    assert_eq!(layer_vals[1][0], F::from(10u32));
    assert_eq!(layer_vals[1][1], F::from(42u32));

    let mut prover_transcript = Sha256Transcript::default();
    let proof = prove::<F, Sha256Transcript>(&circuit, &witness, &mut prover_transcript);

    let mut verifier_transcript = Sha256Transcript::default();
    assert!(verify::<F, Sha256Transcript>(
        &circuit,
        &witness,
        &proof,
        &mut verifier_transcript
    ));
}

#[test]
fn test_gkr_wrong_witness_rejects() {
    let layer = CircuitLayer {
        input_var_num: 1,
        output_var_num: 1,
        mul_gates: vec![MulGate {
            o_id: 0,
            i_ids: [0, 1],
            coef: F::ONE,
        }],
        add_gates: vec![AddGate {
            o_id: 1,
            i_id: 0,
            coef: F::ONE,
        }],
        const_gates: vec![],
    };

    let circuit = LayeredCircuit {
        layers: vec![layer],
    };

    let witness = vec![F::from(3u32), F::from(5u32)];
    let mut prover_transcript = Sha256Transcript::default();
    let proof = prove::<F, Sha256Transcript>(&circuit, &witness, &mut prover_transcript);

    let wrong_witness = vec![F::from(4u32), F::from(5u32)];
    let mut verifier_transcript = Sha256Transcript::default();
    assert!(!verify::<F, Sha256Transcript>(
        &circuit,
        &wrong_witness,
        &proof,
        &mut verifier_transcript
    ));
}

#[test]
fn test_gkr_truncated_proof_rejects() {
    let layer = CircuitLayer {
        input_var_num: 1,
        output_var_num: 0,
        mul_gates: vec![],
        add_gates: vec![AddGate {
            o_id: 0,
            i_id: 0,
            coef: F::ONE,
        }],
        const_gates: vec![],
    };

    let circuit = LayeredCircuit {
        layers: vec![layer],
    };

    let witness = vec![F::from(7u32), F::from(11u32)];

    let truncated_proof = Proof { data: vec![0u8; 4] };
    let mut verifier_transcript = Sha256Transcript::default();
    assert!(!verify::<F, Sha256Transcript>(
        &circuit,
        &witness,
        &truncated_proof,
        &mut verifier_transcript
    ));
}
