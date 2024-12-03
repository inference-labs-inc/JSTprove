use crate::{
    circuit::{
        config::{BN254Config, Config, GF2Config, M31Config},
        ir::{
            common::rand_gen::{RandomCircuitConfig, RandomRange},
            source::RootCircuit as IrSourceRoot,
        },
    },
    compile::compile,
    field::FieldArith,
    utils::error::Error,
};

fn do_test<C: Config>(mut config: RandomCircuitConfig, seed: RandomRange) {
    for i in seed.min..seed.max {
        config.seed = i;
        let root = IrSourceRoot::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let res = compile(&root);
        match res {
            Ok((ir_hint_normalized, layered_circuit)) => {
                assert_eq!(ir_hint_normalized.validate(), Ok(()));
                assert_eq!(layered_circuit.validate(), Ok(()));
                assert_eq!(ir_hint_normalized.input_size(), root.input_size());
                assert_eq!(
                    layered_circuit.input_size(),
                    ir_hint_normalized.circuits[&0].outputs.len()
                );
                assert_eq!(
                    layered_circuit.num_actual_outputs - layered_circuit.expected_num_output_zeroes,
                    root.circuits[&0].outputs.len() - root.expected_num_output_zeroes
                );
                for _ in 0..5 {
                    let input: Vec<C::CircuitField> = (0..root.input_size())
                        .map(|_| C::CircuitField::random_unsafe(&mut rand::thread_rng()))
                        .collect();
                    match root.eval_unsafe_with_errors(input.clone()) {
                        Ok((src_output, src_cond)) => {
                            let (ir_output, _) = ir_hint_normalized.eval_unsafe(input);
                            let (lc_output, lc_cond) = layered_circuit.eval_unsafe(ir_output);
                            assert_eq!(src_cond, lc_cond);
                            assert_eq!(src_output, lc_output);
                        }
                        Err(e) => match e {
                            Error::UserError(_) => {}
                            Error::InternalError(e) => {
                                panic!("{:?}", e);
                            }
                        },
                    }
                }
            }
            Err(e) => match e {
                Error::UserError(_) => {}
                Error::InternalError(e) => {
                    panic!("{:?}", e);
                }
            },
        }
    }
}

fn do_tests<C: Config>(seed: usize) {
    let config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 10 },
        num_inputs: RandomRange { min: 1, max: 10 },
        num_instructions: RandomRange { min: 1, max: 10 },
        num_constraints: RandomRange { min: 0, max: 10 },
        num_outputs: RandomRange { min: 1, max: 10 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.5,
    };
    do_test::<C>(
        config,
        RandomRange {
            min: 100000 + seed,
            max: 103000 + seed,
        },
    );
    let config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 20 },
        num_inputs: RandomRange { min: 1, max: 3 },
        num_instructions: RandomRange { min: 30, max: 50 },
        num_constraints: RandomRange { min: 0, max: 5 },
        num_outputs: RandomRange { min: 1, max: 3 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.05,
    };
    do_test::<C>(
        config,
        RandomRange {
            min: 200000 + seed,
            max: 201000 + seed,
        },
    );
}

#[test]
fn test_m31() {
    do_tests::<M31Config>(1000000);
}

#[test]
fn test_bn254() {
    do_tests::<BN254Config>(2000000);
}

#[test]
fn test_gf2() {
    do_tests::<GF2Config>(3000000);
}

#[test]
fn deterministic() {
    let mut config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 10 },
        num_inputs: RandomRange { min: 1, max: 10 },
        num_instructions: RandomRange { min: 1, max: 10 },
        num_constraints: RandomRange { min: 0, max: 10 },
        num_outputs: RandomRange { min: 1, max: 10 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.5,
    };
    for i in 100000..103000 {
        config.seed = i;
        let root = IrSourceRoot::<M31Config>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let res = compile(&root);
        let res2 = compile(&root);
        match (res, res2) {
            (
                Ok((ir_hint_normalized, layered_circuit)),
                Ok((ir_hint_normalized2, layered_circuit2)),
            ) => {
                if layered_circuit != layered_circuit2 {
                    println!("========================================================");
                    println!("{}\n", layered_circuit);
                    println!("========================================================");
                    println!("{}\n", layered_circuit2);
                    println!("========================================================");
                    panic!("gg");
                }
                assert_eq!(ir_hint_normalized, ir_hint_normalized2);
            }
            (Err(e), Err(e2)) => {
                assert_eq!(e, e2);
            }
            (Ok(_), Err(_)) => {
                panic!("res is Ok, res2 is Err");
            }
            (Err(_), Ok(_)) => {
                panic!("res is Err, res2 is Ok");
            }
        }
    }
}
