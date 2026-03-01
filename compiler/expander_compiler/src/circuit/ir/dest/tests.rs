use mersenne31::M31;
use rand::{Rng, RngCore};

use super::{
    Circuit, CircuitRelaxed,
    Instruction::{self, ConstantLike, InternalVariable, SubCircuitCall},
    RootCircuit, RootCircuitRelaxed,
};
use crate::field::FieldArith;
use crate::frontend::M31Config as C;
use crate::{
    circuit::{
        config::Config,
        ir::{
            common::rand_gen::*,
            expr::{Expression, Term},
        },
        layered::Coef,
    },
    frontend::CircuitField,
};

type CField = M31;

#[test]
fn validate_vars() {
    let mut root = RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        Circuit {
            instructions: vec![
                ConstantLike {
                    value: Coef::Constant(CField::one()),
                },
                InternalVariable {
                    expr: Expression::from_terms(vec![Term::new_linear(CField::one(), 4)]),
                },
            ],
            constraints: vec![2],
            outputs: vec![4],
            num_inputs: 2,
        },
    );
    assert!(root.validate().is_err());
    root.circuits.get_mut(&0).unwrap().instructions[1] = InternalVariable {
        expr: Expression::from_terms(vec![Term::new_linear(CField::one(), 3)]),
    };
    assert!(root.validate().is_ok());
    root.circuits.get_mut(&0).unwrap().constraints[0] = 10;
    assert!(root.validate().is_err());
    root.circuits.get_mut(&0).unwrap().constraints[0] = 2;
    assert!(root.validate().is_ok());
    root.circuits.get_mut(&0).unwrap().outputs[0] = 10;
    assert!(root.validate().is_err());
}

#[test]
fn validate_vars_dup() {
    let mut root = RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        Circuit {
            instructions: vec![
                ConstantLike {
                    value: Coef::Constant(CField::one()),
                },
                InternalVariable {
                    expr: Expression::from_terms(vec![Term::new_linear(CField::one(), 1)]),
                },
            ],
            constraints: vec![2],
            outputs: vec![1, 1],
            num_inputs: 2,
        },
    );
    assert!(root.validate().is_err());
    root.circuits.get_mut(&0).unwrap().outputs[0] = 2;
    assert_eq!(root.validate(), Ok(()));
}

#[test]
fn validate_sub_circuit() {
    let mut root = RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        Circuit {
            instructions: vec![SubCircuitCall {
                sub_circuit_id: 1,
                inputs: vec![1, 2],
                num_outputs: 2,
            }],
            constraints: vec![4],
            outputs: vec![],
            num_inputs: 2,
        },
    );
    root.circuits.insert(
        1,
        Circuit {
            instructions: vec![],
            constraints: vec![],
            outputs: vec![1, 2],
            num_inputs: 2,
        },
    );
    assert!(root.validate().is_ok());
    let mut r2 = root.clone();
    r2.circuits.get_mut(&1).unwrap().num_inputs = 1;
    assert!(r2.validate().is_err());
    let mut r2 = root.clone();
    r2.circuits.get_mut(&0).unwrap().instructions[0] = SubCircuitCall {
        sub_circuit_id: 1,
        inputs: vec![1, 3],
        num_outputs: 3,
    };
    assert!(r2.validate().is_err());
    let mut r2 = root.clone();
    r2.circuits.get_mut(&0).unwrap().instructions[0] = SubCircuitCall {
        sub_circuit_id: 10,
        inputs: vec![1, 3],
        num_outputs: 2,
    };
    assert!(r2.validate().is_err());
    let mut r2 = root.clone();
    r2.circuits.get_mut(&0).unwrap().instructions[0] = SubCircuitCall {
        sub_circuit_id: 0,
        inputs: vec![1, 3],
        num_outputs: 3,
    };
    assert!(r2.validate().is_err());
}

#[test]
fn eval_output() {
    let mut root = RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        Circuit {
            instructions: vec![
                SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![1, 2, 3, 4],
                    num_outputs: 1,
                },
                SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![8, 5, 6, 7],
                    num_outputs: 1,
                },
            ],
            constraints: vec![1],
            outputs: vec![9],
            num_inputs: 7,
        },
    );
    root.circuits.insert(
        1,
        Circuit {
            instructions: vec![
                SubCircuitCall {
                    sub_circuit_id: 2,
                    inputs: vec![1, 3],
                    num_outputs: 1,
                },
                SubCircuitCall {
                    sub_circuit_id: 2,
                    inputs: vec![5, 4],
                    num_outputs: 1,
                },
                InternalVariable {
                    expr: Expression::from_terms(vec![
                        Term::new_linear(CField::one(), 6),
                        Term::new_linear(CField::from(9), 2),
                    ]),
                },
            ],
            constraints: vec![],
            outputs: vec![7],
            num_inputs: 4,
        },
    );
    root.circuits.insert(
        2,
        Circuit {
            instructions: vec![InternalVariable {
                expr: Expression::from_terms(vec![
                    Term::new_quad(CField::one(), 1, 1),
                    Term::new_linear(CField::from(10), 2),
                ]),
            }],
            constraints: vec![],
            outputs: vec![3],
            num_inputs: 2,
        },
    );
    assert_eq!(root.validate(), Ok(()));
    assert_eq!(root.input_size(), 7);
    let fn2 = |x: CField, y: CField| x * x + y * CField::from(10);
    let fn1 = |x: CField, r: &[CField]| fn2(fn2(x, r[1]), r[2]) + r[0] * CField::from(9);
    let fn0 = |x: CField, r: &[CField]| fn1(fn1(x, &r[0..3]), &r[3..]);
    for _ in 0..100 {
        let inputs: Vec<CField> = (0..7)
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (res, cond) = root.eval_unsafe(inputs.clone());
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], fn0(inputs[0], &inputs[1..]));
        assert!(!cond);
    }
}

#[test]
fn eval_constraint() {
    let mut root = RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        Circuit {
            instructions: vec![SubCircuitCall {
                sub_circuit_id: 1,
                inputs: vec![2],
                num_outputs: 0,
            }],
            constraints: vec![1],
            outputs: vec![],
            num_inputs: 2,
        },
    );
    root.circuits.insert(
        1,
        Circuit {
            instructions: vec![],
            constraints: vec![1],
            outputs: vec![],
            num_inputs: 1,
        },
    );
    assert_eq!(root.validate(), Ok(()));
    assert_eq!(root.input_size(), 2);
    for i in 0..2 {
        for j in 0..2 {
            let (_, cond) = root.eval_unsafe(vec![CField::from(i), CField::from(j)]);
            assert_eq!(i == 0 && j == 0, cond);
        }
    }
}

impl<C: Config> RandomInstruction for Instruction<C> {
    fn random_no_sub_circuit(
        mut rnd: impl RngCore,
        num_terms: &RandomRange,
        num_vars: usize,
        num_public_inputs: usize,
    ) -> Self {
        if rnd.gen::<f64>() < 0.3 {
            ConstantLike {
                value: Coef::random_no_random(&mut rnd, num_public_inputs),
            }
        } else {
            let mut terms = Vec::with_capacity(num_terms.random(&mut rnd));
            for _ in 0..terms.capacity() {
                let coef = CircuitField::<C>::from(rnd.next_u32());
                let op = rnd.next_u64() as usize % 3;
                terms.push(match op {
                    0 => Term::new_const(coef),
                    1 => Term::new_linear(coef, rnd.next_u64() as usize % num_vars + 1),
                    2 => Term::new_quad(
                        coef,
                        rnd.next_u64() as usize % num_vars + 1,
                        rnd.next_u64() as usize % num_vars + 1,
                    ),
                    _ => unreachable!(),
                });
            }
            InternalVariable {
                expr: Expression::from_terms(terms),
            }
        }
    }
}

#[test]
fn test_random_generator() {
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
    for i in 0..100 {
        config.seed = i;
        let root = RootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        root.eval_unsafe(
            (0..root.input_size())
                .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                .collect(),
        );
    }
}

#[test]
fn opt_remove_unreachable() {
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
    for i in 0..3000 {
        config.seed = i;
        let root = RootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let (optroot, im) = root.remove_unreachable();
        assert_eq!(im.cur_size(), root.input_size());
        assert_eq!(optroot.validate(), Ok(()));
        let inputs: Vec<CField> = (0..root.input_size())
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out1, cond1) = root.eval_unsafe(inputs.clone());
        let (out2, cond2) = optroot.eval_unsafe(im.map_inputs(&inputs));
        assert_eq!(out1, out2);
        assert_eq!(cond1, cond2);
    }
}

#[test]
fn opt_remove_unreachable_2() {
    let mut config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 20 },
        num_inputs: RandomRange { min: 1, max: 3 },
        num_instructions: RandomRange { min: 30, max: 50 },
        num_constraints: RandomRange { min: 0, max: 5 },
        num_outputs: RandomRange { min: 1, max: 3 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.05,
    };
    for i in 0..1000 {
        config.seed = i;
        let root = RootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let (optroot, im) = root.remove_unreachable();
        assert_eq!(im.cur_size(), root.input_size());
        assert_eq!(optroot.validate(), Ok(()));
        let inputs: Vec<CField> = (0..root.input_size())
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out1, cond1) = root.eval_unsafe(inputs.clone());
        let (out2, cond2) = optroot.eval_unsafe(im.map_inputs(&inputs));
        assert_eq!(out1, out2);
        assert_eq!(cond1, cond2);
    }
}

#[test]
fn opt_remove_unreachable_relaxed() {
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
    for i in 0..3000 {
        config.seed = i;
        let root = RootCircuitRelaxed::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let (optroot, im) = root.remove_unreachable();
        assert_eq!(im.cur_size(), root.input_size());
        assert_eq!(optroot.validate(), Ok(()));
        let inputs: Vec<CField> = (0..root.input_size())
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out1, cond1) = root.eval_unsafe(inputs.clone());
        let (out2, cond2) = optroot.eval_unsafe(im.map_inputs(&inputs));
        assert_eq!(out1, out2);
        assert_eq!(cond1, cond2);
    }
}

#[test]
fn adjust_for_layering() {
    let mut root = RootCircuitRelaxed::<C>::default();
    root.circuits.insert(
        0,
        CircuitRelaxed {
            instructions: vec![
                SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![1, 1],
                    num_outputs: 1,
                },
                SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![2, 2],
                    num_outputs: 1,
                },
                ConstantLike {
                    value: Coef::Constant(CField::one()),
                },
                SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![4, 4],
                    num_outputs: 1,
                },
            ],
            constraints: vec![1],
            outputs: vec![],
            num_inputs: 1,
        },
    );
    root.circuits.insert(
        1,
        CircuitRelaxed {
            instructions: vec![],
            constraints: vec![],
            outputs: vec![1],
            num_inputs: 2,
        },
    );
    assert_eq!(root.validate(), Ok(()));
    let r2 = root.solve_duplicates();
    assert_eq!(r2.validate(), Ok(()));
    assert_eq!(
        r2.circuits[&0].instructions,
        vec![
            InternalVariable {
                expr: Expression::new_linear(CField::one(), 1)
            },
            SubCircuitCall {
                sub_circuit_id: 1,
                inputs: vec![1, 2],
                num_outputs: 1,
            },
            InternalVariable {
                expr: Expression::new_linear(CField::one(), 3)
            },
            SubCircuitCall {
                sub_circuit_id: 1,
                inputs: vec![3, 4],
                num_outputs: 1,
            },
            ConstantLike {
                value: Coef::Constant(CField::one()),
            },
            ConstantLike {
                value: Coef::Constant(CField::one()),
            },
            SubCircuitCall {
                sub_circuit_id: 1,
                inputs: vec![6, 7],
                num_outputs: 1,
            },
        ]
    );
}

#[test]
fn adjust_for_layering_and_reassign_duplicate_sub_circuit_outputs() {
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
    for i in 0..100 {
        config.seed = i + 100000;
        let root = RootCircuitRelaxed::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let inputs: Vec<CField> = (0..root.input_size())
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out1, cond1) = root.eval_unsafe(inputs.clone());
        let mut roota = root.clone();
        roota.reassign_duplicate_sub_circuit_outputs(false);
        assert_eq!(roota.validate(), Ok(()));
        let (out2, cond2) = roota.eval_unsafe(inputs.clone());
        assert_eq!(out1, out2);
        assert_eq!(cond1, cond2);
        let (ropt, im) = roota.remove_unreachable();
        assert_eq!(ropt.validate(), Ok(()));
        let (out3, cond3) = ropt.eval_unsafe(im.map_inputs(&inputs));
        assert_eq!(out1, out3);
        assert_eq!(cond1, cond3);
        let r2 = ropt.solve_duplicates();
        assert_eq!(r2.validate(), Ok(()));
        let (out4, cond4) = r2.eval_unsafe(im.map_inputs(&inputs));
        assert_eq!(out1, out4);
        assert_eq!(cond1, cond4);
    }
}
