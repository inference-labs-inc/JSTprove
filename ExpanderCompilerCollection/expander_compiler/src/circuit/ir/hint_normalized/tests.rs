use rand::{Rng, RngCore};

use super::{
    Instruction::{self, ConstantLike, LinComb, Mul},
    RootCircuit,
};
use crate::field::FieldArith;
use crate::{
    circuit::{
        config::{Config, M31Config as C},
        ir::{common::rand_gen::*, expr},
        layered::Coef,
    },
    hints,
};

type CField = <C as Config>::CircuitField;

#[test]
fn remove_hints_simple() {
    let mut root = super::RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        super::Circuit {
            instructions: vec![
                super::Instruction::Hint {
                    hint_id: 0,
                    inputs: vec![1, 1, 1, 1, 1, 1],
                    num_outputs: 2,
                },
                super::Instruction::Mul(vec![4, 3, 2, 1]),
                super::Instruction::Hint {
                    hint_id: 0,
                    inputs: vec![1, 1, 1, 1, 1, 1],
                    num_outputs: 2,
                },
                super::Instruction::Mul(vec![7, 6, 5, 4, 3, 2, 1]),
                super::Instruction::Hint {
                    hint_id: 0,
                    inputs: vec![1, 1, 1, 1, 1, 1],
                    num_outputs: 2,
                },
                super::Instruction::Mul(vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
            ],
            constraints: vec![1],
            outputs: vec![1],
            num_inputs: 2,
        },
    );
    let (root_hint_less, _) = root.remove_and_export_hints();
    assert_eq!(
        root_hint_less.circuits[&0].instructions,
        vec![
            super::super::hint_less::Instruction::Mul(vec![4, 3, 2, 1]),
            super::super::hint_less::Instruction::Mul(vec![6, 5, 9, 4, 3, 2, 1]),
            super::super::hint_less::Instruction::Mul(vec![8, 7, 10, 6, 5, 9, 4, 3, 2, 1]),
        ]
    );
}

#[test]
fn export_hints_simple() {
    let mut root = super::RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        super::Circuit {
            instructions: vec![
                super::Instruction::SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![1],
                    num_outputs: 1,
                },
                super::Instruction::Hint {
                    hint_id: 0,
                    inputs: vec![1, 1, 1, 1, 1, 1],
                    num_outputs: 2,
                },
            ],
            constraints: vec![1],
            outputs: vec![1],
            num_inputs: 2,
        },
    );
    root.circuits.insert(
        1,
        super::Circuit {
            instructions: vec![super::Instruction::Hint {
                hint_id: 0,
                inputs: vec![1, 1, 1, 1, 1, 1],
                num_outputs: 2,
            }],
            constraints: vec![],
            outputs: vec![1],
            num_inputs: 1,
        },
    );
    let (rr, re) = root.remove_and_export_hints();
    assert_eq!(rr.validate(), Ok(()));
    assert_eq!(re.validate(), Ok(()));
    assert_eq!(rr.input_size(), re.circuits[&0].outputs.len());
    assert_eq!(re.circuits[&1].outputs, vec![1, 2, 3]);
    assert_eq!(re.circuits[&0].outputs, vec![1, 2, 6, 7, 4, 5]);
}

impl<C: Config> RandomInstruction for Instruction<C> {
    fn random_no_sub_circuit(
        mut rnd: impl RngCore,
        num_terms: &RandomRange,
        num_vars: usize,
        num_public_inputs: usize,
    ) -> Self {
        if rnd.gen::<f64>() < 0.2 {
            ConstantLike(Coef::random_no_random(&mut rnd, num_public_inputs))
        } else if rnd.gen::<f64>() < 0.3 {
            LinComb(expr::LinComb {
                terms: (0..num_terms.random(&mut rnd))
                    .map(|_| expr::LinCombTerm {
                        coef: C::CircuitField::from(rnd.next_u32()),
                        var: rnd.next_u64() as usize % num_vars + 1,
                    })
                    .collect(),
                constant: C::CircuitField::from(rnd.next_u32()),
            })
        } else if rnd.gen::<f64>() < 0.4 {
            Mul((0..num_terms.random(&mut rnd).max(2))
                .map(|_| rnd.next_u64() as usize % num_vars + 1)
                .collect())
        } else {
            let (hint_id, num_inputs, num_outputs) = if rnd.gen::<f64>() < 0.5 {
                hints::random_builtin(&mut rnd)
            } else {
                (
                    rnd.next_u64() as usize,
                    num_terms.random(&mut rnd).max(1),
                    num_terms.random(&mut rnd).max(1),
                )
            };
            super::Instruction::Hint {
                hint_id,
                inputs: (0..num_inputs)
                    .map(|_| rnd.next_u64() as usize % num_vars + 1)
                    .collect(),
                num_outputs,
            }
        }
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
fn remove_and_export_random_1() {
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
        config.seed = i + 10000;
        let root = RootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let (root_hint_less, root_exported) = root.remove_and_export_hints();
        assert_eq!(root_hint_less.validate(), Ok(()));
        assert_eq!(root_exported.validate(), Ok(()));
        assert_eq!(
            root_hint_less.input_size(),
            root_exported.circuits[&0].outputs.len()
        );
        let inputs: Vec<CField> = (0..root.input_size())
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out1, cond1) = root.eval_unsafe(inputs.clone());
        let (out_ex, _) = root_exported.eval_unsafe(inputs);
        let (out2, cond2) = root_hint_less.eval_unsafe(out_ex);
        assert_eq!(out1, out2);
        assert_eq!(cond1, cond2);
    }
}

#[test]
fn remove_and_export_random_2() {
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
    for i in 0..3000 {
        config.seed = i + 10000;
        let root = RootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        let (root_hint_less, root_exported) = root.remove_and_export_hints();
        assert_eq!(root_hint_less.validate(), Ok(()));
        assert_eq!(root_exported.validate(), Ok(()));
        assert_eq!(
            root_hint_less.input_size(),
            root_exported.circuits[&0].outputs.len()
        );
        let inputs: Vec<CField> = (0..root.input_size())
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out1, cond1) = root.eval_unsafe(inputs.clone());
        let (out_ex, _) = root_exported.eval_unsafe(inputs);
        let (out2, cond2) = root_hint_less.eval_unsafe(out_ex);
        assert_eq!(out1, out2);
        assert_eq!(cond1, cond2);
    }
}
