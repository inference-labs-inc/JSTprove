use mersenne31::M31;
use rand::{Rng, RngCore};

use super::{
    Circuit, ConstraintType,
    Instruction::{self, ConstantLike, LinComb, Mul},
    RootCircuit,
};
use crate::frontend::M31Config as C;
use crate::{
    circuit::{
        config::Config,
        ir::{common::rand_gen::*, expr},
        layered::Coef,
    },
    hints,
};
use crate::{field::FieldArith, frontend::CircuitField};

type CField = M31;

impl<C: Config> RandomInstruction for Instruction<C> {
    fn random_no_sub_circuit(
        mut rnd: impl RngCore,
        num_terms: &RandomRange,
        num_vars: usize,
        num_public_inputs: usize,
    ) -> Self {
        let prob1 = rnd.gen::<f64>();
        if prob1 < 0.1 {
            ConstantLike(Coef::random_no_random(&mut rnd, num_public_inputs))
        } else if prob1 < 0.35 {
            LinComb(expr::LinComb {
                terms: (0..num_terms.random(&mut rnd))
                    .map(|_| expr::LinCombTerm {
                        coef: CircuitField::<C>::from(rnd.next_u32()),
                        var: rnd.next_u64() as usize % num_vars + 1,
                    })
                    .collect(),
                constant: CircuitField::<C>::from(rnd.next_u32()),
            })
        } else if prob1 < 0.58 {
            Mul((0..num_terms.random(&mut rnd).max(2))
                .map(|_| rnd.next_u64() as usize % num_vars + 1)
                .collect())
        } else if prob1 < 0.66 {
            let (hint_id, num_inputs, num_outputs) = if rnd.gen::<f64>() < 0.5 {
                hints::random_builtin(&mut rnd)
            } else {
                (
                    rnd.next_u64() as usize,
                    num_terms.random(&mut rnd).max(1),
                    num_terms.random(&mut rnd).max(1),
                )
            };
            if rnd.gen::<f64>() < 0.8 || num_outputs != 1 {
                super::Instruction::Hint {
                    hint_id,
                    inputs: (0..num_inputs)
                        .map(|_| rnd.next_u64() as usize % num_vars + 1)
                        .collect(),
                    num_outputs,
                }
            } else {
                super::Instruction::CustomGate {
                    gate_type: hint_id,
                    inputs: (0..num_inputs)
                        .map(|_| rnd.next_u64() as usize % num_vars + 1)
                        .collect(),
                }
            }
        } else if prob1 < 0.74 {
            super::Instruction::UnconstrainedSelect {
                cond: rnd.next_u64() as usize % num_vars + 1,
                if_true: rnd.next_u64() as usize % num_vars + 1,
                if_false: rnd.next_u64() as usize % num_vars + 1,
            }
        } else if prob1 < 0.8 {
            super::Instruction::Div {
                x: rnd.next_u64() as usize % num_vars + 1,
                y: rnd.next_u64() as usize % num_vars + 1,
                checked: rnd.gen::<f64>() < 0.5,
            }
        } else if prob1 < 0.83 {
            super::Instruction::IsZero(rnd.next_u64() as usize % num_vars + 1)
        } else if prob1 < 0.86 {
            let op = match rnd.next_u64() % 4 {
                0 => super::UnconstrainedBinOpType::Eq,
                1 => super::UnconstrainedBinOpType::NotEq,
                2 => super::UnconstrainedBinOpType::BoolAnd,
                3 => super::UnconstrainedBinOpType::BoolOr,
                _ => unreachable!(),
            };
            super::Instruction::UnconstrainedBinOp {
                x: rnd.next_u64() as usize % num_vars + 1,
                y: rnd.next_u64() as usize % num_vars + 1,
                op,
            }
        } else if prob1 < 0.91 {
            super::Instruction::UnconstrainedBinOp {
                x: rnd.next_u64() as usize % num_vars + 1,
                y: rnd.next_u64() as usize % num_vars + 1,
                op: super::UnconstrainedBinOpType::Div,
            }
        } else if prob1 < 0.98 {
            let op = match rnd.next_u64() % 3 {
                0 => super::UnconstrainedBinOpType::BitAnd,
                1 => super::UnconstrainedBinOpType::BitOr,
                2 => super::UnconstrainedBinOpType::BitXor,
                _ => unreachable!(),
            };
            super::Instruction::UnconstrainedBinOp {
                x: rnd.next_u64() as usize % num_vars + 1,
                y: rnd.next_u64() as usize % num_vars + 1,
                op,
            }
        } else {
            super::Instruction::ToBinary {
                x: rnd.next_u64() as usize % num_vars + 1,
                num_bits: [1, 3, 66, 266, 267, 268, 270, 300, 300, 300]
                    [rnd.next_u64() as usize % 10],
            }
        }
    }
}

impl RandomConstraintType for ConstraintType {
    fn random(mut r: impl RngCore) -> Self {
        match r.next_u64() % 3 {
            0 => ConstraintType::Zero,
            1 => ConstraintType::NonZero,
            2 => ConstraintType::Bool,
            _ => unreachable!(),
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
        let e1 = root.eval_unsafe_with_errors(inputs.clone());
        let e2 = optroot.eval_unsafe_with_errors(im.map_inputs(&inputs));
        if e1.is_ok() {
            assert_eq!(e2, e1);
        } else if e1.as_ref().err().unwrap().is_internal() {
            panic!("{:?}", e1);
        }
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
        let e1 = root.eval_unsafe_with_errors(inputs.clone());
        let e2 = optroot.eval_unsafe_with_errors(im.map_inputs(&inputs));
        if e1.is_ok() {
            assert_eq!(e2, e1);
        } else if e1.as_ref().err().unwrap().is_internal() {
            panic!("{:?}", e1);
        }
    }
}

fn test_detect_chains_inner(is_mul: bool, seq_typ: usize) {
    let n = 1000000;
    let mut root = RootCircuit::<C>::default();
    let mut insns = vec![];
    let mut lst = 1;
    let get_insn = if is_mul {
        |x, y| Instruction::<C>::Mul(vec![x, y])
    } else {
        |x, y| {
            Instruction::LinComb(expr::LinComb {
                terms: vec![
                    expr::LinCombTerm {
                        coef: CField::one(),
                        var: x,
                    },
                    expr::LinCombTerm {
                        coef: CField::one(),
                        var: y,
                    },
                ],
                constant: CField::zero(),
            })
        }
    };
    if seq_typ == 1 {
        lst = n;
        for i in (1..n).rev() {
            insns.push(get_insn(lst, i));
            lst = n * 2 - i;
        }
    } else if seq_typ == 2 {
        for i in 2..=n {
            insns.push(get_insn(lst, i));
            lst = n - 1 + i;
        }
    } else {
        let mut q: Vec<usize> = (1..=n).collect();
        let mut i = 0;
        lst = n;
        while i + 1 < q.len() {
            lst += 1;
            insns.push(get_insn(q[i], q[i + 1]));
            q.push(lst);
            i += 2;
        }
    }
    root.circuits.insert(
        0,
        Circuit::<C> {
            num_inputs: n,
            instructions: insns,
            constraints: vec![],
            outputs: vec![lst],
        },
    );
    assert_eq!(root.validate(), Ok(()));
    root.detect_chains();
    let (root, _) = root.remove_unreachable();
    println!("{:?}", root);
    assert_eq!(root.validate(), Ok(()));
}

#[test]
fn test_detect_chains() {
    test_detect_chains_inner(false, 1);
    test_detect_chains_inner(false, 2);
    test_detect_chains_inner(false, 3);
    test_detect_chains_inner(true, 1);
    test_detect_chains_inner(true, 2);
    test_detect_chains_inner(true, 3);
}
