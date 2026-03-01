use crate::circuit::ir::common::RawConstraint;
use crate::circuit::ir::expr;
use crate::field::FieldArith;
use crate::frontend::CircuitField;
use crate::utils::error::Error;
use crate::{
    circuit::{
        config::Config,
        ir::{
            self,
            expr::{LinComb, LinCombTerm},
        },
        layered::Coef,
    },
    hints::BuiltinHintIds,
};

use super::basic::{
    process_root_circuit, InsnTransformAndExecute, InsnTransformResult, RootBuilder,
};

type IrcIn<C> = ir::source::Irc<C>;
type IrcOut<C> = ir::hint_normalized::Irc<C>;
type InsnIn<C> = ir::source::Instruction<C>;
type InsnOut<C> = ir::hint_normalized::Instruction<C>;
type Builder<'a, C> = super::basic::Builder<'a, C, IrcIn<C>, IrcOut<C>>;

impl<'a, C: Config> Builder<'a, C> {
    fn push_const(&mut self, c: CircuitField<C>) -> usize {
        self.push_insn(InsnOut::ConstantLike(Coef::Constant(c)))
            .unwrap()
    }
    fn push_add(&mut self, a: usize, b: usize) -> usize {
        self.push_insn(InsnOut::LinComb(LinComb {
            terms: vec![
                LinCombTerm {
                    coef: CircuitField::<C>::one(),
                    var: a,
                },
                LinCombTerm {
                    coef: CircuitField::<C>::one(),
                    var: b,
                },
            ],
            constant: CircuitField::<C>::zero(),
        }))
        .unwrap()
    }
    fn push_sub(&mut self, a: usize, b: usize) -> usize {
        self.push_insn(InsnOut::LinComb(LinComb {
            terms: vec![
                LinCombTerm {
                    coef: CircuitField::<C>::one(),
                    var: a,
                },
                LinCombTerm {
                    coef: -CircuitField::<C>::one(),
                    var: b,
                },
            ],
            constant: CircuitField::<C>::zero(),
        }))
        .unwrap()
    }
    fn push_mul(&mut self, a: usize, b: usize) -> usize {
        self.push_insn(InsnOut::Mul(vec![a, b])).unwrap()
    }
    fn copy(&mut self, a: usize) -> InsnTransformResult<C, IrcOut<C>> {
        self.copys(&[a])
    }
    fn copys(&mut self, a: &[usize]) -> InsnTransformResult<C, IrcOut<C>> {
        InsnTransformResult::Vars(a.to_vec())
    }
    fn bool_cond(&mut self, a: usize) -> usize {
        let one = self.push_const(CircuitField::<C>::one());
        let a_minus_one = self.push_sub(a, one);
        self.push_mul(a, a_minus_one)
    }
    fn assert_bool(&mut self, a: usize) {
        let t = self.bool_cond(a);
        self.assert((), t);
    }
    fn mark_bool(&mut self, a: usize) {
        let t = self.bool_cond(a);
        self.mark((), t);
    }
}

impl<'a, C: Config> InsnTransformAndExecute<'a, C, IrcIn<C>, IrcOut<C>> for Builder<'a, C> {
    fn transform_in_to_out(&mut self, in_insn: &InsnIn<C>) -> InsnTransformResult<C, IrcOut<C>> {
        use ir::source::Instruction::*;
        InsnTransformResult::Insn(match in_insn {
            LinComb(lcs) => InsnOut::LinComb(lcs.clone()),
            Mul(vars) => InsnOut::Mul(vars.clone()),
            Div { x, y, checked } => match self.constant_value(*y) {
                Some(yv) => {
                    if yv.is_zero() {
                        return InsnTransformResult::Err(Error::UserError(
                            "division by zero constant".to_string(),
                        ));
                    }
                    let y = self.push_const(yv.inv().unwrap());
                    InsnOut::Mul(vec![*x, y])
                }
                None => {
                    if *checked {
                        let one = self.push_const(CircuitField::<C>::one());
                        let inv = self
                            .push_insn(InsnOut::Hint {
                                hint_id: BuiltinHintIds::Div as u64 as usize,
                                inputs: vec![one, *y],
                                num_outputs: 1,
                            })
                            .unwrap();
                        let multy = self.push_mul(*y, inv);
                        let sub1 = self.push_sub(multy, one);
                        self.assert((), sub1);
                        InsnOut::Mul(vec![*x, inv])
                    } else {
                        let div_res = self
                            .push_insn(InsnOut::Hint {
                                hint_id: BuiltinHintIds::Div as u64 as usize,
                                inputs: vec![*x, *y],
                                num_outputs: 1,
                            })
                            .unwrap();
                        let multy = self.push_mul(*y, div_res);
                        let subx = self.push_sub(multy, *x);
                        self.assert((), subx);
                        return self.copy(div_res);
                    }
                }
            },
            BoolBinOp { x, y, op } => {
                self.assert_bool(*x);
                self.assert_bool(*y);
                let x_plus_y = self.push_add(*x, *y);
                let x_times_y = self.push_mul(*x, *y);
                let res = match op {
                    ir::source::BoolBinOpType::And => x_times_y,
                    ir::source::BoolBinOpType::Or => self.push_sub(x_plus_y, x_times_y),
                    ir::source::BoolBinOpType::Xor => self
                        .push_insn(InsnOut::LinComb(expr::LinComb {
                            terms: vec![
                                LinCombTerm {
                                    coef: CircuitField::<C>::one(),
                                    var: x_plus_y,
                                },
                                LinCombTerm {
                                    coef: -(CircuitField::<C>::one() + CircuitField::<C>::one()),
                                    var: x_times_y,
                                },
                            ],
                            constant: CircuitField::<C>::zero(),
                        }))
                        .unwrap(),
                };
                self.mark_bool(res);
                return self.copy(res);
            }
            IsZero(x) => {
                if let Some(xv) = self.constant_value(*x) {
                    InsnOut::ConstantLike(Coef::Constant(if xv.is_zero() {
                        CircuitField::<C>::one()
                    } else {
                        CircuitField::<C>::zero()
                    }))
                } else {
                    let one = self.push_const(CircuitField::<C>::one());
                    let inv = self
                        .push_insn(InsnOut::Hint {
                            hint_id: BuiltinHintIds::Div as u64 as usize,
                            inputs: vec![one, *x],
                            num_outputs: 1,
                        })
                        .unwrap();
                    let prod = self.push_mul(*x, inv);
                    let m = self.push_sub(one, prod);
                    let xm = self.push_mul(*x, m);
                    self.assert((), xm);
                    self.mark_bool(m);
                    return self.copy(m);
                }
            }
            Commit(_) => {
                // TODO: warn user that this is different from gnark
                InsnOut::ConstantLike(Coef::Random)
            }
            Hint {
                hint_id,
                inputs,
                num_outputs,
            } => ir::hint_normalized::Instruction::Hint {
                hint_id: *hint_id,
                inputs: inputs.clone(),
                num_outputs: *num_outputs,
            },
            ConstantLike(coef) => ir::hint_normalized::Instruction::ConstantLike(*coef),
            SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => ir::hint_normalized::Instruction::SubCircuitCall {
                sub_circuit_id: *sub_circuit_id,
                inputs: inputs.clone(),
                num_outputs: *num_outputs,
            },
            UnconstrainedBinOp { x, y, op } => {
                let xc = self.constant_value(*x);
                let yc = self.constant_value(*y);
                if let (Some(xv), Some(yv)) = (&xc, &yc) {
                    InsnOut::ConstantLike(Coef::Constant(match op.eval(xv, yv) {
                        Ok(v) => v,
                        Err(e) => return InsnTransformResult::Err(e),
                    }))
                } else {
                    use ir::source::UnconstrainedBinOpType::*;
                    let op = match op {
                        Div => BuiltinHintIds::Div,
                        Pow => BuiltinHintIds::Pow,
                        IntDiv => BuiltinHintIds::IntDiv,
                        Mod => BuiltinHintIds::Mod,
                        ShiftL => BuiltinHintIds::ShiftL,
                        ShiftR => BuiltinHintIds::ShiftR,
                        LesserEq => BuiltinHintIds::LesserEq,
                        GreaterEq => BuiltinHintIds::GreaterEq,
                        Lesser => BuiltinHintIds::Lesser,
                        Greater => BuiltinHintIds::Greater,
                        Eq => BuiltinHintIds::Eq,
                        NotEq => BuiltinHintIds::NotEq,
                        BoolOr => BuiltinHintIds::BoolOr,
                        BoolAnd => BuiltinHintIds::BoolAnd,
                        BitOr => BuiltinHintIds::BitOr,
                        BitAnd => BuiltinHintIds::BitAnd,
                        BitXor => BuiltinHintIds::BitXor,
                    };
                    InsnOut::Hint {
                        hint_id: op as usize,
                        inputs: vec![*x, *y],
                        num_outputs: 1,
                    }
                }
            }
            UnconstrainedSelect {
                cond,
                if_true,
                if_false,
            } => {
                if let Some(cond) = self.constant_value(*cond) {
                    if cond.is_zero() {
                        return self.copy(*if_false);
                    } else {
                        return self.copy(*if_true);
                    }
                } else {
                    InsnOut::Hint {
                        hint_id: BuiltinHintIds::Select as u64 as usize,
                        inputs: vec![*cond, *if_true, *if_false],
                        num_outputs: 1,
                    }
                }
            }
            CustomGate { gate_type, inputs } => ir::hint_normalized::Instruction::CustomGate {
                gate_type: *gate_type,
                inputs: inputs.clone(),
            },
            ToBinary { x, num_bits } => {
                let bits = self.push_insn_multi_out(InsnOut::Hint {
                    hint_id: BuiltinHintIds::ToBinary as u64 as usize,
                    inputs: vec![*x],
                    num_outputs: *num_bits,
                });
                let mut sum = self.push_const(CircuitField::<C>::zero());
                let mut coef = CircuitField::<C>::one();
                for &bit in bits.iter() {
                    self.assert_bool(bit);
                    let bit = self
                        .push_insn(InsnOut::LinComb(expr::LinComb {
                            terms: vec![LinCombTerm { coef, var: bit }],
                            constant: CircuitField::<C>::zero(),
                        }))
                        .unwrap();
                    sum = self.push_add(sum, bit);
                    coef = coef.double();
                }
                let sum = self
                    .push_insn(InsnOut::LinComb(expr::LinComb {
                        terms: vec![
                            LinCombTerm {
                                coef: -CircuitField::<C>::one(),
                                var: sum,
                            },
                            LinCombTerm {
                                coef: CircuitField::<C>::one(),
                                var: *x,
                            },
                        ],
                        constant: CircuitField::<C>::zero(),
                    }))
                    .unwrap();
                self.assert((), sum);
                return self.copys(&bits);
            }
        })
    }

    fn transform_in_con_to_out(
        &mut self,
        in_con: &ir::source::Constraint,
    ) -> Result<RawConstraint, Error> {
        match in_con.typ {
            ir::source::ConstraintType::Zero => Ok(in_con.var),
            ir::source::ConstraintType::Bool => Ok(self.bool_cond(in_con.var)),
            ir::source::ConstraintType::NonZero => {
                let one = self.push_const(CircuitField::<C>::one());
                let inv = self
                    .push_insn(InsnOut::Hint {
                        hint_id: BuiltinHintIds::Div as u64 as usize,
                        inputs: vec![one, in_con.var],
                        num_outputs: 1,
                    })
                    .unwrap();
                let multy = self.push_mul(in_con.var, inv);
                let sub1 = self.push_sub(multy, one);
                Ok(sub1)
            }
        }
    }

    fn execute_out<'b>(
        &mut self,
        out_insn: &InsnOut<C>,
        root: Option<&'b RootBuilder<'a, C, IrcIn<C>, IrcOut<C>>>,
    ) where
        'a: 'b,
    {
        match out_insn {
            InsnOut::LinComb(lc) => {
                self.add_lin_comb(lc);
            }
            InsnOut::Mul(inputs) => {
                self.add_mul_vec(inputs.clone());
            }
            InsnOut::Hint { num_outputs, .. } => {
                self.add_out_vars(*num_outputs);
            }
            InsnOut::ConstantLike(coef) => match coef {
                Coef::Constant(c) => {
                    self.add_const(*c);
                }
                Coef::Random => {
                    self.add_out_vars(1);
                }
                Coef::PublicInput(_) => {
                    self.add_out_vars(1);
                }
            },
            InsnOut::SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => {
                self.sub_circuit_call(*sub_circuit_id, inputs, *num_outputs, root);
            }
            InsnOut::CustomGate { .. } => {
                self.add_out_vars(1);
            }
        }
    }
}

pub fn process<C: Config>(
    rc: &ir::common::RootCircuit<IrcIn<C>>,
) -> Result<ir::common::RootCircuit<IrcOut<C>>, Error> {
    process_root_circuit(rc)
}

#[cfg(test)]
mod tests {
    use crate::field::FieldArith;
    use crate::frontend::M31Config as C;
    use crate::{
        circuit::ir::{self, common::rand_gen::*},
        utils::error::Error,
    };

    use mersenne31::M31;

    type CField = M31;

    #[test]
    fn simple_invariant() {
        let mut root = ir::common::RootCircuit::<super::IrcIn<C>>::default();
        let lc = ir::expr::LinComb {
            terms: vec![
                ir::expr::LinCombTerm {
                    coef: CField::one(),
                    var: 1,
                },
                ir::expr::LinCombTerm {
                    coef: CField::one(),
                    var: 2,
                },
            ],
            constant: CField::one(),
        };
        root.circuits.insert(
            0,
            ir::common::Circuit::<super::IrcIn<C>> {
                instructions: vec![ir::source::Instruction::LinComb(lc.clone())],
                constraints: vec![ir::source::Constraint {
                    typ: ir::source::ConstraintType::Zero,
                    var: 3,
                }],
                outputs: vec![],
                num_inputs: 2,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        match &root_processed.circuits[&0].instructions[0] {
            ir::hint_normalized::Instruction::LinComb(lc2) => {
                assert_eq!(lc, *lc2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn random_circuits_1() {
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
            config.seed = i + 200000;
            let root = ir::common::RootCircuit::<super::IrcIn<C>>::random(&config);
            assert_eq!(root.validate(), Ok(()));
            match super::process(&root) {
                Ok(root_processed) => {
                    assert_eq!(root_processed.validate(), Ok(()));
                    assert_eq!(root.input_size(), root_processed.input_size());
                    for _ in 0..5 {
                        let inputs: Vec<CField> = (0..root.input_size())
                            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                            .collect();
                        let e1 = root.eval_unsafe_with_errors(inputs.clone());
                        if e1.is_ok() {
                            let e2 = root_processed.eval_unsafe_with_errors(inputs);
                            assert_eq!(e2, e1);
                        } else if e1.as_ref().err().unwrap().is_internal() {
                            panic!("{:?}", e1);
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

    #[test]
    fn random_circuits_2() {
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
            config.seed = i + 300000;
            let root = ir::common::RootCircuit::<super::IrcIn<C>>::random(&config);
            assert_eq!(root.validate(), Ok(()));
            match super::process(&root) {
                Ok(root_processed) => {
                    assert_eq!(root_processed.validate(), Ok(()));
                    assert_eq!(root.input_size(), root_processed.input_size());
                    for _ in 0..5 {
                        let inputs: Vec<CField> = (0..root.input_size())
                            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                            .collect();
                        let e1 = root.eval_unsafe_with_errors(inputs.clone());
                        if e1.is_ok() {
                            let e2 = root_processed.eval_unsafe_with_errors(inputs);
                            assert_eq!(e2, e1);
                        } else if e1.as_ref().err().unwrap().is_internal() {
                            panic!("{:?}", e1);
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

    #[test]
    fn large_add() {
        let mut root = ir::common::RootCircuit::<super::IrcIn<C>>::default();
        let terms = (1..=100000)
            .map(|i| ir::expr::LinCombTerm {
                coef: CField::one(),
                var: i,
            })
            .collect();
        let lc = ir::expr::LinComb {
            terms,
            constant: CField::one(),
        };
        root.circuits.insert(
            0,
            ir::common::Circuit::<super::IrcIn<C>> {
                instructions: vec![ir::source::Instruction::LinComb(lc.clone())],
                constraints: vec![ir::source::Constraint {
                    typ: ir::source::ConstraintType::Zero,
                    var: 100001,
                }],
                outputs: vec![],
                num_inputs: 100000,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        match &root_processed.circuits[&0].instructions[0] {
            ir::hint_normalized::Instruction::LinComb(lc2) => {
                assert_eq!(lc, *lc2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn large_mul() {
        let mut root = ir::common::RootCircuit::<super::IrcIn<C>>::default();
        let terms: Vec<usize> = (1..=100000).collect();
        root.circuits.insert(
            0,
            ir::common::Circuit::<super::IrcIn<C>> {
                instructions: vec![ir::source::Instruction::Mul(terms.clone())],
                constraints: vec![ir::source::Constraint {
                    typ: ir::source::ConstraintType::Zero,
                    var: 100001,
                }],
                outputs: vec![],
                num_inputs: 100000,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        match &root_processed.circuits[&0].instructions[0] {
            ir::hint_normalized::Instruction::Mul(terms2) => {
                assert_eq!(terms, *terms2);
            }
            _ => panic!(),
        }
    }
}
