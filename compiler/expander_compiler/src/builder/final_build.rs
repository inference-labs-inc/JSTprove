use core::panic;
use std::collections::HashMap;

use crate::circuit::ir::common::{Instruction, RawConstraint};
use crate::circuit::ir::expr::Expression;
use crate::circuit::{config::Config, ir, layered::Coef};
use crate::field::FieldArith;
use crate::frontend::CircuitField;
use crate::utils::error::Error;

use super::basic::{
    process_circuit, to_really_single, try_get_really_single_id, ConstraintStatus,
    InsnTransformAndExecute, InsnTransformResult, RootBuilder,
};

type IrcIn<C> = ir::hint_less::Irc<C>;
type IrcOut<C> = ir::hint_less::Irc<C>;
type InsnIn<C> = ir::hint_less::Instruction<C>;
type InsnOut<C> = ir::hint_less::Instruction<C>;
type Builder<'a, C> = super::basic::Builder<'a, C, IrcIn<C>, IrcOut<C>>;

impl<'a, C: Config> InsnTransformAndExecute<'a, C, IrcIn<C>, IrcOut<C>> for Builder<'a, C> {
    fn transform_in_to_out(&mut self, in_insn: &InsnIn<C>) -> InsnTransformResult<C, IrcOut<C>> {
        InsnTransformResult::Insn(in_insn.clone())
    }

    fn transform_in_con_to_out(&mut self, in_con: &RawConstraint) -> Result<RawConstraint, Error> {
        Ok(*in_con)
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
            InsnOut::CustomGate { .. } => self.add_out_vars(1),
        }
    }
}

impl<'a, C: Config> Builder<'a, C> {
    fn export_for_layering(&mut self) -> Result<ir::dest::CircuitRelaxed<C>, Error> {
        let mut last_subc_o_mid_id = 0;
        let mut out_var_max = self.in_circuit.get_num_inputs_all();
        let mut out_insn_id = 0;
        let mut fin_insns: Vec<ir::dest::Instruction<C>> = Vec::new();
        for (expr, status) in self.constraints.entry(()).or_default().iter() {
            match status {
                ConstraintStatus::Marked => {}
                ConstraintStatus::Asserted => {
                    to_really_single(&mut self.mid_vars, expr);
                }
            }
        }
        self.in_circuit.outputs.iter().for_each(|x| {
            to_really_single(&mut self.mid_vars, &self.out_var_exprs[self.in_to_out[*x]]);
        });
        for (i, expr) in self.mid_vars.vec().iter().enumerate().skip(out_var_max + 1) {
            let non_iv = *expr == Expression::new_linear(CircuitField::<C>::one(), i);
            if i <= last_subc_o_mid_id {
                assert!(non_iv);
                continue;
            }
            if !non_iv {
                fin_insns.push(ir::dest::Instruction::InternalVariable { expr: expr.clone() });
                continue;
            }
            let out_id = self.mid_to_out[i].as_ref().unwrap().x;
            while out_var_max != out_id - 1 || self.out_insns[out_insn_id].num_outputs() == 0 {
                if self.out_insns[out_insn_id].num_outputs() != 0 {
                    assert_eq!(self.out_insns[out_insn_id].as_sub_circuit_call(), None);
                }
                out_var_max += self.out_insns[out_insn_id].num_outputs();
                out_insn_id += 1;
            }
            match &self.out_insns[out_insn_id] {
                ir::hint_less::Instruction::LinComb(_) | ir::hint_less::Instruction::Mul(_) => {
                    panic!("unexpected situation");
                }
                ir::hint_less::Instruction::ConstantLike(c) => {
                    fin_insns.push(ir::dest::Instruction::ConstantLike { value: *c });
                }
                ir::hint_less::Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let fin_inputs = inputs
                        .iter()
                        .map(|x| {
                            try_get_really_single_id(&self.mid_vars, &self.out_var_exprs[*x])
                                .unwrap()
                        })
                        .collect();
                    fin_insns.push(ir::dest::Instruction::SubCircuitCall {
                        sub_circuit_id: *sub_circuit_id,
                        inputs: fin_inputs,
                        num_outputs: *num_outputs,
                    });
                    last_subc_o_mid_id = i + num_outputs - 1;
                }
                ir::hint_less::Instruction::CustomGate { gate_type, inputs } => {
                    let fin_inputs = inputs
                        .iter()
                        .map(|x| {
                            try_get_really_single_id(&self.mid_vars, &self.out_var_exprs[*x])
                                .unwrap()
                        })
                        .collect();
                    fin_insns.push(ir::dest::Instruction::InternalVariable {
                        expr: Expression::new_custom(
                            CircuitField::<C>::one(),
                            *gate_type,
                            fin_inputs,
                        ),
                    });
                }
            }
            out_var_max += self.out_insns[out_insn_id].num_outputs();
            out_insn_id += 1;
        }
        // special check for zero outputs sub-circuit calls
        for insn in self.out_insns.iter() {
            if insn.num_outputs() > 0 {
                continue;
            }
            match insn {
                ir::hint_less::Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let fin_inputs = inputs
                        .iter()
                        .map(|x| {
                            try_get_really_single_id(&self.mid_vars, &self.out_var_exprs[*x])
                                .unwrap()
                        })
                        .collect();
                    fin_insns.push(ir::dest::Instruction::SubCircuitCall {
                        sub_circuit_id: *sub_circuit_id,
                        inputs: fin_inputs,
                        num_outputs: *num_outputs,
                    });
                }
                _ => panic!("unexpected situation"),
            }
        }
        let fin_outputs: Vec<usize> = self
            .in_circuit
            .outputs
            .iter()
            .map(|x| {
                try_get_really_single_id(&self.mid_vars, &self.out_var_exprs[self.in_to_out[*x]])
                    .unwrap()
            })
            .collect();
        let mut constraints = Vec::new();
        for (expr, status) in self.constraints[&()].iter() {
            match status {
                ConstraintStatus::Marked => {}
                ConstraintStatus::Asserted => {
                    if let Some(v) = expr.constant_value() {
                        if v.is_zero() {
                            continue;
                        }
                        return Err(Error::UserError(
                            "non-zero constant in constraint".to_string(),
                        ));
                    }
                    constraints.push(try_get_really_single_id(&self.mid_vars, expr).unwrap());
                }
            }
        }
        Ok(ir::dest::CircuitRelaxed {
            outputs: fin_outputs,
            instructions: fin_insns,
            constraints,
            num_inputs: self.in_circuit.num_inputs,
        })
    }
}

pub fn process<'a, C: Config>(
    rc: &'a ir::common::RootCircuit<IrcIn<C>>,
) -> Result<ir::dest::RootCircuitRelaxed<C>, Error> {
    let mut root: RootBuilder<'a, C, IrcIn<C>, IrcOut<C>> = RootBuilder {
        builders: HashMap::new(),
        rc,
        out_circuits: HashMap::new(),
    };
    let order = rc.topo_order();
    for &circuit_id in order.iter().rev() {
        let (new_circuit, final_builder) =
            process_circuit(&mut root, circuit_id, rc.circuits.get(&circuit_id).unwrap())?;
        root.out_circuits.insert(circuit_id, new_circuit);
        root.builders.insert(circuit_id, final_builder);
    }
    let mut out_circuits = HashMap::new();
    for &circuit_id in order.iter().rev() {
        let builder = root.builders.get_mut(&circuit_id).unwrap();
        out_circuits.insert(circuit_id, builder.export_for_layering()?);
    }
    Ok(ir::dest::RootCircuitRelaxed {
        num_public_inputs: rc.num_public_inputs,
        expected_num_output_zeroes: rc.expected_num_output_zeroes,
        circuits: out_circuits,
    })
}

#[cfg(test)]
mod tests {
    use std::vec;

    use mersenne31::M31;

    use crate::field::FieldArith;
    use crate::frontend::M31Config as C;
    use crate::{
        circuit::ir::{
            self,
            common::rand_gen::*,
            expr::{Expression, Term},
        },
        utils::error::Error,
    };

    type CField = M31;

    #[test]
    fn simple_add() {
        let mut root = ir::common::RootCircuit::<super::IrcIn<C>>::default();
        root.circuits.insert(
            0,
            ir::common::Circuit::<super::IrcIn<C>> {
                instructions: vec![ir::hint_less::Instruction::LinComb(ir::expr::LinComb {
                    terms: vec![
                        ir::expr::LinCombTerm {
                            coef: CField::one(),
                            var: 1,
                        },
                        ir::expr::LinCombTerm {
                            coef: CField::from(2),
                            var: 2,
                        },
                    ],
                    constant: CField::from(3),
                })],
                constraints: vec![3],
                outputs: vec![],
                num_inputs: 2,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        let c0 = &root_processed.circuits[&0];
        assert_eq!(
            c0.instructions[1],
            ir::dest::Instruction::InternalVariable {
                expr: Expression::from_terms(vec![
                    Term::new_linear(CField::one(), 1),
                    Term::new_linear(CField::from(2), 2),
                    Term::new_const(CField::from(3))
                ])
            }
        );
        assert_eq!(c0.constraints, vec![4]);
    }

    #[test]
    fn simple_mul() {
        let mut root = ir::common::RootCircuit::<super::IrcIn<C>>::default();
        root.circuits.insert(
            0,
            ir::common::Circuit::<super::IrcIn<C>> {
                instructions: vec![ir::hint_less::Instruction::Mul(vec![1, 2, 3, 4])],
                constraints: vec![5],
                outputs: vec![5],
                num_inputs: 4,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        let root_fin = root_processed.solve_duplicates();
        assert_eq!(root_fin.validate(), Ok(()));
        let (out, _) = root_fin.eval_unsafe(vec![
            CField::from(2),
            CField::from(3),
            CField::from(5),
            CField::from(7),
        ]);
        assert_eq!(out, vec![CField::from(2 * 3 * 5 * 7)]);
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
            config.seed = i + 100000;
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
                        let e2 = root_processed.eval_unsafe_with_errors(inputs);
                        if e1.is_ok() {
                            assert_eq!(e2, e1);
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
                        let e2 = root_processed.eval_unsafe_with_errors(inputs);
                        if e1.is_ok() {
                            assert_eq!(e2, e1);
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
}
