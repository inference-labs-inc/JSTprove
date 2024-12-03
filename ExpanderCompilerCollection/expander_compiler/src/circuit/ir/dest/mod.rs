use std::collections::{HashMap, HashSet};

use crate::circuit::{config::Config, layered::Coef};
use crate::field::FieldArith;
use crate::hints;
use crate::utils::error::Error;

use super::common::EvalResult;
use super::expr::{Term, VarSpec};
use super::{
    common::{self, Instruction as _, IrConfig, RawConstraint},
    expr::Expression,
};

#[cfg(test)]
pub mod tests;

pub mod display;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Instruction<C: Config> {
    InternalVariable {
        expr: Expression<C>,
    },
    SubCircuitCall {
        sub_circuit_id: usize,
        inputs: Vec<usize>,
        num_outputs: usize,
    },
    ConstantLike {
        value: Coef<C>,
    },
}

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct Irc<C: Config> {
    _a: C,
}
impl<C: Config> IrConfig for Irc<C> {
    type Instruction = Instruction<C>;
    type Constraint = RawConstraint;
    type Config = C;
    const ALLOW_DUPLICATE_SUB_CIRCUIT_INPUTS: bool = false;
    const ALLOW_DUPLICATE_CONSTRAINTS: bool = true;
    const ALLOW_DUPLICATE_OUTPUTS: bool = false;
}

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct IrcRelaxed<C: Config> {
    _a: C,
}
impl<C: Config> IrConfig for IrcRelaxed<C> {
    type Instruction = Instruction<C>;
    type Constraint = RawConstraint;
    type Config = C;
    const ALLOW_DUPLICATE_SUB_CIRCUIT_INPUTS: bool = true;
    const ALLOW_DUPLICATE_CONSTRAINTS: bool = true;
    const ALLOW_DUPLICATE_OUTPUTS: bool = true;
}

impl<C: Config> common::Instruction<C> for Instruction<C> {
    fn inputs(&self) -> Vec<usize> {
        match self {
            Instruction::InternalVariable { expr } => expr.get_vars(),
            Instruction::SubCircuitCall { inputs, .. } => inputs.clone(),
            Instruction::ConstantLike { .. } => Vec::new(),
        }
    }
    fn num_outputs(&self) -> usize {
        match self {
            Instruction::InternalVariable { .. } => 1,
            Instruction::SubCircuitCall { num_outputs, .. } => *num_outputs,
            Instruction::ConstantLike { .. } => 1,
        }
    }
    fn as_sub_circuit_call(&self) -> Option<(usize, &Vec<usize>, usize)> {
        match self {
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => Some((*sub_circuit_id, inputs, *num_outputs)),
            _ => None,
        }
    }
    fn sub_circuit_call(sub_circuit_id: usize, inputs: Vec<usize>, num_outputs: usize) -> Self {
        Instruction::SubCircuitCall {
            sub_circuit_id,
            inputs,
            num_outputs,
        }
    }
    fn replace_vars<F: Fn(usize) -> usize>(&self, f: F) -> Self {
        match self {
            Instruction::InternalVariable { expr } => Instruction::InternalVariable {
                expr: expr.replace_vars(f),
            },
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => Instruction::SubCircuitCall {
                sub_circuit_id: *sub_circuit_id,
                inputs: inputs.iter().map(|&i| f(i)).collect(),
                num_outputs: *num_outputs,
            },
            Instruction::ConstantLike { value } => Instruction::ConstantLike {
                value: value.clone(),
            },
        }
    }
    fn from_kx_plus_b(x: usize, k: C::CircuitField, b: C::CircuitField) -> Self {
        Instruction::InternalVariable {
            expr: Expression::from_terms(vec![Term::new_linear(k, x), Term::new_const(b)]),
        }
    }
    fn validate(&self, num_public_inputs: usize) -> Result<(), Error> {
        match self {
            Instruction::ConstantLike { value } => value.validate(num_public_inputs),
            _ => Ok(()),
        }
    }
    fn eval_unsafe(&self, values: &[C::CircuitField]) -> EvalResult<C> {
        match self {
            Instruction::InternalVariable { expr } => {
                let mut sum = C::CircuitField::zero();
                for term in expr.iter() {
                    match &term.vars {
                        VarSpec::Const => {
                            sum += term.coef;
                        }
                        VarSpec::Linear(i) => {
                            sum += values[*i] * term.coef;
                        }
                        VarSpec::Quad(i, j) => {
                            sum += values[*i] * values[*j] * term.coef;
                        }
                        VarSpec::Custom { gate_type, inputs } => {
                            let args: Vec<C::CircuitField> =
                                inputs.iter().map(|i| values[*i]).collect();
                            sum += hints::stub_impl(*gate_type, &args, 1)[0] * term.coef;
                        }
                        VarSpec::RandomLinear(i) => {
                            sum += values[*i] * Coef::<C>::Random.get_value_unsafe();
                        }
                    }
                }
                EvalResult::Value(sum)
            }
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                ..
            } => EvalResult::SubCircuitCall(*sub_circuit_id, inputs),
            Instruction::ConstantLike { value } => EvalResult::Value(value.get_value_unsafe()),
        }
    }
}

pub type Circuit<C> = common::Circuit<Irc<C>>;
pub type RootCircuit<C> = common::RootCircuit<Irc<C>>;
pub type CircuitRelaxed<C> = common::Circuit<IrcRelaxed<C>>;
pub type RootCircuitRelaxed<C> = common::RootCircuit<IrcRelaxed<C>>;

impl<C: Config> CircuitRelaxed<C> {
    fn solve_duplicates(&self) -> Circuit<C> {
        let mut new_id = vec![0];
        let mut new_instructions: Vec<Instruction<C>> = Vec::new();
        let mut insn_of_var = vec![None];
        let mut new_var_max = self.get_num_inputs_all();
        for i in 1..=self.get_num_inputs_all() {
            new_id.push(i);
            insn_of_var.push(None);
        }
        for insn in self.instructions.iter() {
            let new_insn = match insn.replace_vars(|x| new_id[x]) {
                Instruction::InternalVariable { expr } => {
                    insn_of_var.push(Some(new_instructions.len()));
                    Instruction::InternalVariable { expr }
                }
                Instruction::ConstantLike { value } => {
                    insn_of_var.push(Some(new_instructions.len()));
                    Instruction::ConstantLike { value }
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let mut occured = HashSet::new();
                    let mut new_inputs = Vec::new();
                    for &i in inputs.iter() {
                        if !occured.contains(&i) {
                            occured.insert(i);
                            new_inputs.push(i);
                        } else {
                            let copy_insn = match &insn_of_var[i] {
                                Some(insn_id) => new_instructions[*insn_id].clone(),
                                None => Instruction::InternalVariable {
                                    expr: Expression::new_linear(C::CircuitField::one(), i),
                                },
                            };
                            new_var_max += 1;
                            new_instructions.push(copy_insn);
                            new_inputs.push(new_var_max);
                            insn_of_var.push(None);
                        }
                    }
                    for _ in 0..num_outputs {
                        insn_of_var.push(None);
                    }
                    Instruction::SubCircuitCall {
                        sub_circuit_id,
                        inputs: new_inputs,
                        num_outputs,
                    }
                }
            };
            for _ in 0..new_insn.num_outputs() {
                new_var_max += 1;
                new_id.push(new_var_max);
            }
            new_instructions.push(new_insn);
        }
        let outputs_set: HashSet<usize> = self.outputs.iter().cloned().collect();
        if outputs_set.len() != self.outputs.len() {
            panic!("unexpected situation: duplicate outputs should be removed in previous optimization pass");
        }
        let new_outputs: Vec<usize> = self.outputs.iter().map(|&o| new_id[o]).collect();
        let constraints_set: HashSet<usize> = self.constraints.iter().map(|&c| new_id[c]).collect();
        let new_constraints: Vec<usize> = constraints_set.iter().cloned().collect();
        Circuit {
            instructions: new_instructions,
            outputs: new_outputs,
            constraints: new_constraints,
            num_inputs: self.num_inputs,
        }
    }

    fn export_constraints(
        &self,
        is_root: bool,
        sub_num_add_outputs: &HashMap<usize, usize>,
    ) -> (Self, usize) {
        let mut new_id: Vec<usize> = vec![0; self.get_num_inputs_all() + 1];
        let mut instructions = Vec::new();
        let mut new_var_max = self.get_num_inputs_all();
        let mut add_outputs_sub = Vec::new();
        for (i, new_id_ptr) in new_id
            .iter_mut()
            .enumerate()
            .take(self.get_num_inputs_all() + 1)
            .skip(1)
        {
            *new_id_ptr = i;
        }
        for insn in self.instructions.iter() {
            for _ in 0..insn.num_outputs() {
                new_var_max += 1;
                new_id.push(new_var_max);
            }
            let new_insn = match insn.replace_vars(|x| new_id[x]) {
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let sub_hi = sub_num_add_outputs[&sub_circuit_id];
                    for _ in 0..sub_hi {
                        new_var_max += 1;
                        add_outputs_sub.push(new_var_max);
                    }
                    Instruction::SubCircuitCall {
                        sub_circuit_id,
                        inputs,
                        num_outputs: num_outputs + sub_hi,
                    }
                }
                Instruction::InternalVariable { expr } => Instruction::InternalVariable { expr },
                Instruction::ConstantLike { value } => Instruction::ConstantLike { value },
            };
            instructions.push(new_insn);
        }
        let mut add_outputs: Vec<usize> = self.constraints.iter().map(|&x| new_id[x]).collect();
        let mut outputs: Vec<usize> = if is_root {
            vec![]
        } else {
            self.outputs.iter().map(|x| new_id[*x]).collect()
        };
        let add = add_outputs.len() + add_outputs_sub.len();
        outputs.append(&mut add_outputs);
        outputs.append(&mut add_outputs_sub);
        if is_root {
            outputs.extend(self.outputs.iter().map(|x| new_id[*x]));
        }
        (
            CircuitRelaxed {
                num_inputs: self.num_inputs,
                instructions,
                constraints: vec![],
                outputs,
            },
            add,
        )
    }
}

impl<C: Config> RootCircuitRelaxed<C> {
    pub fn solve_duplicates(&self) -> RootCircuit<C> {
        let mut new_circuits = HashMap::new();
        for (id, circuit) in self.circuits.iter() {
            new_circuits.insert(*id, circuit.solve_duplicates());
        }
        RootCircuit {
            num_public_inputs: self.num_public_inputs,
            expected_num_output_zeroes: self.expected_num_output_zeroes,
            circuits: new_circuits,
        }
    }

    pub fn export_constraints(&self) -> RootCircuitRelaxed<C> {
        let mut exported_circuits = HashMap::new();
        let mut sub_num_add_outputs = HashMap::new();
        let order = self.topo_order();
        for id in order.iter().rev() {
            let circuit: &common::Circuit<IrcRelaxed<C>> = self.circuits.get(id).unwrap();
            let (c, add) = circuit.export_constraints(*id == 0, &sub_num_add_outputs);
            exported_circuits.insert(*id, c);
            sub_num_add_outputs.insert(*id, add);
        }
        let expected_zeroes = self.expected_num_output_zeroes + sub_num_add_outputs[&0];
        RootCircuitRelaxed {
            num_public_inputs: self.num_public_inputs,
            expected_num_output_zeroes: expected_zeroes,
            circuits: exported_circuits,
        }
    }
}

impl<C: Config> RootCircuit<C> {
    pub fn validate_circuit_has_inputs(&self) -> Result<(), Error> {
        for circuit in self.circuits.values() {
            if circuit.num_inputs == 0 {
                return Err(Error::UserError("circuit has no inputs".to_string()));
            }
        }
        Ok(())
    }
}
