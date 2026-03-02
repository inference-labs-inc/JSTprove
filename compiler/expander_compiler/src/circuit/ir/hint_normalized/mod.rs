use std::collections::HashMap;

use crate::field::FieldArith;
use crate::frontend::CircuitField;
use crate::hints::registry::HintCaller;
use crate::utils::error::Error;
use crate::{
    circuit::{
        config::Config,
        input_mapping::{InputMapping, EMPTY},
        layered::Coef,
    },
    hints,
};

use super::{
    common::{self, EvalResult, Instruction as _, IrConfig, RawConstraint},
    expr,
};

#[cfg(test)]
mod tests;

pub mod serde;
pub mod witness_solver;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Instruction<C: Config> {
    LinComb(expr::LinComb<C>),
    Mul(Vec<usize>),
    Hint {
        hint_id: usize,
        inputs: Vec<usize>,
        num_outputs: usize,
    },
    ConstantLike(Coef<C>),
    SubCircuitCall {
        sub_circuit_id: usize,
        inputs: Vec<usize>,
        num_outputs: usize,
    },
    CustomGate {
        gate_type: usize,
        inputs: Vec<usize>,
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
    const ALLOW_DUPLICATE_SUB_CIRCUIT_INPUTS: bool = true;
    const ALLOW_DUPLICATE_CONSTRAINTS: bool = true;
    const ALLOW_DUPLICATE_OUTPUTS: bool = true;
}

impl<C: Config> common::Instruction<C> for Instruction<C> {
    fn inputs(&self) -> Vec<usize> {
        match self {
            Instruction::LinComb(lc) => lc.get_vars(),
            Instruction::Mul(inputs) => inputs.clone(),
            Instruction::Hint { inputs, .. } => inputs.clone(),
            Instruction::ConstantLike(_) => vec![],
            Instruction::SubCircuitCall { inputs, .. } => inputs.clone(),
            Instruction::CustomGate { inputs, .. } => inputs.clone(),
        }
    }
    fn num_outputs(&self) -> usize {
        match self {
            Instruction::LinComb(_) => 1,
            Instruction::Mul(_) => 1,
            Instruction::Hint { num_outputs, .. } => *num_outputs,
            Instruction::ConstantLike(_) => 1,
            Instruction::SubCircuitCall { num_outputs, .. } => *num_outputs,
            Instruction::CustomGate { .. } => 1,
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
            Instruction::LinComb(lc) => Instruction::LinComb(lc.replace_vars(f)),
            Instruction::Mul(inputs) => Instruction::Mul(inputs.iter().map(|i| f(*i)).collect()),
            Instruction::Hint {
                hint_id,
                inputs,
                num_outputs,
            } => Instruction::Hint {
                hint_id: *hint_id,
                inputs: inputs.iter().map(|i| f(*i)).collect(),
                num_outputs: *num_outputs,
            },
            Instruction::ConstantLike(coef) => Instruction::ConstantLike(*coef),
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => Instruction::SubCircuitCall {
                sub_circuit_id: *sub_circuit_id,
                inputs: inputs.iter().map(|i| f(*i)).collect(),
                num_outputs: *num_outputs,
            },
            Instruction::CustomGate { gate_type, inputs } => Instruction::CustomGate {
                gate_type: *gate_type,
                inputs: inputs.iter().map(|i| f(*i)).collect(),
            },
        }
    }
    fn from_kx_plus_b(x: usize, k: CircuitField<C>, b: CircuitField<C>) -> Self {
        Instruction::LinComb(expr::LinComb::from_kx_plus_b(x, k, b))
    }
    fn validate(&self, num_public_inputs: usize) -> Result<(), Error> {
        match self {
            Instruction::Mul(inputs) => {
                if inputs.len() >= 2 {
                    Ok(())
                } else {
                    Err(Error::InternalError(
                        "mul instruction must have at least 2 inputs".to_string(),
                    ))
                }
            }
            Instruction::Hint {
                hint_id,
                inputs,
                num_outputs,
            } => {
                hints::validate_hint(*hint_id, inputs.len(), *num_outputs)?;
                if !inputs.is_empty() {
                    Ok(())
                } else {
                    Err(Error::InternalError(
                        "hint instruction must have at least 1 input".to_string(),
                    ))
                }
            }
            Instruction::ConstantLike(coef) => coef.validate(num_public_inputs),
            Instruction::CustomGate { inputs, .. } => {
                if !inputs.is_empty() {
                    Ok(())
                } else {
                    Err(Error::InternalError(
                        "custom gate instruction must have at least 1 input".to_string(),
                    ))
                }
            }
            _ => Ok(()),
        }
    }
    fn eval_unsafe(&self, values: &[CircuitField<C>]) -> EvalResult<C> {
        match self {
            Instruction::LinComb(lc) => EvalResult::Value(lc.eval(values)),
            Instruction::Mul(inputs) => {
                let mut res = CircuitField::<C>::one();
                for &i in inputs.iter() {
                    res *= values[i];
                }
                EvalResult::Value(res)
            }
            Instruction::Hint {
                hint_id,
                inputs,
                num_outputs,
            } => {
                let outputs = hints::stub_impl(
                    *hint_id,
                    &inputs.iter().map(|i| values[*i]).collect(),
                    *num_outputs,
                );
                EvalResult::Values(outputs)
            }
            Instruction::ConstantLike(coef) => EvalResult::Value(coef.get_value_unsafe()),
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                ..
            } => EvalResult::SubCircuitCall(*sub_circuit_id, inputs),
            Instruction::CustomGate { gate_type, inputs } => {
                let outputs =
                    hints::stub_impl(*gate_type, &inputs.iter().map(|i| values[*i]).collect(), 1);
                EvalResult::Values(outputs)
            }
        }
    }
}

impl<C: Config> Instruction<C> {
    fn eval_safe(
        &self,
        values: &[CircuitField<C>],
        public_inputs: &[CircuitField<C>],
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> EvalResult<C> {
        if let Instruction::ConstantLike(coef) = self {
            return match coef {
                Coef::Constant(c) => EvalResult::Value(*c),
                Coef::PublicInput(i) => EvalResult::Value(public_inputs[*i]),
                Coef::Random => EvalResult::Error(Error::UserError(
                    "random coef occured in witness solver".to_string(),
                )),
            };
        }
        if let Instruction::Hint {
            hint_id,
            inputs,
            num_outputs,
        } = self
        {
            let inputs: Vec<CircuitField<C>> = inputs.iter().map(|i| values[*i]).collect();
            return match hints::safe_impl(hint_caller, *hint_id, &inputs, *num_outputs) {
                Ok(outputs) => EvalResult::Values(outputs),
                Err(e) => EvalResult::Error(e),
            };
        }
        if let Instruction::CustomGate { .. } = self {
            return EvalResult::Error(Error::UserError(
                "CustomGate currently unsupported".to_string(),
            ));
        }
        self.eval_unsafe(values)
    }
}

pub type Circuit<C> = common::Circuit<Irc<C>>;
pub type RootCircuit<C> = common::RootCircuit<Irc<C>>;

impl<C: Config> Circuit<C> {
    fn compute_hint_sizes(
        &self,
        sub_hint_sizes: &HashMap<usize, (usize, usize)>,
    ) -> (usize, usize) {
        let mut res = 0;
        let mut res_self = 0;
        for insn in self.instructions.iter() {
            match insn {
                Instruction::Hint { num_outputs, .. } => {
                    res_self += num_outputs;
                }
                Instruction::SubCircuitCall { sub_circuit_id, .. } => {
                    res += sub_hint_sizes[sub_circuit_id].0;
                }
                _ => {}
            };
        }
        (res + res_self, res_self)
    }

    fn remove_hints(
        &self,
        self_id: usize,
        sub_hint_sizes: &HashMap<usize, (usize, usize)>,
    ) -> super::hint_less::Circuit<C> {
        let mut new_id: Vec<usize> = vec![0; self.get_num_variables() + 1];
        let mut instructions = Vec::new();
        let mut cur_var_max = self.num_inputs;
        let mut new_var_max = self.num_inputs;
        for (i, new_id_ptr) in new_id
            .iter_mut()
            .enumerate()
            .take(self.num_inputs + 1)
            .skip(1)
        {
            *new_id_ptr = i;
        }
        for insn in self.instructions.iter() {
            if let Instruction::Hint { num_outputs, .. } = insn {
                for i in 1..=*num_outputs {
                    new_var_max += 1;
                    new_id[cur_var_max + i] = new_var_max;
                }
            }
            cur_var_max += insn.num_outputs();
        }
        cur_var_max = self.num_inputs;
        let mut sub_hint_ptr = new_var_max;
        new_var_max += sub_hint_sizes[&self_id].0 - sub_hint_sizes[&self_id].1;
        let expected_sub_hint_ptr = new_var_max;
        for insn in self.instructions.iter() {
            match insn {
                Instruction::Hint { num_outputs, .. } => {
                    cur_var_max += *num_outputs;
                }
                _ => {
                    instructions.push(match insn.replace_vars(|x| new_id[x]) {
                        Instruction::ConstantLike(coef) => {
                            super::hint_less::Instruction::ConstantLike(coef)
                        }
                        Instruction::LinComb(lc) => super::hint_less::Instruction::LinComb(lc),
                        Instruction::Mul(inputs) => super::hint_less::Instruction::Mul(inputs),
                        Instruction::SubCircuitCall {
                            sub_circuit_id,
                            mut inputs,
                            num_outputs,
                        } => {
                            for _ in 0..sub_hint_sizes[&sub_circuit_id].0 {
                                sub_hint_ptr += 1;
                                inputs.push(sub_hint_ptr);
                            }
                            super::hint_less::Instruction::SubCircuitCall {
                                sub_circuit_id,
                                inputs,
                                num_outputs,
                            }
                        }
                        Instruction::Hint { .. } => unreachable!(),
                        Instruction::CustomGate { gate_type, inputs } => {
                            super::hint_less::Instruction::CustomGate { gate_type, inputs }
                        }
                    });
                    for _ in 0..insn.num_outputs() {
                        new_var_max += 1;
                        cur_var_max += 1;
                        new_id[cur_var_max] = new_var_max;
                    }
                }
            }
        }
        assert_eq!(sub_hint_ptr, expected_sub_hint_ptr);
        super::hint_less::Circuit {
            num_inputs: self.num_inputs + sub_hint_sizes[&self_id].0,
            instructions,
            constraints: self.constraints.iter().map(|x| new_id[*x]).collect(),
            outputs: self.outputs.iter().map(|x| new_id[*x]).collect(),
        }
    }

    fn export_hints(&self, is_root: bool, sub_hint_sizes: &HashMap<usize, (usize, usize)>) -> Self {
        let mut new_id: Vec<usize> = vec![0; self.get_num_inputs_all() + 1];
        let mut instructions = Vec::new();
        let mut new_var_max = self.num_inputs;
        let mut add_outputs = Vec::new();
        let mut add_outputs_sub = Vec::new();
        for (i, new_id_ptr) in new_id
            .iter_mut()
            .enumerate()
            .take(self.num_inputs + 1)
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
                Instruction::Hint {
                    hint_id,
                    inputs,
                    num_outputs,
                } => {
                    for i in 1..=num_outputs {
                        add_outputs.push(new_var_max - num_outputs + i);
                    }
                    Instruction::Hint {
                        hint_id,
                        inputs,
                        num_outputs,
                    }
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let sub_hi = sub_hint_sizes[&sub_circuit_id].0;
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
                Instruction::ConstantLike(coef) => Instruction::ConstantLike(coef),
                Instruction::LinComb(lc) => Instruction::LinComb(lc),
                Instruction::Mul(inputs) => Instruction::Mul(inputs),
                Instruction::CustomGate { gate_type, inputs } => {
                    Instruction::CustomGate { gate_type, inputs }
                }
            };
            instructions.push(new_insn);
        }
        let mut outputs: Vec<usize> = if is_root {
            (1..=self.num_inputs).collect()
        } else {
            self.outputs.iter().map(|x| new_id[*x]).collect()
        };
        outputs.append(&mut add_outputs);
        outputs.append(&mut add_outputs_sub);
        Circuit {
            num_inputs: self.num_inputs,
            instructions,
            constraints: if is_root { vec![1] } else { vec![] },
            outputs,
        }
    }

    fn add_back_removed_inputs(&self, im: &InputMapping) -> Self {
        let mut new_id: Vec<usize> = vec![0; self.num_inputs + 1];
        let mut instructions = Vec::new();
        let mut new_var_max = im.cur_size();
        for (i, x) in im.mapping().iter().enumerate() {
            if *x != EMPTY {
                new_id[*x + 1] = i + 1;
            }
        }
        for insn in self.instructions.iter() {
            instructions.push(insn.replace_vars(|x| new_id[x]));
            for _ in 0..insn.num_outputs() {
                new_var_max += 1;
                new_id.push(new_var_max);
            }
        }
        Circuit {
            num_inputs: im.cur_size(),
            instructions,
            constraints: self.constraints.iter().map(|x| new_id[*x]).collect(),
            outputs: self.outputs.iter().map(|x| new_id[*x]).collect(),
        }
    }
}

impl<C: Config> RootCircuit<C> {
    pub fn remove_and_export_hints(&self) -> (super::hint_less::RootCircuit<C>, Self) {
        let mut sub_hint_sizes = HashMap::new();
        let order = self.topo_order();
        for id in order.iter().rev() {
            let circuit = self.circuits.get(id).unwrap();
            let hint_size = circuit.compute_hint_sizes(&sub_hint_sizes);
            sub_hint_sizes.insert(*id, hint_size);
        }
        let mut circuits = HashMap::new();
        for (id, circuit) in self.circuits.iter() {
            circuits.insert(*id, circuit.remove_hints(*id, &sub_hint_sizes));
        }
        let removed_root = super::hint_less::RootCircuit {
            num_public_inputs: self.num_public_inputs,
            expected_num_output_zeroes: self.expected_num_output_zeroes,
            circuits,
        };
        let mut exported_circuits = HashMap::new();
        let order = self.topo_order();
        for id in order.iter().rev() {
            let circuit = self.circuits.get(id).unwrap();
            let c = circuit.export_hints(*id == 0, &sub_hint_sizes);
            exported_circuits.insert(*id, c);
        }
        (
            removed_root,
            RootCircuit {
                num_public_inputs: self.num_public_inputs,
                expected_num_output_zeroes: 0,
                circuits: exported_circuits,
            },
        )
    }

    pub fn add_back_removed_inputs(&mut self, im: &InputMapping) {
        let c0 = self.circuits.get(&0).unwrap().add_back_removed_inputs(im);
        self.circuits.insert(0, c0);
    }

    pub fn eval_safe(
        &self,
        inputs: Vec<CircuitField<C>>,
        public_inputs: &[CircuitField<C>],
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> Result<Vec<CircuitField<C>>, Error> {
        assert_eq!(inputs.len(), self.input_size());
        self.eval_sub_safe(&self.circuits[&0], inputs, public_inputs, hint_caller)
    }

    fn eval_sub_safe(
        &self,
        circuit: &Circuit<C>,
        inputs: Vec<CircuitField<C>>,
        public_inputs: &[CircuitField<C>],
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> Result<Vec<CircuitField<C>>, Error> {
        let mut values = vec![CircuitField::<C>::zero(); 1];
        values.extend(inputs);
        for insn in circuit.instructions.iter() {
            match insn.eval_safe(&values, public_inputs, hint_caller) {
                EvalResult::Value(v) => {
                    values.push(v);
                }
                EvalResult::Values(mut vs) => {
                    values.append(&mut vs);
                }
                EvalResult::SubCircuitCall(sub_circuit_id, inputs) => {
                    let res = self.eval_sub_safe(
                        &self.circuits[&sub_circuit_id],
                        inputs.iter().map(|&i| values[i]).collect(),
                        public_inputs,
                        hint_caller,
                    )?;
                    values.extend(res);
                }
                EvalResult::Error(e) => {
                    return Err(e);
                }
            }
        }
        let mut res = Vec::new();
        for &o in circuit.outputs.iter() {
            res.push(values[o]);
        }
        Ok(res)
    }

    pub fn eval_safe_simd<SF: arith::SimdField<Scalar = CircuitField<C>>>(
        &self,
        inputs: Vec<SF>,
        public_inputs: &[SF],
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> Result<Vec<SF>, Error> {
        assert_eq!(inputs.len(), self.input_size());
        let mut result_values = Vec::new();
        self.eval_sub_safe_simd(
            &self.circuits[&0],
            inputs,
            public_inputs,
            hint_caller,
            &mut result_values,
        )?;
        Ok(result_values)
    }

    fn eval_sub_safe_simd<SF: arith::SimdField<Scalar = CircuitField<C>>>(
        &self,
        circuit: &Circuit<C>,
        inputs: Vec<SF>,
        public_inputs: &[SF],
        hint_caller: &impl HintCaller<CircuitField<C>>,
        result_values: &mut Vec<SF>,
    ) -> Result<(), Error> {
        let mut values = vec![SF::zero(); 1];
        values.extend(inputs);
        for insn in circuit.instructions.iter() {
            match insn {
                Instruction::LinComb(lc) => {
                    let res = lc.eval_simd(&values);
                    values.push(res);
                }
                Instruction::Mul(inputs) => {
                    let mut res = values[inputs[0]];
                    for &i in inputs.iter().skip(1) {
                        res *= values[i];
                    }
                    values.push(res);
                }
                Instruction::Hint {
                    hint_id,
                    inputs,
                    num_outputs,
                } => {
                    let mut inputs_scalar = vec![Vec::with_capacity(inputs.len()); SF::PACK_SIZE];
                    for x in inputs.iter().map(|i| values[*i]) {
                        let tmp = x.unpack();
                        for (i, y) in tmp.iter().enumerate() {
                            inputs_scalar[i].push(*y);
                        }
                    }
                    let mut outputs_tmp =
                        vec![CircuitField::<C>::zero(); num_outputs * SF::PACK_SIZE];
                    for (i, inputs) in inputs_scalar.iter().enumerate() {
                        let outputs =
                            hints::safe_impl(hint_caller, *hint_id, inputs, *num_outputs)?;
                        for (j, x) in outputs.iter().enumerate() {
                            outputs_tmp[j * SF::PACK_SIZE + i] = *x;
                        }
                    }
                    for i in 0..*num_outputs {
                        values.push(SF::pack(
                            &outputs_tmp[i * SF::PACK_SIZE..(i + 1) * SF::PACK_SIZE],
                        ));
                    }
                }
                Instruction::ConstantLike(coef) => {
                    let res = match coef {
                        Coef::Constant(c) => SF::one().scale(c),
                        Coef::PublicInput(i) => public_inputs[*i],
                        Coef::Random => {
                            return Err(Error::UserError(
                                "random coef occured in witness solver".to_string(),
                            ))
                        }
                    };
                    values.push(res);
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    ..
                } => {
                    self.eval_sub_safe_simd(
                        &self.circuits[sub_circuit_id],
                        inputs.iter().map(|&i| values[i]).collect(),
                        public_inputs,
                        hint_caller,
                        &mut values,
                    )?;
                }
                Instruction::CustomGate { .. } => {
                    panic!("CustomGate currently unsupported");
                }
            }
        }
        for &o in circuit.outputs.iter() {
            result_values.push(values[o]);
        }
        Ok(())
    }
}
