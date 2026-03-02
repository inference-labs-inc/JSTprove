use crate::circuit::{config::Config, layered::Coef};
use crate::field::FieldArith;
use crate::frontend::CircuitField;
use crate::hints;
use crate::utils::error::Error;

use super::{
    common::{self, EvalResult, IrConfig, RawConstraint},
    expr,
};

#[cfg(test)]
mod tests;

pub mod display;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Instruction<C: Config> {
    LinComb(expr::LinComb<C>),
    Mul(Vec<usize>),
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

#[derive(Debug, Clone)]
pub enum BoolBinOpType {
    Xor,
    Or,
    And,
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
            Instruction::ConstantLike(_) => vec![],
            Instruction::SubCircuitCall { inputs, .. } => inputs.clone(),
            Instruction::CustomGate { inputs, .. } => inputs.clone(),
        }
    }
    fn num_outputs(&self) -> usize {
        match self {
            Instruction::LinComb(_) => 1,
            Instruction::Mul(_) => 1,
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
            Instruction::ConstantLike(coef) => coef.validate(num_public_inputs),
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

pub type Circuit<C> = common::Circuit<Irc<C>>;
pub type RootCircuit<C> = common::RootCircuit<Irc<C>>;
