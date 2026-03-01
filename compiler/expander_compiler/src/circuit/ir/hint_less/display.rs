use std::fmt;

use super::{Config, Instruction};

impl<C: Config> fmt::Display for Instruction<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instruction::LinComb(lc) => write!(f, "{lc}"),
            Instruction::Mul(inputs) => {
                for (i, input) in inputs.iter().enumerate() {
                    write!(f, "v{input}")?;
                    if i < inputs.len() - 1 {
                        write!(f, "*")?;
                    }
                }
                Ok(())
            }
            Instruction::ConstantLike(coef) => write!(f, "{coef}"),
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                ..
            } => {
                write!(f, "sub{sub_circuit_id}(")?;
                for (i, input) in inputs.iter().enumerate() {
                    write!(f, "v{input}")?;
                    if i < inputs.len() - 1 {
                        write!(f, ",")?;
                    }
                }
                write!(f, ")")
            }
            Instruction::CustomGate {
                gate_type, inputs, ..
            } => {
                write!(f, "custom{gate_type}(")?;
                for (i, input) in inputs.iter().enumerate() {
                    write!(f, "v{input}")?;
                    if i < inputs.len() - 1 {
                        write!(f, ",")?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}
