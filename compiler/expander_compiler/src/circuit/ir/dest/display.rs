use std::fmt;

use super::{Config, Instruction};

impl<C: Config> fmt::Display for Instruction<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instruction::InternalVariable { expr } => write!(f, "{expr}"),
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
            Instruction::ConstantLike { value } => write!(f, "{value}"),
        }
    }
}
