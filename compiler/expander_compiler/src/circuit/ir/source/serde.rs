use std::io::{Error as IoError, Read, Write};

use serdes::{ExpSerde, SerdeResult};

use crate::circuit::{config::Config, ir::expr::LinComb, layered::Coef};

use super::{BoolBinOpType, Constraint, ConstraintType, Instruction, UnconstrainedBinOpType};

impl<C: Config> ExpSerde for Instruction<C> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        match self {
            Instruction::LinComb(lin_comb) => {
                1u8.serialize_into(&mut writer)?;
                lin_comb.serialize_into(&mut writer)?;
            }
            Instruction::Mul(inputs) => {
                2u8.serialize_into(&mut writer)?;
                inputs.serialize_into(&mut writer)?;
            }
            Instruction::Div { x, y, checked } => {
                3u8.serialize_into(&mut writer)?;
                x.serialize_into(&mut writer)?;
                y.serialize_into(&mut writer)?;
                checked.serialize_into(&mut writer)?;
            }
            Instruction::BoolBinOp { x, y, op } => {
                4u8.serialize_into(&mut writer)?;
                x.serialize_into(&mut writer)?;
                y.serialize_into(&mut writer)?;
                (op.clone() as u8).serialize_into(&mut writer)?;
            }
            Instruction::IsZero(x) => {
                5u8.serialize_into(&mut writer)?;
                x.serialize_into(&mut writer)?;
            }
            Instruction::Commit(inputs) => {
                6u8.serialize_into(&mut writer)?;
                inputs.serialize_into(&mut writer)?;
            }
            Instruction::Hint {
                hint_id,
                inputs,
                num_outputs,
            } => {
                7u8.serialize_into(&mut writer)?;
                hint_id.serialize_into(&mut writer)?;
                inputs.serialize_into(&mut writer)?;
                num_outputs.serialize_into(&mut writer)?;
            }
            Instruction::ConstantLike(coef) => {
                8u8.serialize_into(&mut writer)?;
                coef.serialize_into(&mut writer)?;
            }
            Instruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => {
                9u8.serialize_into(&mut writer)?;
                sub_circuit_id.serialize_into(&mut writer)?;
                inputs.serialize_into(&mut writer)?;
                num_outputs.serialize_into(&mut writer)?;
            }
            Instruction::UnconstrainedBinOp { x, y, op } => {
                10u8.serialize_into(&mut writer)?;
                x.serialize_into(&mut writer)?;
                y.serialize_into(&mut writer)?;
                (op.clone() as u8).serialize_into(&mut writer)?;
            }
            Instruction::UnconstrainedSelect {
                cond,
                if_true,
                if_false,
            } => {
                11u8.serialize_into(&mut writer)?;
                cond.serialize_into(&mut writer)?;
                if_true.serialize_into(&mut writer)?;
                if_false.serialize_into(&mut writer)?;
            }
            Instruction::CustomGate { gate_type, inputs } => {
                12u8.serialize_into(&mut writer)?;
                gate_type.serialize_into(&mut writer)?;
                inputs.serialize_into(&mut writer)?;
            }
            Instruction::ToBinary { x, num_bits } => {
                13u8.serialize_into(&mut writer)?;
                x.serialize_into(&mut writer)?;
                num_bits.serialize_into(&mut writer)?;
            }
        };
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let instruction_type = u8::deserialize_from(&mut reader)?;
        Ok(match instruction_type {
            1 => Instruction::LinComb(LinComb::deserialize_from(&mut reader)?),
            2 => Instruction::Mul(Vec::<usize>::deserialize_from(&mut reader)?),
            3 => Instruction::Div {
                x: usize::deserialize_from(&mut reader)?,
                y: usize::deserialize_from(&mut reader)?,
                checked: bool::deserialize_from(&mut reader)?,
            },
            4 => Instruction::BoolBinOp {
                x: usize::deserialize_from(&mut reader)?,
                y: usize::deserialize_from(&mut reader)?,
                op: match u8::deserialize_from(&mut reader)? {
                    1 => BoolBinOpType::Xor,
                    2 => BoolBinOpType::Or,
                    3 => BoolBinOpType::And,
                    _ => {
                        return Err(IoError::new(
                            std::io::ErrorKind::InvalidData,
                            "invalid BoolBinOpType",
                        ))?
                    }
                },
            },
            5 => Instruction::IsZero(usize::deserialize_from(&mut reader)?),
            6 => Instruction::Commit(Vec::<usize>::deserialize_from(&mut reader)?),
            7 => Instruction::Hint {
                hint_id: usize::deserialize_from(&mut reader)?,
                inputs: Vec::<usize>::deserialize_from(&mut reader)?,
                num_outputs: usize::deserialize_from(&mut reader)?,
            },
            8 => {
                let coef = Coef::<C>::deserialize_from(&mut reader)?;
                Instruction::ConstantLike(coef)
            }
            9 => Instruction::SubCircuitCall {
                sub_circuit_id: usize::deserialize_from(&mut reader)?,
                inputs: Vec::<usize>::deserialize_from(&mut reader)?,
                num_outputs: usize::deserialize_from(&mut reader)?,
            },
            10 => Instruction::UnconstrainedBinOp {
                x: usize::deserialize_from(&mut reader)?,
                y: usize::deserialize_from(&mut reader)?,
                op: match u8::deserialize_from(&mut reader)? {
                    1 => UnconstrainedBinOpType::Div,
                    2 => UnconstrainedBinOpType::Pow,
                    3 => UnconstrainedBinOpType::IntDiv,
                    4 => UnconstrainedBinOpType::Mod,
                    5 => UnconstrainedBinOpType::ShiftL,
                    6 => UnconstrainedBinOpType::ShiftR,
                    7 => UnconstrainedBinOpType::LesserEq,
                    8 => UnconstrainedBinOpType::GreaterEq,
                    9 => UnconstrainedBinOpType::Lesser,
                    10 => UnconstrainedBinOpType::Greater,
                    11 => UnconstrainedBinOpType::Eq,
                    12 => UnconstrainedBinOpType::NotEq,
                    13 => UnconstrainedBinOpType::BoolOr,
                    14 => UnconstrainedBinOpType::BoolAnd,
                    15 => UnconstrainedBinOpType::BitOr,
                    16 => UnconstrainedBinOpType::BitAnd,
                    17 => UnconstrainedBinOpType::BitXor,
                    _ => {
                        return Err(IoError::new(
                            std::io::ErrorKind::InvalidData,
                            "invalid UnconstrainedBinOpType",
                        ))?
                    }
                },
            },
            11 => Instruction::UnconstrainedSelect {
                cond: usize::deserialize_from(&mut reader)?,
                if_true: usize::deserialize_from(&mut reader)?,
                if_false: usize::deserialize_from(&mut reader)?,
            },
            12 => Instruction::CustomGate {
                gate_type: usize::deserialize_from(&mut reader)?,
                inputs: Vec::<usize>::deserialize_from(&mut reader)?,
            },
            13 => Instruction::ToBinary {
                x: usize::deserialize_from(&mut reader)?,
                num_bits: usize::deserialize_from(&mut reader)?,
            },
            _ => {
                return Err(IoError::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid InstructionType",
                ))?
            }
        })
    }
}

impl ExpSerde for Constraint {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        (self.typ as u8).serialize_into(&mut writer)?;
        self.var.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        Ok(Constraint {
            typ: match u8::deserialize_from(&mut reader)? {
                1 => ConstraintType::Zero,
                2 => ConstraintType::NonZero,
                3 => ConstraintType::Bool,
                _ => {
                    return Err(IoError::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid ConstraintType",
                    ))?
                }
            },
            var: usize::deserialize_from(&mut reader)?,
        })
    }
}
