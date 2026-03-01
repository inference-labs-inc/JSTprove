use ethnum::U256;

use crate::{
    circuit::{config::Config, layered::Coef},
    field::{Field, FieldArith},
    frontend::CircuitField,
    hints::{self, circom_shift_l_impl, circom_shift_r_impl, to_binary},
    utils::error::Error,
};

use super::{
    common::{self, EvalResult, IrConfig},
    expr,
};

#[cfg(test)]
mod tests;

pub mod chains;
pub mod serde;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Instruction<C: Config> {
    LinComb(expr::LinComb<C>),
    Mul(Vec<usize>),
    Div {
        x: usize,
        y: usize,
        checked: bool,
    },
    BoolBinOp {
        x: usize,
        y: usize,
        op: BoolBinOpType,
    },
    IsZero(usize),
    Commit(Vec<usize>),
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
    UnconstrainedBinOp {
        x: usize,
        y: usize,
        op: UnconstrainedBinOpType,
    },
    UnconstrainedSelect {
        cond: usize,
        if_true: usize,
        if_false: usize,
    },
    CustomGate {
        gate_type: usize,
        inputs: Vec<usize>,
    },
    ToBinary {
        x: usize,
        num_bits: usize,
    },
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum BoolBinOpType {
    Xor = 1,
    Or,
    And,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum UnconstrainedBinOpType {
    Div = 1,
    Pow,
    IntDiv,
    Mod,
    ShiftL,
    ShiftR,
    LesserEq,
    GreaterEq,
    Lesser,
    Greater,
    Eq,
    NotEq,
    BoolOr,
    BoolAnd,
    BitOr,
    BitAnd,
    BitXor,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Constraint {
    pub typ: ConstraintType,
    pub var: usize,
}

#[derive(Debug, Clone, Hash, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    Zero = 1,
    NonZero,
    Bool,
}

impl<C: Config> common::Constraint<C> for Constraint {
    type Type = ConstraintType;
    fn var(&self) -> usize {
        self.var
    }
    fn typ(&self) -> ConstraintType {
        self.typ
    }
    fn replace_var<F: Fn(usize) -> usize>(&self, f: F) -> Self {
        Constraint {
            typ: self.typ,
            var: f(self.var),
        }
    }
    fn new(var: usize, typ: ConstraintType) -> Self {
        Constraint { var, typ }
    }
}

impl<C: Config> common::ConstraintType<C> for ConstraintType {
    fn verify(&self, value: &CircuitField<C>) -> bool {
        match self {
            ConstraintType::Zero => value.is_zero(),
            ConstraintType::NonZero => !value.is_zero(),
            ConstraintType::Bool => value.is_zero() || *value == CircuitField::<C>::one(),
        }
    }
}

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct Irc<C: Config> {
    _a: C,
}
impl<C: Config> IrConfig for Irc<C> {
    type Instruction = Instruction<C>;
    type Constraint = Constraint;
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
            Instruction::Div { x, y, .. } => vec![*x, *y],
            Instruction::BoolBinOp { x, y, .. } => vec![*x, *y],
            Instruction::IsZero(x) => vec![*x],
            Instruction::Commit(inputs) => inputs.clone(),
            Instruction::Hint { inputs, .. } => inputs.clone(),
            Instruction::ConstantLike(_) => vec![],
            Instruction::SubCircuitCall { inputs, .. } => inputs.clone(),
            Instruction::UnconstrainedBinOp { x, y, .. } => vec![*x, *y],
            Instruction::UnconstrainedSelect {
                cond,
                if_true,
                if_false,
            } => vec![*cond, *if_true, *if_false],
            Instruction::CustomGate { inputs, .. } => inputs.clone(),
            Instruction::ToBinary { x, .. } => vec![*x],
        }
    }
    fn num_outputs(&self) -> usize {
        match self {
            Instruction::LinComb(_) => 1,
            Instruction::Mul(_) => 1,
            Instruction::Div { .. } => 1,
            Instruction::BoolBinOp { .. } => 1,
            Instruction::IsZero(_) => 1,
            Instruction::Commit(_) => 1,
            Instruction::Hint { num_outputs, .. } => *num_outputs,
            Instruction::ConstantLike(_) => 1,
            Instruction::SubCircuitCall { num_outputs, .. } => *num_outputs,
            Instruction::UnconstrainedBinOp { .. } => 1,
            Instruction::UnconstrainedSelect { .. } => 1,
            Instruction::CustomGate { .. } => 1,
            Instruction::ToBinary { num_bits, .. } => *num_bits,
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
            Instruction::Div { x, y, checked } => Instruction::Div {
                x: f(*x),
                y: f(*y),
                checked: *checked,
            },
            Instruction::BoolBinOp { x, y, op } => Instruction::BoolBinOp {
                x: f(*x),
                y: f(*y),
                op: op.clone(),
            },
            Instruction::IsZero(x) => Instruction::IsZero(f(*x)),
            Instruction::Commit(inputs) => {
                Instruction::Commit(inputs.iter().map(|i| f(*i)).collect())
            }
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
            Instruction::UnconstrainedBinOp { x, y, op } => Instruction::UnconstrainedBinOp {
                x: f(*x),
                y: f(*y),
                op: op.clone(),
            },
            Instruction::UnconstrainedSelect {
                cond,
                if_true,
                if_false,
            } => Instruction::UnconstrainedSelect {
                cond: f(*cond),
                if_true: f(*if_true),
                if_false: f(*if_false),
            },
            Instruction::CustomGate { gate_type, inputs } => Instruction::CustomGate {
                gate_type: *gate_type,
                inputs: inputs.iter().map(|i| f(*i)).collect(),
            },
            Instruction::ToBinary { x, num_bits } => Instruction::ToBinary {
                x: f(*x),
                num_bits: *num_bits,
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
            Instruction::ToBinary { num_bits, .. } => {
                if *num_bits > 0 {
                    Ok(())
                } else {
                    Err(Error::InternalError(
                        "to_binary instruction must have at least 1 bit".to_string(),
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
            Instruction::Div { x, y, checked } => {
                let x = values[*x];
                let y = values[*y];
                if y.is_zero() {
                    if x.is_zero() && !checked {
                        EvalResult::Value(CircuitField::<C>::zero())
                    } else {
                        EvalResult::Error(Error::UserError("division by zero".to_string()))
                    }
                } else {
                    EvalResult::Value(x * y.inv().unwrap())
                }
            }
            Instruction::BoolBinOp { x, y, op } => {
                let x = values[*x];
                let y = values[*y];
                if !x.is_zero() && x != CircuitField::<C>::one() {
                    return EvalResult::Error(Error::UserError("invalid bool value".to_string()));
                }
                if !y.is_zero() && y != CircuitField::<C>::one() {
                    return EvalResult::Error(Error::UserError("invalid bool value".to_string()));
                }
                match op {
                    BoolBinOpType::Xor => {
                        EvalResult::Value(x + y - CircuitField::<C>::from(2) * x * y)
                    }
                    BoolBinOpType::Or => EvalResult::Value(x + y - x * y),
                    BoolBinOpType::And => EvalResult::Value(x * y),
                }
            }
            Instruction::IsZero(x) => {
                EvalResult::Value(CircuitField::<C>::from(values[*x].is_zero() as u32))
            }
            Instruction::Commit(_) => {
                panic!("commit is not implemented")
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
            Instruction::UnconstrainedBinOp { x, y, op } => {
                let x = values[*x];
                let y = values[*y];
                match op.eval(&x, &y) {
                    Ok(res) => EvalResult::Value(res),
                    Err(e) => EvalResult::Error(e),
                }
            }
            Instruction::UnconstrainedSelect {
                cond,
                if_true,
                if_false,
            } => EvalResult::Value(if values[*cond].is_zero() {
                values[*if_false]
            } else {
                values[*if_true]
            }),
            Instruction::CustomGate { gate_type, inputs } => {
                let outputs =
                    hints::stub_impl(*gate_type, &inputs.iter().map(|i| values[*i]).collect(), 1);
                EvalResult::Values(outputs)
            }
            Instruction::ToBinary { x, num_bits } => match to_binary(values[*x], *num_bits) {
                Ok(outputs) => EvalResult::Values(outputs),
                Err(e) => EvalResult::Error(e),
            },
        }
    }
}

impl UnconstrainedBinOpType {
    pub fn eval<F: Field>(&self, x: &F, y: &F) -> Result<F, Error> {
        match self {
            UnconstrainedBinOpType::Div => {
                if y.is_zero() {
                    Err(Error::UserError("division by zero".to_string()))
                } else {
                    Ok(*x * y.inv().unwrap())
                }
            }
            UnconstrainedBinOpType::Pow => {
                let mut t = *x;
                let mut res = F::one();
                let mut y = y.to_u256();
                while y != U256::ZERO {
                    if (y & U256::from(1u64)) == U256::from(1u64) {
                        res *= t;
                    }
                    y >>= 1;
                    t = t * t;
                }
                Ok(res)
            }
            UnconstrainedBinOpType::IntDiv => binop_on_u256(x, y, |x, y| {
                if y == U256::ZERO {
                    Err(Error::UserError("division by zero".to_string()))
                } else {
                    Ok(x / y)
                }
            }),
            UnconstrainedBinOpType::Mod => binop_on_u256(x, y, |x, y| {
                if y == U256::ZERO {
                    Err(Error::UserError("division by zero".to_string()))
                } else {
                    Ok(x % y)
                }
            }),
            UnconstrainedBinOpType::ShiftL => {
                binop_on_u256(x, y, |x, y| Ok(circom_shift_l_impl::<F>(x, y)))
            }
            UnconstrainedBinOpType::ShiftR => {
                binop_on_u256(x, y, |x, y| Ok(circom_shift_r_impl::<F>(x, y)))
            }
            UnconstrainedBinOpType::LesserEq => {
                binop_on_u256(x, y, |x, y| Ok(U256::from((x <= y) as u32)))
            }
            UnconstrainedBinOpType::GreaterEq => {
                binop_on_u256(x, y, |x, y| Ok(U256::from((x >= y) as u32)))
            }
            UnconstrainedBinOpType::Lesser => {
                binop_on_u256(x, y, |x, y| Ok(U256::from((x < y) as u32)))
            }
            UnconstrainedBinOpType::Greater => {
                binop_on_u256(x, y, |x, y| Ok(U256::from((x > y) as u32)))
            }
            UnconstrainedBinOpType::Eq => Ok(F::from((x == y) as u32)),
            UnconstrainedBinOpType::NotEq => Ok(F::from((x != y) as u32)),
            UnconstrainedBinOpType::BoolOr => {
                let tx = !x.is_zero();
                let ty = !y.is_zero();
                Ok(F::from((tx || ty) as u32))
            }
            UnconstrainedBinOpType::BoolAnd => {
                let tx = !x.is_zero();
                let ty = !y.is_zero();
                Ok(F::from((tx && ty) as u32))
            }
            UnconstrainedBinOpType::BitOr => binop_on_u256(x, y, |x, y| Ok(x | y)),
            UnconstrainedBinOpType::BitAnd => binop_on_u256(x, y, |x, y| Ok(x & y)),
            UnconstrainedBinOpType::BitXor => binop_on_u256(x, y, |x, y| Ok(x ^ y)),
        }
    }
}

fn binop_on_u256<F: Field, G: Fn(U256, U256) -> Result<U256, Error>>(
    x: &F,
    y: &F,
    f: G,
) -> Result<F, Error> {
    let x: U256 = x.to_u256();
    let y: U256 = y.to_u256();
    match f(x, y) {
        Ok(res) => Ok(F::from_u256(res)),
        Err(e) => Err(e),
    }
}

pub type Circuit<C> = common::Circuit<Irc<C>>;
pub type RootCircuit<C> = common::RootCircuit<Irc<C>>;
