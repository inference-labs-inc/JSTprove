use std::hash::{DefaultHasher, Hash, Hasher};

use ethnum::U256;
use rand::RngCore;

use crate::{field::Field, utils::error::Error};

#[repr(u64)]
pub enum BuiltinHintIds {
    Identity = 0xccc000000000,
    Div,
    Eq,
    NotEq,
    BoolOr,
    BoolAnd,
    BitOr,
    BitAnd,
    BitXor,
    Select,
    Pow,
    IntDiv,
    Mod,
    ShiftL,
    ShiftR,
    LesserEq,
    GreaterEq,
    Lesser,
    Greater,
    ToBinary,
}

#[cfg(not(target_pointer_width = "64"))]
compile_error!("compilation is only allowed for 64-bit targets");

impl BuiltinHintIds {
    pub fn from_usize(id: usize) -> Option<BuiltinHintIds> {
        if id < (BuiltinHintIds::Identity as u64 as usize) {
            return None;
        }
        if id > (BuiltinHintIds::Identity as u64 as usize + 100) {
            return None;
        }
        match id {
            x if x == BuiltinHintIds::Identity as u64 as usize => Some(BuiltinHintIds::Identity),
            x if x == BuiltinHintIds::Div as u64 as usize => Some(BuiltinHintIds::Div),
            x if x == BuiltinHintIds::Eq as u64 as usize => Some(BuiltinHintIds::Eq),
            x if x == BuiltinHintIds::NotEq as u64 as usize => Some(BuiltinHintIds::NotEq),
            x if x == BuiltinHintIds::BoolOr as u64 as usize => Some(BuiltinHintIds::BoolOr),
            x if x == BuiltinHintIds::BoolAnd as u64 as usize => Some(BuiltinHintIds::BoolAnd),
            x if x == BuiltinHintIds::BitOr as u64 as usize => Some(BuiltinHintIds::BitOr),
            x if x == BuiltinHintIds::BitAnd as u64 as usize => Some(BuiltinHintIds::BitAnd),
            x if x == BuiltinHintIds::BitXor as u64 as usize => Some(BuiltinHintIds::BitXor),
            x if x == BuiltinHintIds::Select as u64 as usize => Some(BuiltinHintIds::Select),
            x if x == BuiltinHintIds::Pow as u64 as usize => Some(BuiltinHintIds::Pow),
            x if x == BuiltinHintIds::IntDiv as u64 as usize => Some(BuiltinHintIds::IntDiv),
            x if x == BuiltinHintIds::Mod as u64 as usize => Some(BuiltinHintIds::Mod),
            x if x == BuiltinHintIds::ShiftL as u64 as usize => Some(BuiltinHintIds::ShiftL),
            x if x == BuiltinHintIds::ShiftR as u64 as usize => Some(BuiltinHintIds::ShiftR),
            x if x == BuiltinHintIds::LesserEq as u64 as usize => Some(BuiltinHintIds::LesserEq),
            x if x == BuiltinHintIds::GreaterEq as u64 as usize => Some(BuiltinHintIds::GreaterEq),
            x if x == BuiltinHintIds::Lesser as u64 as usize => Some(BuiltinHintIds::Lesser),
            x if x == BuiltinHintIds::Greater as u64 as usize => Some(BuiltinHintIds::Greater),
            x if x == BuiltinHintIds::ToBinary as u64 as usize => Some(BuiltinHintIds::ToBinary),
            _ => None,
        }
    }
}

fn stub_impl_general<F: Field>(hint_id: usize, inputs: &Vec<F>, num_outputs: usize) -> Vec<F> {
    let mut hasher = DefaultHasher::new();
    hint_id.hash(&mut hasher);
    inputs.hash(&mut hasher);
    let mut outputs = Vec::with_capacity(num_outputs);
    for _ in 0..num_outputs {
        let t = hasher.finish();
        outputs.push(F::from(t as u32));
        t.hash(&mut hasher);
    }
    outputs
}

fn validate_builtin_hint(
    hint_id: BuiltinHintIds,
    num_inputs: usize,
    num_outputs: usize,
) -> Result<(), Error> {
    match hint_id {
        BuiltinHintIds::Identity => {
            if num_inputs != num_outputs {
                return Err(Error::InternalError(
                    "identity hint requires exactly the same number of inputs and outputs"
                        .to_string(),
                ));
            }
            if num_inputs == 0 {
                return Err(Error::InternalError(
                    "identity hint requires at least 1 input".to_string(),
                ));
            }
        }
        BuiltinHintIds::Div
        | BuiltinHintIds::Eq
        | BuiltinHintIds::NotEq
        | BuiltinHintIds::BoolOr
        | BuiltinHintIds::BoolAnd
        | BuiltinHintIds::BitOr
        | BuiltinHintIds::BitAnd
        | BuiltinHintIds::BitXor
        | BuiltinHintIds::Pow
        | BuiltinHintIds::IntDiv
        | BuiltinHintIds::Mod
        | BuiltinHintIds::ShiftL
        | BuiltinHintIds::ShiftR
        | BuiltinHintIds::LesserEq
        | BuiltinHintIds::GreaterEq
        | BuiltinHintIds::Lesser
        | BuiltinHintIds::Greater => {
            if num_inputs != 2 {
                return Err(Error::InternalError(
                    "binary op requires exactly 2 inputs".to_string(),
                ));
            }
            if num_outputs != 1 {
                return Err(Error::InternalError(
                    "binary op requires exactly 1 output".to_string(),
                ));
            }
        }
        BuiltinHintIds::Select => {
            if num_inputs != 3 {
                return Err(Error::InternalError(
                    "select requires exactly 3 inputs".to_string(),
                ));
            }
            if num_outputs != 1 {
                return Err(Error::InternalError(
                    "select requires exactly 1 output".to_string(),
                ));
            }
        }
        BuiltinHintIds::ToBinary => {
            if num_inputs != 1 {
                return Err(Error::InternalError(
                    "to_binary requires exactly 1 input".to_string(),
                ));
            }
            if num_outputs == 0 {
                return Err(Error::InternalError(
                    "to_binary requires at least 1 output".to_string(),
                ));
            }
        }
    }
    Ok(())
}

pub fn validate_hint(hint_id: usize, num_inputs: usize, num_outputs: usize) -> Result<(), Error> {
    match BuiltinHintIds::from_usize(hint_id) {
        Some(hint_id) => validate_builtin_hint(hint_id, num_inputs, num_outputs),
        None => {
            if num_outputs == 0 {
                return Err(Error::InternalError(
                    "custom hint requires at least 1 output".to_string(),
                ));
            }
            if num_inputs == 0 {
                return Err(Error::InternalError(
                    "custom hint requires at least 1 input".to_string(),
                ));
            }
            Ok(())
        }
    }
}

pub fn impl_builtin_hint<F: Field>(
    hint_id: BuiltinHintIds,
    inputs: &[F],
    num_outputs: usize,
) -> Vec<F> {
    match hint_id {
        BuiltinHintIds::Identity => inputs.iter().take(num_outputs).cloned().collect(),
        BuiltinHintIds::Div => binop_hint(inputs, |x, y| match y.inv() {
            Some(inv) => x * inv,
            None => F::zero(),
        }),
        BuiltinHintIds::Eq => binop_hint(inputs, |x, y| F::from((x == y) as u32)),
        BuiltinHintIds::NotEq => binop_hint(inputs, |x, y| F::from((x != y) as u32)),
        BuiltinHintIds::BoolOr => binop_hint(inputs, |x, y| {
            F::from((!x.is_zero() || !y.is_zero()) as u32)
        }),
        BuiltinHintIds::BoolAnd => binop_hint(inputs, |x, y| {
            F::from((!x.is_zero() && !y.is_zero()) as u32)
        }),
        BuiltinHintIds::BitOr => binop_hint_on_u256(inputs, |x, y| x | y),
        BuiltinHintIds::BitAnd => binop_hint_on_u256(inputs, |x, y| x & y),
        BuiltinHintIds::BitXor => binop_hint_on_u256(inputs, |x, y| x ^ y),
        BuiltinHintIds::Select => {
            let mut outputs = Vec::with_capacity(num_outputs);
            outputs.push(if !inputs[0].is_zero() {
                inputs[1]
            } else {
                inputs[2]
            });
            outputs
        }
        BuiltinHintIds::Pow => binop_hint(inputs, |x, y| {
            let mut t = x;
            let mut res = F::one();
            let mut y: U256 = y.to_u256();
            while y != U256::ZERO {
                if y & U256::from(1u32) != U256::ZERO {
                    res *= t;
                }
                y >>= 1;
                t = t * t;
            }
            res
        }),
        BuiltinHintIds::IntDiv => {
            binop_hint_on_u256(
                inputs,
                |x, y| if y == U256::ZERO { U256::ZERO } else { x / y },
            )
        }
        BuiltinHintIds::Mod => {
            binop_hint_on_u256(
                inputs,
                |x, y| if y == U256::ZERO { U256::ZERO } else { x % y },
            )
        }
        BuiltinHintIds::ShiftL => binop_hint_on_u256(inputs, |x, y| circom_shift_l_impl::<F>(x, y)),
        BuiltinHintIds::ShiftR => binop_hint_on_u256(inputs, |x, y| circom_shift_r_impl::<F>(x, y)),
        BuiltinHintIds::LesserEq => binop_hint(inputs, |x, y| F::from((x <= y) as u32)),
        BuiltinHintIds::GreaterEq => binop_hint(inputs, |x, y| F::from((x >= y) as u32)),
        BuiltinHintIds::Lesser => binop_hint(inputs, |x, y| F::from((x < y) as u32)),
        BuiltinHintIds::Greater => binop_hint(inputs, |x, y| F::from((x > y) as u32)),
        BuiltinHintIds::ToBinary => to_binary(inputs[0], num_outputs).unwrap(), // TODO: error propagation
    }
}

fn binop_hint<F: Field, G: Fn(F, F) -> F>(inputs: &[F], f: G) -> Vec<F> {
    vec![f(inputs[0], inputs[1])]
}

fn binop_hint_on_u256<F: Field, G: Fn(U256, U256) -> U256>(inputs: &[F], f: G) -> Vec<F> {
    let x_u256: U256 = inputs[0].to_u256();
    let y_u256: U256 = inputs[1].to_u256();
    let z_u256 = f(x_u256, y_u256);
    vec![F::from_u256(z_u256)]
}

pub fn to_binary<F: Field>(x: F, num_outputs: usize) -> Result<Vec<F>, Error> {
    let mut outputs = Vec::with_capacity(num_outputs);
    let mut y = x.to_u256();
    for _ in 0..num_outputs {
        outputs.push(F::from_u256(y & U256::from(1u32)));
        y >>= 1;
    }
    if y != U256::ZERO {
        return Err(Error::UserError(
            "to_binary hint input too large".to_string(),
        ));
    }
    Ok(outputs)
}

pub fn stub_impl<F: Field>(hint_id: usize, inputs: &Vec<F>, num_outputs: usize) -> Vec<F> {
    match BuiltinHintIds::from_usize(hint_id) {
        Some(hint_id) => impl_builtin_hint(hint_id, inputs, num_outputs),
        None => stub_impl_general(hint_id, inputs, num_outputs),
    }
}

pub fn random_builtin(mut rand: impl RngCore) -> (usize, usize, usize) {
    loop {
        let hint_id = (rand.next_u64() as usize % 100) + (BuiltinHintIds::Identity as u64 as usize);
        if let Some(hint_id) = BuiltinHintIds::from_usize(hint_id) {
            match hint_id {
                BuiltinHintIds::Identity => {
                    let num_inputs = (rand.next_u64() % 10) as usize + 1;
                    let num_outputs = num_inputs;
                    return (hint_id as usize, num_inputs, num_outputs);
                }
                BuiltinHintIds::Div
                | BuiltinHintIds::Eq
                | BuiltinHintIds::NotEq
                | BuiltinHintIds::BoolOr
                | BuiltinHintIds::BoolAnd
                | BuiltinHintIds::BitOr
                | BuiltinHintIds::BitAnd
                | BuiltinHintIds::BitXor
                | BuiltinHintIds::Pow
                | BuiltinHintIds::IntDiv
                | BuiltinHintIds::Mod
                | BuiltinHintIds::ShiftL
                | BuiltinHintIds::ShiftR
                | BuiltinHintIds::LesserEq
                | BuiltinHintIds::GreaterEq
                | BuiltinHintIds::Lesser
                | BuiltinHintIds::Greater => {
                    return (hint_id as usize, 2, 1);
                }
                BuiltinHintIds::Select => {
                    return (hint_id as usize, 3, 1);
                }
                BuiltinHintIds::ToBinary => {
                    return (hint_id as usize, 1, 300);
                }
            }
        }
    }
}

pub fn u256_bit_length(x: U256) -> usize {
    256 - x.leading_zeros() as usize
}

pub fn circom_shift_l_impl<F: Field>(x: U256, k: U256) -> U256 {
    let top = F::MODULUS / 2;
    if k <= top {
        let shift = if (k >> U256::from(64u32)) == U256::ZERO {
            k.as_u64() as usize
        } else {
            u256_bit_length(F::MODULUS)
        };
        if shift >= 256 {
            return U256::ZERO;
        }
        let value = x << shift;
        let mask = U256::from(1u32) << u256_bit_length(F::MODULUS);
        let mask = mask - 1;
        value & mask
    } else {
        circom_shift_r_impl::<F>(x, F::MODULUS - k)
    }
}

pub fn circom_shift_r_impl<F: Field>(x: U256, k: U256) -> U256 {
    let top = F::MODULUS / 2;
    if k <= top {
        let shift = if (k >> U256::from(64u32)) == U256::ZERO {
            k.as_u64() as usize
        } else {
            u256_bit_length(F::MODULUS)
        };
        if shift >= 256 {
            return U256::ZERO;
        }
        x >> shift
    } else {
        circom_shift_l_impl::<F>(x, F::MODULUS - k)
    }
}
