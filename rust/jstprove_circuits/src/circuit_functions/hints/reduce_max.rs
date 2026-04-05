use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const REDUCE_MAX_HINT_KEY: &str = "jstprove.reduce_max_hint";

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
/// # Errors
/// Returns `Error::UserError` on invalid inputs.
#[allow(clippy::missing_errors_doc)]
pub fn reduce_max_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() < 2 {
        return Err(Error::UserError(format!(
            "reduce_max_hint: expected at least 2 inputs (values + scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "reduce_max_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let n = inputs.len() - 1;
    let mut max_val = field_to_i64(inputs[0]);
    for item in inputs.iter().take(n).skip(1) {
        let v = field_to_i64(*item);
        if v > max_val {
            max_val = v;
        }
    }

    outputs[0] = if max_val >= 0 {
        F::from_u256(U256::from(max_val as u64))
    } else {
        let mag = U256::from(max_val.unsigned_abs());
        F::from_u256(F::MODULUS - mag)
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254Fr;

    type F = BN254Fr;

    fn field(n: i64) -> F {
        if n >= 0 {
            F::from_u256(U256::from(n as u64))
        } else {
            let mag = U256::from(n.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    }

    fn to_i64_test(f: F) -> i64 {
        let p_half = F::MODULUS / 2;
        let xu = f.to_u256();
        if xu > p_half {
            -((F::MODULUS - xu).as_u64() as i64)
        } else {
            xu.as_u64() as i64
        }
    }

    fn run_hint(values: &[i64], scale: u64) -> i64 {
        let mut inputs: Vec<F> = values.iter().map(|&v| field(v)).collect();
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = [F::zero()];
        reduce_max_hint::<F>(&inputs, &mut outputs).unwrap();
        to_i64_test(outputs[0])
    }

    #[test]
    fn reduce_max_positive() {
        assert_eq!(run_hint(&[100, 300, 200], 1 << 18), 300);
    }

    #[test]
    fn reduce_max_negative() {
        assert_eq!(run_hint(&[-300, -100, -200], 1 << 18), -100);
    }

    #[test]
    fn reduce_max_single() {
        assert_eq!(run_hint(&[42], 1 << 18), 42);
    }

    #[test]
    fn reduce_max_too_few_inputs() {
        let mut out = [F::zero()];
        assert!(reduce_max_hint::<F>(&[F::zero()], &mut out).is_err());
        assert!(reduce_max_hint::<F>(&[], &mut out).is_err());
    }
}
