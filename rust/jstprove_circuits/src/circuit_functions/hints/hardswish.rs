use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const HARDSWISH_HINT_KEY: &str = "jstprove.hardswish_hint";

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::similar_names
)]
pub fn hardswish_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "hardswish_hint: expected 2 inputs (x, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "hardswish_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);
    let scale_u64 = inputs[1].to_u256().as_u64();

    if scale_u64 == 0 {
        return Err(Error::UserError(
            "hardswish_hint: scale is zero".to_string(),
        ));
    }

    let x_f64 = x_i64 as f64;
    let alpha = scale_u64 as f64;

    let x_real = x_f64 / alpha;
    let inner = (x_real + 3.0).clamp(0.0, 6.0);
    let y_real = x_real * inner / 6.0;
    let y_q = (y_real * alpha).round() as i64;

    outputs[0] = if y_q >= 0 {
        F::from_u256(U256::from(y_q as u64))
    } else {
        let mag = U256::from(y_q.unsigned_abs());
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

    #[test]
    fn hardswish_positive_large() {
        let scale: u64 = 1 << 10;
        let x_q = 5000_i64;
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        hardswish_hint::<F>(&inputs, &mut outputs).unwrap();
        let result = super::field_to_i64(outputs[0]);
        assert!(result > 0);
    }

    #[test]
    fn hardswish_zero() {
        let scale: u64 = 1 << 10;
        let inputs = [field(0), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        hardswish_hint::<F>(&inputs, &mut outputs).unwrap();
        let result = super::field_to_i64(outputs[0]);
        assert_eq!(result, 0);
    }

    #[test]
    fn hardswish_negative_large() {
        let scale: u64 = 1 << 10;
        let x_q = -4000_i64;
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        hardswish_hint::<F>(&inputs, &mut outputs).unwrap();
        let result = super::field_to_i64(outputs[0]);
        assert_eq!(result, 0);
    }
}
