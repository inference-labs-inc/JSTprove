use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const LEAKY_RELU_HINT_KEY: &str = "jstprove.leaky_relu_hint";

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc
)]
pub fn leaky_relu_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 3 {
        return Err(Error::UserError(format!(
            "leaky_relu_hint: expected 3 inputs (x, alpha_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "leaky_relu_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);
    let alpha_q_i64 = field_to_i64(inputs[1]);
    let scale_u64 = inputs[2].to_u256().as_u64();

    if scale_u64 == 0 {
        return Err(Error::UserError(
            "leaky_relu_hint: scale is zero".to_string(),
        ));
    }

    let y_i64 = if x_i64 >= 0 {
        x_i64
    } else {
        let product = i128::from(x_i64) * i128::from(alpha_q_i64);
        let divided = product / i128::from(scale_u64);
        divided as i64
    };

    outputs[0] = if y_i64 >= 0 {
        F::from_u256(U256::from(y_i64 as u64))
    } else {
        let mag = U256::from(y_i64.unsigned_abs());
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
    fn leaky_relu_positive_passthrough() {
        let scale: u64 = 1 << 10;
        let alpha_q = (0.01_f64 * scale as f64).round() as i64;
        let x_q: i64 = 500;
        let inputs = [field(x_q), field(alpha_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        leaky_relu_hint::<F>(&inputs, &mut outputs).unwrap();
        assert_eq!(field_to_i64(outputs[0]), 500);
    }

    #[test]
    fn leaky_relu_negative_scaled() {
        let scale: u64 = 1 << 10;
        let alpha_q = (0.01_f64 * scale as f64).round() as i64;
        let x_q: i64 = -1000;
        let inputs = [field(x_q), field(alpha_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        leaky_relu_hint::<F>(&inputs, &mut outputs).unwrap();
        let result = field_to_i64(outputs[0]);
        assert!(result < 0);
        let expected = (-1000_i128 * alpha_q as i128 / scale as i128) as i64;
        assert_eq!(result, expected);
    }
}
