// Hint function for the ONNX `Pow` operator.
//
// # ZK design
// `Pow` computes x^y in fixed-point. This hint performs the computation
// **outside** the circuit (native f64) and injects the result as an
// unconstrained witness.
//
// # Soundness limitation
// The output is NOT constrained (can be negative for odd-exponent negative base).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const POW_HINT_KEY: &str = "jstprove.pow_hint";

/// Hint function for elementwise `Pow` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised base `x_q` (signed fixed-point).
/// - `inputs[1]`: quantised exponent `y_q` (signed fixed-point).
/// - `inputs[2]`: the scaling factor `scale = 2^scale_exponent`.
///
/// # Outputs
/// - `outputs[0]`: `round(pow(x_q/scale, y_q/scale) * scale)`, signed two's complement encoding.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 3` or `outputs.len() != 1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names
)]
pub fn pow_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 3 {
        return Err(Error::UserError(format!(
            "pow_hint: expected 3 inputs (x_q, exp_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "pow_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);
    let exp_i64 = field_to_i64(inputs[1]);

    let scale_u64 = inputs[2].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "pow_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    let x_real = x_i64 as f64 / scale_f64;
    let exp_real = exp_i64 as f64 / scale_f64;

    let y_real = x_real.powf(exp_real);

    let y_scaled = y_real * scale_f64;
    let y_q: i64 = if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled <= i64::MIN as f64 {
        i64::MIN
    } else if y_scaled.is_nan() || y_scaled.is_infinite() {
        return Err(Error::UserError(format!(
            "pow_hint: invalid result for base={x_i64} exponent={exp_i64} (NaN or Inf)"
        )));
    } else {
        y_scaled.round() as i64
    };

    // Encode signed result.
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

    fn to_i64(f: F) -> i64 {
        let p_half = F::MODULUS / 2;
        let u = f.to_u256();
        if u > p_half {
            -((F::MODULUS - u).as_u64() as i64)
        } else {
            u.as_u64() as i64
        }
    }

    fn run_hint(x_q: i64, exp_q: i64, scale: u64) -> i64 {
        let inputs = [field(x_q), field(exp_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        pow_hint::<F>(&inputs, &mut outputs).unwrap();
        to_i64(outputs[0])
    }

    #[test]
    fn pow_two_squared() {
        // 2.0^2.0 = 4.0
        let scale: u64 = 1 << 18;
        let x_q = 2 * scale as i64;
        let exp_q = 2 * scale as i64;
        let result = run_hint(x_q, exp_q, scale);
        let expected = 4 * scale as i64;
        assert!(
            (result - expected).abs() <= 2,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn pow_zero_exponent() {
        // x^0 = 1.0
        let scale: u64 = 1 << 18;
        let x_q = 3 * scale as i64;
        let exp_q = 0;
        let result = run_hint(x_q, exp_q, scale);
        assert!((result - scale as i64).abs() <= 1, "got {result}");
    }

    #[test]
    fn pow_negative_base_integer_exponent_even() {
        // (-2)^2 = 4
        let scale: u64 = 1 << 18;
        let x_q = -2 * scale as i64;
        let exp_q = 2 * scale as i64;
        let result = run_hint(x_q, exp_q, scale);
        let expected = 4 * scale as i64;
        assert!(
            (result - expected).abs() <= 2,
            "(-2)^2: got {result}, expected {expected}"
        );
    }

    #[test]
    fn pow_negative_base_integer_exponent_odd() {
        // (-2)^3 = -8
        let scale: u64 = 1 << 18;
        let x_q = -2 * scale as i64;
        let exp_q = 3 * scale as i64;
        let result = run_hint(x_q, exp_q, scale);
        let expected = -8 * scale as i64;
        assert!(
            (result - expected).abs() <= 2,
            "(-2)^3: got {result}, expected {expected}"
        );
    }
}
