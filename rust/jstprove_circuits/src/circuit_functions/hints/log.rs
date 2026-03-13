// Hint function for the ONNX `Log` operator.
//
// # ZK design
// `Log` (natural logarithm) is a transcendental function and cannot be
// expressed as a polynomial over a finite field. This hint performs the
// computation outside the circuit (native f64) and injects the result as
// an unconstrained witness.
//
// # Soundness limitation
// The hint alone adds no constraint. The `LogLayer` circuit does NOT apply
// a LogUp range check because Log outputs can be negative (for inputs < 1
// in real-value terms). The output is unconstrained beyond being a valid
// field element.
//
// Full soundness would require a lookup table — a planned future extension.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const LOG_HINT_KEY: &str = "jstprove.log_hint";

/// Hint function for elementwise natural logarithm over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` as a field element (signed encoding).
/// - `inputs[1]`: scaling factor `scale = 2^scale_exponent` as a field element.
///
/// # Outputs
/// - `outputs[0]`: `round(log(x_q / scale) * scale)`, encoded as a signed
///   field element (two's complement mod p for negative values).
///
/// # Errors
/// Returns [`Error::UserError`] for wrong input/output counts or zero scale.
/// If `x_q <= 0` (i.e. the real value is non-positive), the output is clamped
/// to `i64::MIN` to indicate an invalid domain.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn log_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "log_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "log_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);
    let scale_u256 = inputs[1].to_u256();
    let scale_u64: u64 = scale_u256.as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "log_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    let x_real = x_i64 as f64 / scale_f64;
    let y_q: i64 = if x_real > 0.0 {
        let y_real = x_real.ln();
        let y_scaled = y_real * scale_f64;
        // Clamp to i64 range (signed output).
        if y_scaled >= i64::MAX as f64 {
            i64::MAX
        } else if y_scaled <= i64::MIN as f64 {
            i64::MIN
        } else {
            y_scaled.round() as i64
        }
    } else {
        // log is undefined for non-positive inputs; return a large negative.
        i64::MIN
    };

    // Encode as field element: positive → y_q, negative → p − |y_q|.
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

    fn run_hint(x_q: i64, scale: u64) -> i64 {
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        log_hint::<F>(&inputs, &mut outputs).unwrap();
        super::field_to_i64(outputs[0])
    }

    #[test]
    fn log_hint_one_maps_to_zero() {
        // log(1.0) = 0.0; x_q = scale, y_q = 0
        let scale: u64 = 1 << 18;
        let result = run_hint(scale as i64, scale);
        assert_eq!(result, 0, "log(1.0) should be 0");
    }

    #[test]
    fn log_hint_e_maps_to_one_scale() {
        // log(e) = 1.0; x_q = round(e * scale), y_q ≈ scale
        let scale: u64 = 1 << 8; // small scale for precision
        let x_q = (std::f64::consts::E * scale as f64).round() as i64;
        let result = run_hint(x_q, scale);
        let expected = scale as i64;
        assert!(
            (result - expected).abs() <= 2,
            "log(e) got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn log_hint_negative_output_for_small_input() {
        // log(0.5) = -ln(2) ≈ -0.693; x_q = scale/2, y_q < 0
        let scale: u64 = 1 << 10;
        let x_q = scale as i64 / 2; // represents 0.5
        let result = run_hint(x_q, scale);
        assert!(result < 0, "log(0.5) should be negative, got {result}");
    }

    #[test]
    fn log_hint_nonpositive_input() {
        // Non-positive real value → i64::MIN
        let scale: u64 = 1 << 8;
        let result = run_hint(0, scale);
        assert_eq!(result, i64::MIN);
        let result = run_hint(-(scale as i64), scale);
        assert_eq!(result, i64::MIN);
    }

    #[test]
    fn log_hint_wrong_input_count() {
        let mut out = [F::zero()];
        assert!(log_hint::<F>(&[], &mut out).is_err());
        assert!(log_hint::<F>(&[F::zero(); 3], &mut out).is_err());
    }

    #[test]
    fn log_hint_wrong_output_count() {
        let inputs = [F::zero(), F::from_u256(U256::from(256u64))];
        assert!(log_hint::<F>(&inputs, &mut []).is_err());
    }

    #[test]
    fn log_hint_zero_scale_returns_error() {
        let inputs = [F::zero(), F::zero()];
        let mut out = [F::zero()];
        assert!(log_hint::<F>(&inputs, &mut out).is_err());
    }
}
