// Hint function for the ONNX `Sqrt` operator.
//
// # ZK design
// `Sqrt` is a transcendental function and cannot be expressed as a polynomial
// over a finite field. This hint performs the computation **outside** the
// circuit (native f64) and injects the result as an unconstrained witness.
//
// # Soundness limitation
// The hint alone adds no constraint. The `SqrtLayer` circuit pairs this hint
// with a LogUp range check that bounds the output to `[0, 2^n_bits)`.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const SQRT_HINT_KEY: &str = "jstprove.sqrt_hint";

/// Hint function for elementwise `Sqrt` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` (non-negative fixed-point).
/// - `inputs[1]`: the scaling factor `scale = 2^scale_exponent`.
///
/// # Outputs
/// - `outputs[0]`: `round(sqrt(x_q / scale) * scale)`, clamped to `[0, i64::MAX]`.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 2` or `outputs.len() != 1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn sqrt_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "sqrt_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "sqrt_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);

    let scale_u256 = inputs[1].to_u256();
    let scale_u64: u64 = scale_u256.as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "sqrt_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    // sqrt is only defined for non-negative inputs.
    let x_real = (x_i64 as f64 / scale_f64).max(0.0);
    let y_real = x_real.sqrt();

    let y_scaled = y_real * scale_f64;
    let y_q: i64 = if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled < 0.0 {
        0
    } else {
        y_scaled.round() as i64
    };

    outputs[0] = F::from_u256(U256::from(y_q as u64));
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
        sqrt_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs[0].to_u256().as_u64() as i64
    }

    #[test]
    fn sqrt_hint_zero_input() {
        // sqrt(0) = 0
        let scale: u64 = 1 << 18;
        assert_eq!(run_hint(0, scale), 0);
    }

    #[test]
    fn sqrt_hint_one() {
        // sqrt(1.0): x_q = scale, y_q = scale
        let scale: u64 = 1 << 18;
        let x_q = scale as i64;
        let result = run_hint(x_q, scale);
        assert!((result - scale as i64).abs() <= 1, "got {result}");
    }

    #[test]
    fn sqrt_hint_four() {
        // sqrt(4.0): x_q = 4*scale, y_q = 2*scale
        let scale: u64 = 1 << 18;
        let x_q = 4 * scale as i64;
        let result = run_hint(x_q, scale);
        let expected = 2 * scale as i64;
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn sqrt_hint_negative_clamps_to_zero() {
        // sqrt of negative → 0
        let scale: u64 = 1 << 18;
        let x_q = -(scale as i64);
        let result = run_hint(x_q, scale);
        assert_eq!(result, 0);
    }

    #[test]
    fn sqrt_hint_large_input_clamps() {
        // Very large x_q should produce a finite, non-negative result ≤ i64::MAX.
        let scale: u64 = 1 << 18;
        let result = run_hint(i64::MAX, scale);
        assert!(result >= 0, "expected non-negative, got {result}");
    }
}
