// Hint function for the ONNX `Sigmoid` operator.
//
// # ZK design
// `Sigmoid` is a transcendental function and cannot be expressed as a
// polynomial over a finite field. This hint performs the computation
// **outside** the circuit (native f64) and injects the result as an
// unconstrained witness.
//
// # Soundness limitation
// The hint alone adds no constraint. The `SigmoidLayer` circuit pairs this
// hint with a LogUp range check that bounds the output to `[0, 2^n_bits)`,
// which constrains the output to a valid fixed-point range but does NOT prove
// it equals `sigmoid(x)` exactly. Full soundness would require a lookup table
// covering all quantised input values — a planned future extension.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const SIGMOID_HINT_KEY: &str = "jstprove.sigmoid_hint";

/// Hint function for elementwise `Sigmoid` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` as a field element (may encode a
///   negative integer in two's complement mod p: if `x_q >= p/2`, it is
///   treated as negative).
/// - `inputs[1]`: the scaling factor `scale = 2^scale_exponent` as a
///   field element (always a positive integer less than p/2).
///
/// # Outputs
/// - `outputs[0]`: `round(sigmoid(x_q / scale) * scale)`, clamped to
///   `[0, i64::MAX]`, stored as a field element.
///   `sigmoid(x) = 1 / (1 + exp(-x))`
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 2` or `outputs.len() != 1`.
/// Out-of-range values are clamped, never an error.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn sigmoid_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "sigmoid_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "sigmoid_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);

    // Decode scale as u64 (always positive, fits in u64 for practical scales)
    let scale_u64 = inputs[1].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "sigmoid_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    // Compute sigmoid in f64 on the real-valued (de-quantised) input.
    // sigmoid(x) = 1 / (1 + exp(-x))
    let x_real = x_i64 as f64 / scale_f64;
    let y_real = 1.0 / (1.0 + (-x_real).exp());

    // Re-quantise: y_q = round(sigmoid(x_real) * scale), clamped to [0, i64::MAX].
    // sigmoid is always in (0, 1), so y_scaled is always in (0, scale).
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
            let mag = U256::from((-n) as u64);
            F::from_u256(F::MODULUS - mag)
        }
    }

    fn run_hint(x_q: i64, scale: u64) -> i64 {
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        sigmoid_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs[0].to_u256().as_u64() as i64
    }

    #[test]
    fn sigmoid_hint_wrong_input_count_returns_error() {
        let mut outputs = [F::zero()];
        assert!(sigmoid_hint::<F>(&[], &mut outputs).is_err());
        assert!(sigmoid_hint::<F>(&[F::zero()], &mut outputs).is_err());
        assert!(sigmoid_hint::<F>(&[F::zero(); 3], &mut outputs).is_err());
    }

    #[test]
    fn sigmoid_hint_wrong_output_count_returns_error() {
        let inputs = [F::zero(), F::from_u256(U256::from(256u64))];
        assert!(sigmoid_hint::<F>(&inputs, &mut []).is_err());
        let mut outputs = [F::zero(); 2];
        assert!(sigmoid_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn sigmoid_hint_zero_input() {
        // sigmoid(0) = 0.5; y_q = round(0.5 * scale)
        let scale: u64 = 1 << 8; // 256
        let result = run_hint(0, scale);
        assert_eq!(result, scale as i64 / 2); // round(0.5 * 256) = 128
    }

    #[test]
    fn sigmoid_hint_large_positive_saturates_to_scale() {
        // sigmoid(very large) → 1.0; y_q ≈ scale
        let scale: u64 = 1 << 8;
        let x_q = 1000 * scale as i64; // x_real = 1000.0
        let result = run_hint(x_q, scale);
        assert_eq!(result, scale as i64, "sigmoid(+inf) should be ≈ scale");
    }

    #[test]
    fn sigmoid_hint_large_negative_saturates_to_zero() {
        // sigmoid(very negative) → 0.0; y_q = 0
        let scale: u64 = 1 << 8;
        let x_q = -1000 * scale as i64; // x_real = -1000.0
        let result = run_hint(x_q, scale);
        assert_eq!(result, 0, "sigmoid(-inf) should be 0");
    }

    #[test]
    fn sigmoid_hint_positive_input() {
        // sigmoid(1.0): y_q = round(1 / (1 + exp(-1)) * scale)
        let scale: u64 = 1 << 8; // 256
        let x_q = scale as i64; // x_real = 1.0
        let result = run_hint(x_q, scale);
        let expected = (1.0_f64 / (1.0 + (-1.0_f64).exp()) * scale as f64).round() as i64;
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn sigmoid_hint_negative_input() {
        // sigmoid(-1.0) = 1 - sigmoid(1.0) (antisymmetry)
        let scale: u64 = 1 << 8;
        let x_q = -(scale as i64); // x_real = -1.0
        let result = run_hint(x_q, scale);
        let expected = (1.0_f64 / (1.0 + (1.0_f64).exp()) * scale as f64).round() as i64;
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn sigmoid_hint_zero_scale_returns_error() {
        let inputs = [F::zero(), F::zero()]; // scale = 0
        let mut outputs = [F::zero()];
        assert!(sigmoid_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn sigmoid_output_is_non_negative() {
        let scale: u64 = 1 << 18;
        for &x_real in &[-10.0f64, -1.0, 0.0, 1.0, 10.0] {
            let x_q = (x_real * scale as f64).round() as i64;
            let result = run_hint(x_q, scale);
            assert!(
                result >= 0,
                "sigmoid output should be non-negative, got {result}"
            );
        }
    }

    #[test]
    fn sigmoid_output_bounded_by_scale() {
        let scale: u64 = 1 << 18;
        for &x_real in &[-10.0f64, -1.0, 0.0, 1.0, 10.0] {
            let x_q = (x_real * scale as f64).round() as i64;
            let result = run_hint(x_q, scale);
            assert!(
                result <= scale as i64,
                "sigmoid output {result} should be <= scale {scale}"
            );
        }
    }
}
