// Hint function for the ONNX `Exp` operator.
//
// # ZK design
// `Exp` is a transcendental function and cannot be expressed as a polynomial
// over a finite field. This hint performs the computation **outside** the
// circuit (native f64) and injects the result as an unconstrained witness.
//
// # Soundness limitation
// The hint alone adds no constraint. The `ExpLayer` circuit pairs this hint
// with a LogUp range check that bounds the output to `[0, 2^n_bits)`, which
// constrains the output to a valid fixed-point range but does NOT prove it
// equals `exp(x)` exactly.
//
// Full soundness would require a lookup table covering all quantised input
// values — a planned future extension.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const EXP_HINT_KEY: &str = "jstprove.exp_hint";

#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn compute_exp_quantized(x_q: i64, scale: u64) -> i64 {
    let scale_f64 = scale as f64;
    let x_real = x_q as f64 / scale_f64;
    let y_real = x_real.exp();
    let y_scaled = y_real * scale_f64;
    if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled < 0.0 {
        0
    } else {
        y_scaled.round() as i64
    }
}

/// # Errors
/// Returns `Error::UserError` on arity mismatch or zero scale.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn exp_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "exp_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "exp_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);
    let scale_u64: u64 = inputs[1].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "exp_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }

    let y_q = compute_exp_quantized(x_i64, scale_u64);
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
            // Encode negative as p + n
            let mag = U256::from(n.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    }

    fn run_hint(x_q: i64, scale: u64) -> i64 {
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        exp_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs[0].to_u256().as_u64() as i64
    }

    #[test]
    fn exp_hint_zero_input() {
        // exp(0) = 1.0; y_q = round(1.0 * scale) = scale
        let scale: u64 = 1 << 8;
        let result = run_hint(0, scale);
        assert_eq!(result, scale as i64);
    }

    #[test]
    fn exp_hint_positive_input() {
        // x_q = scale => x_real = 1.0; y_q = round(e * scale)
        let scale: u64 = 1 << 8; // 256
        let x_q = scale as i64; // represents x_real = 1.0
        let result = run_hint(x_q, scale);
        let expected = (std::f64::consts::E * scale as f64).round() as i64;
        // Allow ±1 rounding tolerance
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn exp_hint_negative_input() {
        // x_q = -scale => x_real = -1.0; y_q = round(exp(-1) * scale)
        let scale: u64 = 1 << 8;
        let x_q = -(scale as i64); // represents x_real = -1.0
        let result = run_hint(x_q, scale);
        let expected = ((-1.0f64).exp() * scale as f64).round() as i64;
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn exp_hint_wrong_input_count_returns_error() {
        let mut outputs = [F::zero()];
        // 0 inputs instead of 2
        assert!(exp_hint::<F>(&[], &mut outputs).is_err());
        // 1 input instead of 2
        assert!(exp_hint::<F>(&[F::zero()], &mut outputs).is_err());
        // 3 inputs instead of 2
        assert!(exp_hint::<F>(&[F::zero(); 3], &mut outputs).is_err());
    }

    #[test]
    fn exp_hint_wrong_output_count_returns_error() {
        let inputs = [F::zero(), F::from_u256(U256::from(256u64))];
        // 0 outputs instead of 1
        assert!(exp_hint::<F>(&inputs, &mut []).is_err());
        // 2 outputs instead of 1
        let mut outputs = [F::zero(); 2];
        assert!(exp_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn exp_hint_zero_scale_returns_error() {
        let inputs = [F::zero(), F::zero()]; // scale = 0
        let mut outputs = [F::zero()];
        assert!(exp_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn exp_hint_very_negative_clamps_to_zero() {
        let scale: u64 = 1 << 8;
        let x_q = -1000 * scale as i64;
        let result = run_hint(x_q, scale);
        assert_eq!(result, 0, "very negative input should produce 0");
    }

    #[test]
    fn compute_exp_matches_hint() {
        let scale: u64 = 1 << 18;
        for &x_real in &[-5.0f64, -1.0, 0.0, 1.0, 3.0] {
            let x_q = (x_real * scale as f64).round() as i64;
            let from_compute = compute_exp_quantized(x_q, scale);
            let from_hint = run_hint(x_q, scale);
            assert_eq!(
                from_compute, from_hint,
                "compute_exp_quantized and exp_hint disagree for x_real={x_real}"
            );
        }
    }

    #[test]
    fn compute_exp_output_always_non_negative() {
        let scale: u64 = 1 << 18;
        for x_q in -1_000_000i64..=1_000_000 {
            let y = compute_exp_quantized(x_q, scale);
            assert!(
                y >= 0,
                "exp output must be non-negative, got {y} for x_q={x_q}"
            );
        }
    }

    #[test]
    fn compute_exp_monotonic() {
        let scale: u64 = 1 << 18;
        let mut prev = compute_exp_quantized(-500_000, scale);
        for x_q in (-500_000i64 + 1)..=500_000 {
            let curr = compute_exp_quantized(x_q, scale);
            assert!(
                curr >= prev,
                "exp must be monotonically non-decreasing: f({}) = {} < f({}) = {}",
                x_q - 1,
                prev,
                x_q,
                curr
            );
            prev = curr;
        }
    }
}
