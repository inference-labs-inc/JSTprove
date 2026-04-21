// Hint function for the ONNX `Gelu` operator.
//
// # ZK design
// `Gelu` (Gaussian Error Linear Unit) is a transcendental function and cannot
// be expressed as a polynomial over a finite field. This hint performs the
// computation **outside** the circuit (native f64) and injects the result as
// an unconstrained witness.
//
// GELU formula: gelu(x) = x * Φ(x) where Φ is the standard Gaussian CDF.
// We use the tanh approximation:
//   gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// # Soundness limitation
// The hint alone adds no constraint. The `GeluLayer` circuit pairs this
// hint with a LogUp range check that bounds the output, which constrains
// the output to a valid fixed-point range but does NOT prove it equals
// `gelu(x)` exactly. Full soundness would require a lookup table covering
// all quantised input values — a planned future extension.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

/// Hint key used to register and look up this function.
pub const GELU_HINT_KEY: &str = "jstprove.gelu_hint";

// Constants for GELU approximation
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4; // √(2/π)
const GELU_COEF: f64 = 0.044_715;

#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn compute_gelu_quantized(x_q: i64, scale: u64) -> i64 {
    let scale_f64 = scale as f64;
    let x_real = x_q as f64 / scale_f64;
    let inner = SQRT_2_OVER_PI * (x_real + GELU_COEF * x_real.powi(3));
    let y_real = 0.5 * x_real * (1.0 + inner.tanh());
    let y_scaled = y_real * scale_f64;
    if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        y_scaled.round() as i64
    }
}

/// # Errors
/// Returns `Error::UserError` on arity mismatch or zero scale.
#[allow(
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn gelu_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "gelu_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "gelu_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let p_half = F::MODULUS / 2;
    let x_u256 = inputs[0].to_u256();
    let x_i64: i64 = if x_u256 > p_half {
        let neg_magnitude = F::MODULUS - x_u256;
        let max_i64 = U256::from(i64::MAX as u64);
        if neg_magnitude > max_i64 {
            i64::MIN
        } else {
            -(neg_magnitude.as_u64() as i64)
        }
    } else {
        let max_i64 = U256::from(i64::MAX as u64);
        if x_u256 > max_i64 {
            i64::MAX
        } else {
            x_u256.as_u64() as i64
        }
    };

    let scale_u64 = inputs[1].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "gelu_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }

    let y_q = compute_gelu_quantized(x_i64, scale_u64);

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
            let mag = (F::MODULUS - u).as_u64();
            -(mag as i64)
        } else {
            u.as_u64() as i64
        }
    }

    #[test]
    fn gelu_positive_input() {
        // GELU(1.0) ≈ 0.8413 (using exact erf) or ≈ 0.8411 (using tanh approx)
        let scale = 1_000_000i64;
        let x_q = scale; // x = 1.0
        let inputs = [field(x_q), field(scale)];
        let mut outputs = [F::from_u256(U256::ZERO)];

        gelu_hint::<F>(&inputs, &mut outputs).unwrap();
        let y_q = to_i64(outputs[0]);

        // Expected: ~841100 (0.8411 * 1_000_000)
        assert!((y_q - 841100).abs() < 1000, "y_q = {y_q}");
    }

    #[test]
    fn gelu_negative_input() {
        // GELU(-1.0) ≈ -0.1587 (exact) or ≈ -0.1589 (tanh approx)
        let scale = 1_000_000i64;
        let x_q = -scale; // x = -1.0
        let inputs = [field(x_q), field(scale)];
        let mut outputs = [F::from_u256(U256::ZERO)];

        gelu_hint::<F>(&inputs, &mut outputs).unwrap();
        let y_q = to_i64(outputs[0]);

        // Expected: ~-158900 (-0.1589 * 1_000_000)
        assert!((y_q + 158900).abs() < 1000, "y_q = {y_q}");
    }

    #[test]
    fn gelu_zero() {
        let scale = 1_000_000i64;
        let inputs = [field(0), field(scale)];
        let mut outputs = [F::from_u256(U256::ZERO)];

        gelu_hint::<F>(&inputs, &mut outputs).unwrap();
        let y_q = to_i64(outputs[0]);

        assert_eq!(y_q, 0);
    }

    #[test]
    fn compute_gelu_matches_hint() {
        let scale: u64 = 1 << 18;
        for &x_real in &[-3.0f64, -1.0, 0.0, 1.0, 3.0] {
            let x_q = (x_real * scale as f64).round() as i64;
            let from_compute = compute_gelu_quantized(x_q, scale);

            let inputs = [field(x_q), field(scale as i64)];
            let mut outputs = [F::from_u256(U256::ZERO)];
            gelu_hint::<F>(&inputs, &mut outputs).unwrap();
            let from_hint = to_i64(outputs[0]);

            assert_eq!(
                from_compute, from_hint,
                "compute_gelu_quantized and gelu_hint disagree for x_real={x_real}"
            );
        }
    }

    #[test]
    fn compute_gelu_monotonic_in_positive_domain() {
        let scale: u64 = 1 << 18;
        let mut prev = compute_gelu_quantized(0, scale);
        for x_q in 1i64..=500_000 {
            let curr = compute_gelu_quantized(x_q, scale);
            assert!(
                curr >= prev,
                "gelu must be non-decreasing for x>=0: f({}) = {} < f({}) = {}",
                x_q - 1,
                prev,
                x_q,
                curr
            );
            prev = curr;
        }
    }

    #[test]
    fn compute_gelu_asymptotically_linear() {
        let scale: u64 = 1 << 18;
        let x_q = 5 * scale as i64;
        let y = compute_gelu_quantized(x_q, scale);
        assert!(
            (y - x_q).abs() < scale as i64 / 100,
            "gelu(5) should be approximately 5, got {}",
            y as f64 / scale as f64
        );
    }

    #[test]
    fn compute_gelu_negative_saturates_near_zero() {
        let scale: u64 = 1 << 18;
        let x_q = -5 * scale as i64;
        let y = compute_gelu_quantized(x_q, scale);
        assert!(
            y.abs() < scale as i64 / 100,
            "gelu(-5) should be approximately 0, got {}",
            y as f64 / scale as f64
        );
    }
}
