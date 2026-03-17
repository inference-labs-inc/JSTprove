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
const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // √(2/π)
const GELU_COEF: f64 = 0.044715;

/// Hint function for elementwise `Gelu` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` as a field element (may encode a
///   negative integer in two's complement mod p: if `x_q >= p/2`, it is
///   treated as negative).
/// - `inputs[1]`: the scaling factor `scale = 2^scale_exponent` as a
///   field element (always a positive integer less than p/2).
///
/// # Outputs
/// - `outputs[0]`: `round(gelu(x_q / scale) * scale)`, stored as a field
///   element (using two's complement for negative values).
///   `gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 2` or `outputs.len() != 1`.
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

    // Decode x_q from field element using two's complement convention:
    // values >= p/2 represent negative integers.
    let p_half = F::MODULUS / 2;
    let x_u256 = inputs[0].to_u256();
    let x_i64: i64 = if x_u256 > p_half {
        // Negative: -(p - x_u256)
        let neg_magnitude = F::MODULUS - x_u256;
        let max_i64 = U256::from(i64::MAX as u64);
        if neg_magnitude > max_i64 {
            i64::MIN
        } else {
            -(neg_magnitude.as_u64() as i64)
        }
    } else {
        // Non-negative: fit into i64 (clamp if unexpectedly large)
        let max_i64 = U256::from(i64::MAX as u64);
        if x_u256 > max_i64 {
            i64::MAX
        } else {
            x_u256.as_u64() as i64
        }
    };

    // Decode scale as u64 (always positive, fits in u64 for practical scales)
    let scale_u64 = inputs[1].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "gelu_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    // Compute GELU in f64 on the real-valued (de-quantised) input.
    // gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    let x_real = x_i64 as f64 / scale_f64;
    let inner = SQRT_2_OVER_PI * (x_real + GELU_COEF * x_real.powi(3));
    let y_real = 0.5 * x_real * (1.0 + inner.tanh());

    // Re-quantise: y_q = round(gelu(x_real) * scale).
    // GELU can produce negative outputs for negative inputs, so we need
    // two's complement encoding for the result.
    let y_scaled = y_real * scale_f64;
    let y_q: i64 = if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        y_scaled.round() as i64
    };

    // Encode result as field element (two's complement for negatives)
    outputs[0] = if y_q >= 0 {
        F::from_u256(U256::from(y_q as u64))
    } else {
        // Negative: encode as p - |y_q|
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
        assert!((y_q - 841100).abs() < 1000, "y_q = {}", y_q);
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
        assert!((y_q + 158900).abs() < 1000, "y_q = {}", y_q);
    }

    #[test]
    fn gelu_zero() {
        // GELU(0) = 0
        let scale = 1_000_000i64;
        let inputs = [field(0), field(scale)];
        let mut outputs = [F::from_u256(U256::ZERO)];

        gelu_hint::<F>(&inputs, &mut outputs).unwrap();
        let y_q = to_i64(outputs[0]);

        assert_eq!(y_q, 0);
    }
}
