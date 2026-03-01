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

/// Hint key used to register and look up this function.
pub const EXP_HINT_KEY: &str = "jstprove.exp_hint";

/// Hint function for elementwise `Exp` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` as a field element (may encode a
///   negative integer in two's complement mod p: if `x_q >= p/2`, it is
///   treated as negative).
/// - `inputs[1]`: the scaling factor `scale = 2^scale_exponent` as a
///   field element (always a positive integer less than p/2).
///
/// # Outputs
/// - `outputs[0]`: `round(exp(x_q / scale) * scale)`, clamped to
///   `[0, i64::MAX]`, stored as a field element.
///
/// # Panics
/// Does not panic; out-of-range inputs are clamped.
pub fn exp_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    // Decode x_q from field element using two's complement convention:
    // values >= p/2 represent negative integers.
    let p_half = F::MODULUS / 2;
    let x_u256 = inputs[0].to_u256();
    let x_i64: i64 = if x_u256 > p_half {
        // Negative: -(p - x_u256)
        let neg_magnitude = F::MODULUS - x_u256;
        // Clamp to i64 range: if the magnitude exceeds i64::MAX, saturate.
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
    let scale_u256 = inputs[1].to_u256();
    let scale_u64: u64 = scale_u256.as_u64();
    let scale_f64 = scale_u64 as f64;

    // Compute exp in f64 on the real-valued (de-quantised) input
    let x_real = x_i64 as f64 / scale_f64;
    let y_real = x_real.exp();

    // Re-quantise: y_q = round(exp(x_real) * scale), clamped to [0, i64::MAX]
    let y_scaled = y_real * scale_f64;
    let y_q: i64 = if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled < 0.0 {
        // exp(x) > 0 for all real x, so this only fires on numerical
        // underflow (extremely negative x_q); clamp to zero.
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
            // Encode negative as p + n
            let mag = U256::from((-n) as u64);
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
    fn exp_hint_very_negative_clamps_to_zero() {
        // exp(-1000) underflows to 0 after rounding
        let scale: u64 = 1 << 8;
        let x_q = -1000 * scale as i64;
        let result = run_hint(x_q, scale);
        assert_eq!(result, 0, "very negative input should produce 0");
    }
}
