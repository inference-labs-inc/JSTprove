// Hint function for the ONNX `Erf` operator.
//
// # ZK design
// `Erf` (error function) is a transcendental function. This hint performs the
// computation **outside** the circuit (native f64) and injects the result as an
// unconstrained witness.
//
// # Implementation
// We use the Abramowitz & Stegun approximation 7.1.26 (accurate to ~1.5×10⁻⁷):
//   For x ≥ 0: t = 1/(1 + 0.3275911*x)
//   erf(x) ≈ 1 - (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵) * exp(-x²)
//   For x < 0: erf(x) = -erf(-x)
//
// # Soundness limitation
// The output is NOT constrained (erf can be negative).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const ERF_HINT_KEY: &str = "jstprove.erf_hint";

// Abramowitz & Stegun 7.1.26 coefficients
const P: f64 = 0.3275911;
const A1: f64 = 0.254829592;
const A2: f64 = -0.284496736;
const A3: f64 = 1.421413741;
const A4: f64 = -1.453152027;
const A5: f64 = 1.061405429;

#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Hint function for elementwise `Erf` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` (signed fixed-point).
/// - `inputs[1]`: the scaling factor `scale = 2^scale_exponent`.
///
/// # Outputs
/// - `outputs[0]`: `round(erf(x_q / scale) * scale)`, signed two's complement encoding.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 2` or `outputs.len() != 1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn erf_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "erf_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "erf_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);

    let scale_u64 = inputs[1].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "erf_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    let x_real = x_i64 as f64 / scale_f64;
    let y_real = erf_approx(x_real);

    let y_scaled = y_real * scale_f64;
    let y_q: i64 = if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        y_scaled.round() as i64
    };

    // Encode signed result as field element.
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

    fn run_hint(x_q: i64, scale: u64) -> i64 {
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        erf_hint::<F>(&inputs, &mut outputs).unwrap();
        to_i64(outputs[0])
    }

    #[test]
    fn erf_zero() {
        // erf(0) = 0
        let scale: u64 = 1 << 18;
        assert_eq!(run_hint(0, scale), 0);
    }

    #[test]
    fn erf_positive_one() {
        // erf(1.0) ≈ 0.8427
        let scale: u64 = 1 << 18;
        let x_q = scale as i64;
        let result = run_hint(x_q, scale);
        let expected = (0.8427_f64 * scale as f64).round() as i64;
        // Allow 0.1% tolerance
        let tol = (scale as i64) / 1000 + 1;
        assert!(
            (result - expected).abs() <= tol,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn erf_antisymmetry() {
        // erf(-x) = -erf(x)
        let scale: u64 = 1 << 18;
        let x_q = scale as i64;
        let pos = run_hint(x_q, scale);
        let neg = run_hint(-x_q, scale);
        assert!(
            (pos + neg).abs() <= 1,
            "antisymmetry failed: erf(x)={pos}, erf(-x)={neg}"
        );
    }

    #[test]
    fn erf_large_saturates_to_scale() {
        let scale: u64 = 1 << 18;
        let x_q = 10 * scale as i64;
        let result = run_hint(x_q, scale);
        assert!(
            (result - scale as i64).abs() <= 1,
            "should be ~scale, got {result}"
        );
    }
}
