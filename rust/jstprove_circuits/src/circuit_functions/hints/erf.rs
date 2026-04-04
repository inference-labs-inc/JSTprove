use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const ERF_HINT_KEY: &str = "jstprove.erf_hint";
pub const ERF_ABS_HINT_KEY: &str = "jstprove.erf_abs_hint";

const P: f64 = 0.327_591_1;
const A1: f64 = 0.254_829_592;
const A2: f64 = -0.284_496_736;
const A3: f64 = 1.421_413_741;
const A4: f64 = -1.453_152_027;
const A5: f64 = 1.061_405_429;

#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - poly * (-x * x).exp())
}

#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn compute_erf_quantized(x_q: i64, scale: u64) -> i64 {
    let scale_f64 = scale as f64;
    let x_real = x_q as f64 / scale_f64;
    let y_real = erf_approx(x_real);
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
/// Returns `Error::UserError` when `inputs.len() != 1` or `outputs.len() != 2`.
#[allow(clippy::cast_sign_loss)]
pub fn erf_abs_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 1 {
        return Err(Error::UserError(format!(
            "erf_abs_hint: expected 1 input, got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 2 {
        return Err(Error::UserError(format!(
            "erf_abs_hint: expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let x = field_to_i64(inputs[0]);
    let abs_x = x.unsigned_abs();
    outputs[0] = F::from_u256(U256::from(abs_x));
    outputs[1] = F::from(u32::from(x >= 0));
    Ok(())
}

/// # Errors
/// Returns `Error::UserError` when `inputs.len() != 2` or `outputs.len() != 1`.
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

    let y_q = compute_erf_quantized(x_i64, scale_u64);

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
        let scale: u64 = 1 << 18;
        assert_eq!(run_hint(0, scale), 0);
    }

    #[test]
    fn erf_positive_one() {
        let scale: u64 = 1 << 18;
        let x_q = scale as i64;
        let result = run_hint(x_q, scale);
        let expected = (0.8427_f64 * scale as f64).round() as i64;
        let tol = (scale as i64) / 1000 + 1;
        assert!(
            (result - expected).abs() <= tol,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn erf_antisymmetry() {
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

    #[test]
    fn compute_erf_quantized_matches_hint() {
        let scale: u64 = 1 << 18;
        for x_q in [
            -2 * scale as i64,
            -(scale as i64),
            0,
            scale as i64,
            2 * scale as i64,
        ] {
            let hint_result = run_hint(x_q, scale);
            let compute_result = compute_erf_quantized(x_q, scale);
            assert_eq!(
                hint_result, compute_result,
                "mismatch at x_q={x_q}: hint={hint_result}, compute={compute_result}"
            );
        }
    }
}
