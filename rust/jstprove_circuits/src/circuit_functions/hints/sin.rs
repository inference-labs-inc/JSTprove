use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const SIN_HINT_KEY: &str = "jstprove.sin_hint";

#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn compute_sin_quantized(x_q: i64, scale: u64) -> i64 {
    let scale_f64 = scale as f64;
    let x_real = x_q as f64 / scale_f64;
    let y_real = x_real.sin();
    let y_scaled = y_real * scale_f64;
    if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        y_scaled.round() as i64
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
/// # Errors
/// Returns `Error::UserError` on invalid inputs.
#[allow(clippy::missing_errors_doc)]
pub fn sin_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "sin_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "sin_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);

    let scale_u256 = inputs[1].to_u256();
    if scale_u256 > U256::from(u64::MAX) {
        return Err(Error::UserError(format!(
            "sin_hint: scale value {scale_u256} exceeds u64::MAX"
        )));
    }
    let scale_u64 = scale_u256.as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "sin_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }

    let y_q = compute_sin_quantized(x_i64, scale_u64);

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
        sin_hint::<F>(&inputs, &mut outputs).unwrap();
        to_i64(outputs[0])
    }

    #[test]
    fn sin_zero() {
        let scale: u64 = 1 << 18;
        assert_eq!(run_hint(0, scale), 0);
    }

    #[test]
    fn sin_antisymmetry() {
        let scale: u64 = 1 << 18;
        let x_q = scale as i64;
        let pos = run_hint(x_q, scale);
        let neg = run_hint(-x_q, scale);
        assert!(
            (pos + neg).abs() <= 1,
            "antisymmetry failed: sin(x)={pos}, sin(-x)={neg}"
        );
    }

    #[test]
    fn sin_bounded_by_scale() {
        let scale: u64 = 1 << 18;
        for &x_real in &[-10.0f64, -1.0, 0.0, 1.0, 10.0] {
            let x_q = (x_real * scale as f64).round() as i64;
            let result = run_hint(x_q, scale);
            assert!(
                result.abs() <= scale as i64,
                "sin output {result} should be in [-scale, scale]"
            );
        }
    }

    #[test]
    fn compute_sin_matches_hint() {
        let scale: u64 = 1 << 18;
        for &x_real in &[-std::f64::consts::PI, -1.0, 0.0, 1.0, std::f64::consts::PI] {
            let x_q = (x_real * scale as f64).round() as i64;
            let from_compute = compute_sin_quantized(x_q, scale);
            let from_hint = run_hint(x_q, scale);
            assert_eq!(
                from_compute, from_hint,
                "compute_sin_quantized and sin_hint disagree for x_real={x_real}"
            );
        }
    }

    #[test]
    fn sin_hint_wrong_input_count() {
        let mut out = [F::zero()];
        assert!(sin_hint::<F>(&[], &mut out).is_err());
        assert!(sin_hint::<F>(&[F::zero()], &mut out).is_err());
    }

    #[test]
    fn sin_hint_zero_scale() {
        let inputs = [F::zero(), F::zero()];
        let mut out = [F::zero()];
        assert!(sin_hint::<F>(&inputs, &mut out).is_err());
    }
}
