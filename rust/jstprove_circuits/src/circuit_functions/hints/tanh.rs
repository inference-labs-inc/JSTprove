// Hint function for the ONNX `Tanh` operator.
//
// # ZK design
// `Tanh` is a transcendental function. This hint performs the computation
// **outside** the circuit (native f64) and injects the result as an
// unconstrained witness.
//
// # Soundness limitation
// The output is NOT constrained (tanh can be negative, so no range check).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const TANH_HINT_KEY: &str = "jstprove.tanh_hint";

/// Hint function for elementwise `Tanh` over fixed-point integers.
///
/// # Inputs
/// - `inputs[0]`: quantised input `x_q` (signed fixed-point).
/// - `inputs[1]`: the scaling factor `scale = 2^scale_exponent`.
///
/// # Outputs
/// - `outputs[0]`: `round(tanh(x_q / scale) * scale)`, signed two's complement encoding.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 2` or `outputs.len() != 1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn tanh_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() != 2 {
        return Err(Error::UserError(format!(
            "tanh_hint: expected 2 inputs (x_q, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "tanh_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let x_i64 = field_to_i64(inputs[0]);

    let scale_u256 = inputs[1].to_u256();
    if scale_u256 > U256::from(u64::MAX) {
        return Err(Error::UserError(format!(
            "tanh_hint: scale value {scale_u256} exceeds u64::MAX; cannot decode scaling factor",
        )));
    }
    let scale_u64 = scale_u256.as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "tanh_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    let x_real = x_i64 as f64 / scale_f64;
    let y_real = x_real.tanh();

    let y_scaled = y_real * scale_f64;
    let y_q: i64 = if y_scaled >= i64::MAX as f64 {
        i64::MAX
    } else if y_scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        y_scaled.round() as i64
    };

    // Encode signed result as field element (two's complement for negatives).
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
        tanh_hint::<F>(&inputs, &mut outputs).unwrap();
        to_i64(outputs[0])
    }

    #[test]
    fn tanh_zero() {
        // tanh(0) = 0
        let scale: u64 = 1 << 18;
        assert_eq!(run_hint(0, scale), 0);
    }

    #[test]
    fn tanh_positive() {
        let scale: u64 = 1 << 18;
        let x_q = scale as i64; // x = 1.0
        let result = run_hint(x_q, scale);
        let expected = (1.0f64.tanh() * scale as f64).round() as i64;
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn tanh_negative() {
        let scale: u64 = 1 << 18;
        let x_q = -(scale as i64); // x = -1.0
        let result = run_hint(x_q, scale);
        let expected = ((-1.0f64).tanh() * scale as f64).round() as i64;
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn tanh_large_positive_saturates() {
        let scale: u64 = 1 << 18;
        let x_q = 100 * scale as i64;
        let result = run_hint(x_q, scale);
        assert!(
            (result - scale as i64).abs() <= 1,
            "should be ~scale, got {result}"
        );
    }

    #[test]
    fn tanh_large_negative_saturates() {
        let scale: u64 = 1 << 18;
        let x_q = -100 * scale as i64;
        let result = run_hint(x_q, scale);
        assert!(
            (result + scale as i64).abs() <= 1,
            "should be ~-scale, got {result}"
        );
    }

    #[test]
    fn tanh_wrong_input_count() {
        let scale: u64 = 1 << 18;
        let inputs = [F::from_u256(U256::from(scale))]; // only 1 input instead of 2
        let mut outputs = [F::zero()];
        assert!(tanh_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn tanh_wrong_output_count() {
        let scale: u64 = 1 << 18;
        let inputs = [field(0), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero(), F::zero()]; // 2 outputs instead of 1
        assert!(tanh_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn tanh_zero_scale() {
        let inputs = [field(0), F::from_u256(U256::from(0u64))];
        let mut outputs = [F::zero()];
        assert!(tanh_hint::<F>(&inputs, &mut outputs).is_err());
    }
}
