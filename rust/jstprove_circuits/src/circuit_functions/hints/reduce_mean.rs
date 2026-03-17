// Hint function for the ONNX `ReduceMean` operator.
//
// # ZK design
// ReduceMean computes the mean of a set of values along specified axes.
// Division is not expressible as a low-degree polynomial, so a hint is used.
//
// # Protocol
// For each "lane" of n values (the elements being reduced):
//   hint inputs:  [x_0, ..., x_{n-1}, scale]
//   hint outputs: [y]
//   y_q = round(sum(x_i) / n)
//
// Note: since each x_i ≈ real_i * scale, the quantized mean is
//   round(sum(x_i) / n) = round(mean(real_i) * scale)
// which is correct without any explicit use of the scale value.
// The scale is included in the input list for interface consistency.
//
// # Soundness
// No constraint is added. Output can be negative (if inputs are negative).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const REDUCE_MEAN_HINT_KEY: &str = "jstprove.reduce_mean_hint";

/// Hint function for ReduceMean over a lane of fixed-point integers.
///
/// # Inputs
/// - `inputs[0..n-1]`: quantised values `x_i` as field elements (signed encoding).
/// - `inputs[n]` (last): scaling factor `scale = 2^scale_exponent` (positive, unused in computation).
///
/// # Outputs
/// - `outputs[0]`: `round(sum(x_i) / n)`, encoded as a signed field element.
///
/// # Errors
/// Returns [`Error::UserError`] for wrong counts (need ≥ 2 inputs: at least one value + scale).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn reduce_mean_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() < 2 {
        return Err(Error::UserError(format!(
            "reduce_mean_hint: expected at least 2 inputs (values + scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "reduce_mean_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let n = inputs.len() - 1; // last input is scale, not a value
    let sum: f64 = inputs[..n].iter().map(|&x| field_to_i64(x) as f64).sum();
    let mean = sum / n as f64;
    let y_q: i64 = mean.round() as i64;

    // Encode as field element: positive → y_q, negative → p − |y_q|.
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

    fn field_to_i64_test(f: F) -> i64 {
        let p_half = F::MODULUS / 2;
        let xu = f.to_u256();
        if xu > p_half {
            -((F::MODULUS - xu).as_u64() as i64)
        } else {
            xu.as_u64() as i64
        }
    }

    fn run_hint(values: &[i64], scale: u64) -> i64 {
        let mut inputs: Vec<F> = values.iter().map(|&v| field(v)).collect();
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = [F::zero()];
        reduce_mean_hint::<F>(&inputs, &mut outputs).unwrap();
        field_to_i64_test(outputs[0])
    }

    #[test]
    fn reduce_mean_uniform() {
        // mean([100, 100, 100, 100]) = 100
        assert_eq!(run_hint(&[100, 100, 100, 100], 1 << 18), 100);
    }

    #[test]
    fn reduce_mean_two_values() {
        // mean([200, 400]) = 300
        assert_eq!(run_hint(&[200, 400], 1 << 18), 300);
    }

    #[test]
    fn reduce_mean_signed() {
        // mean([-100, 100]) = 0
        assert_eq!(run_hint(&[-100, 100], 1 << 18), 0);
    }

    #[test]
    fn reduce_mean_all_negative() {
        // mean([-200, -400]) = -300
        assert_eq!(run_hint(&[-200, -400], 1 << 18), -300);
    }

    #[test]
    fn reduce_mean_rounding() {
        // mean([1, 2]) = 1.5, rounds to 2
        assert_eq!(run_hint(&[1, 2], 1 << 18), 2);
    }

    #[test]
    fn reduce_mean_too_few_inputs() {
        // Need at least 2 inputs (1 value + scale)
        let mut out = [F::zero()];
        assert!(reduce_mean_hint::<F>(&[F::zero()], &mut out).is_err());
        assert!(reduce_mean_hint::<F>(&[], &mut out).is_err());
    }

    #[test]
    fn reduce_mean_wrong_output_count() {
        let inputs = [field(100), field(200), F::from_u256(U256::from(1u64 << 18))];
        let mut out2 = [F::zero(); 2];
        assert!(reduce_mean_hint::<F>(&inputs, &mut out2).is_err());
    }
}
