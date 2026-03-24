// Hint function for the ONNX `AveragePool` operator.
//
// # ZK design
// AveragePool computes the average over a spatial kernel window.
// Division is not expressible as a low-degree polynomial, so a hint is used.
//
// # Protocol
// For each output position, sum the kernel window values and divide by kernel_size.
//   hint inputs:  [x_0, ..., x_{n-1}]
//   hint outputs: [y]
//   y_q = round(sum(x_i) / n)
//
// # Soundness limitation
// No constraint is added beyond a range check (output is non-negative).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const AVERAGEPOOL_HINT_KEY: &str = "jstprove.averagepool_hint";

/// Hint function for AveragePool over a kernel window of fixed-point integers.
///
/// # Inputs
/// - `inputs[0..n-1]`: quantised values `x_i` as field elements (n ≥ 1).
///
/// # Outputs
/// - `outputs[0]`: `round(sum(x_i) / n)`, encoded as a non-negative field element.
///
/// # Errors
/// Returns [`Error::UserError`] for wrong counts (need ≥ 1 input).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn averagepool_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.is_empty() {
        return Err(Error::UserError(
            "averagepool_hint: expected at least 1 input, got 0".to_string(),
        ));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "averagepool_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let n = inputs.len();
    let sum: f64 = inputs.iter().map(|&x| field_to_i64(x) as f64).sum();
    let mean = sum / n as f64;
    // Inputs are circuit-enforced non-negative, so mean >= 0.
    // No clamp: a negative result would produce a large field element that
    // fails the output range check in the circuit.
    let y_q = mean.round() as i64;

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
            let mag = U256::from(n.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    }

    fn run_hint(values: &[i64]) -> i64 {
        let inputs: Vec<F> = values.iter().map(|&v| field(v)).collect();
        let mut outputs = [F::zero()];
        averagepool_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs[0].to_u256().as_u64() as i64
    }

    #[test]
    fn averagepool_uniform() {
        assert_eq!(run_hint(&[100, 100, 100, 100]), 100);
    }

    #[test]
    fn averagepool_two_values() {
        assert_eq!(run_hint(&[200, 400]), 300);
    }

    #[test]
    fn averagepool_rounding() {
        // mean([1, 2]) = 1.5, rounds to 2
        assert_eq!(run_hint(&[1, 2]), 2);
    }

    #[test]
    fn averagepool_all_zeros() {
        assert_eq!(run_hint(&[0, 0, 0]), 0);
    }
}
