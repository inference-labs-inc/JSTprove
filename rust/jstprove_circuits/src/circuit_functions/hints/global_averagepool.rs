use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const GLOBAL_AVERAGEPOOL_HINT_KEY: &str = "jstprove.global_averagepool_hint";

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc
)]
pub fn global_averagepool_hint<F: FieldArith>(
    inputs: &[F],
    outputs: &mut [F],
) -> Result<(), Error> {
    if inputs.is_empty() {
        return Err(Error::UserError(
            "global_averagepool_hint: expected at least 1 input, got 0".to_string(),
        ));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "global_averagepool_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let n = inputs.len();
    let sum: f64 = inputs.iter().map(|&x| field_to_i64(x) as f64).sum();
    let mean = sum / n as f64;
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
        F::from_u256(U256::from(n as u64))
    }

    fn run_hint(values: &[i64]) -> i64 {
        let inputs: Vec<F> = values.iter().map(|&v| field(v)).collect();
        let mut outputs = [F::zero()];
        global_averagepool_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs[0].to_u256().as_u64() as i64
    }

    #[test]
    fn global_averagepool_uniform() {
        assert_eq!(run_hint(&[100, 100, 100, 100]), 100);
    }

    #[test]
    fn global_averagepool_two_values() {
        assert_eq!(run_hint(&[200, 400]), 300);
    }

    #[test]
    fn global_averagepool_rounding() {
        assert_eq!(run_hint(&[1, 2]), 2);
    }
}
