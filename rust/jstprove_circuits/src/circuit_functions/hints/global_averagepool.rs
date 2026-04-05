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

    let n = inputs.len() as i128;
    let sum: i128 = inputs.iter().map(|&x| i128::from(field_to_i64(x))).sum();
    let quotient = sum / n;
    let remainder = sum % n;
    let double_abs_rem = 2 * remainder.abs();
    let y_q = if double_abs_rem > n || (double_abs_rem == n && remainder > 0) {
        (quotient + remainder.signum()) as i64
    } else {
        quotient as i64
    };

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

    fn run_hint(values: &[i64]) -> i64 {
        let inputs: Vec<F> = values.iter().map(|&v| field(v)).collect();
        let mut outputs = [F::zero()];
        global_averagepool_hint::<F>(&inputs, &mut outputs).unwrap();
        field_to_i64(outputs[0])
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

    #[test]
    fn global_averagepool_negative() {
        assert_eq!(run_hint(&[-100, -100, -100]), -100);
    }

    #[test]
    fn global_averagepool_mixed_tie() {
        assert_eq!(run_hint(&[-1, 2]), 1);
    }

    #[test]
    fn global_averagepool_negative_half_tie() {
        assert_eq!(run_hint(&[-2, 1]), 0);
    }

    #[test]
    fn global_averagepool_i64_min() {
        assert_eq!(run_hint(&[i64::MIN]), i64::MIN);
    }

    #[test]
    fn global_averagepool_i64_max() {
        assert_eq!(run_hint(&[i64::MAX]), i64::MAX);
    }

    #[test]
    fn global_averagepool_i64_extrema_pair() {
        assert_eq!(run_hint(&[i64::MIN, i64::MAX]), 0);
    }
}
