use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::{exp::compute_exp_quantized, field_to_i64};

pub const SOFTMAX_VERIFIED_HINT_KEY: &str = "jstprove.softmax_verified_hint";

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::cast_possible_wrap
)]
/// # Errors
/// Returns `Error::UserError` on arity mismatch or zero/overflowing scale.
pub fn softmax_verified_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let total_out = outputs.len();
    if total_out < 3 || total_out % 2 == 0 {
        return Err(Error::UserError(format!(
            "softmax_verified_hint: expected 2n+1 outputs (n>=1), got {total_out}"
        )));
    }
    let n = (total_out - 1) / 2;
    if inputs.len() != n + 1 {
        return Err(Error::UserError(format!(
            "softmax_verified_hint: expected {} inputs (n={n} + 1 scale), got {}",
            n + 1,
            inputs.len()
        )));
    }

    let scale_u256 = inputs[n].to_u256();
    if scale_u256 > U256::from(u64::MAX) {
        return Err(Error::UserError(format!(
            "softmax_verified_hint: scale {scale_u256} exceeds u64::MAX"
        )));
    }
    let scale_u64 = scale_u256.as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "softmax_verified_hint: scale is zero".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    let xs_i64: Vec<i64> = inputs[..n].iter().map(|&x| field_to_i64(x)).collect();
    let xs_f64: Vec<f64> = xs_i64.iter().map(|&x| x as f64 / scale_f64).collect();

    let mut max_idx = 0usize;
    let mut max_val = xs_i64[0];
    for (i, &v) in xs_i64.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }

    outputs[n] = inputs[max_idx];

    let max_x_f64 = xs_f64[max_idx];
    let exps_f64: Vec<f64> = xs_f64.iter().map(|&x| (x - max_x_f64).exp()).collect();
    let sum_exp_f64: f64 = exps_f64.iter().sum();

    for i in 0..n {
        let y_real = if sum_exp_f64 > 0.0 {
            exps_f64[i] / sum_exp_f64
        } else {
            1.0 / n as f64
        };
        let y_q = (y_real * scale_f64).round().clamp(0.0, i64::MAX as f64) as i64;
        outputs[i] = F::from_u256(U256::from(y_q as u64));

        let shifted_x_q = xs_i64[i] - max_val;
        let e_q = compute_exp_quantized(shifted_x_q, scale_u64);
        outputs[n + 1 + i] = F::from_u256(U256::from(e_q as u64));
    }

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

    #[test]
    fn verified_hint_uniform_input() {
        let scale: u64 = 1 << 18;
        let n = 4;
        let mut inputs: Vec<F> = vec![field(0); n];
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = vec![F::zero(); 2 * n + 1];
        softmax_verified_hint::<F>(&inputs, &mut outputs).unwrap();

        let sum: u64 = outputs[..n].iter().map(|o| o.to_u256().as_u64()).sum();
        assert!(
            (sum as i64 - scale as i64).unsigned_abs() <= n as u64,
            "sum {sum} not close to scale {scale}"
        );

        for i in 0..n {
            let e = outputs[n + 1 + i].to_u256().as_u64();
            assert_eq!(
                e, scale,
                "all exp values should equal scale for uniform input"
            );
        }
    }

    #[test]
    fn verified_hint_dominant_logit() {
        let scale: u64 = 1 << 18;
        let n = 3;
        let mut inputs = vec![field(10 * scale as i64), field(0), field(0)];
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = vec![F::zero(); 2 * n + 1];
        softmax_verified_hint::<F>(&inputs, &mut outputs).unwrap();

        let max_q = field_to_i64(outputs[n]);
        assert_eq!(max_q, 10 * scale as i64, "max should be the dominant logit");

        let y0 = outputs[0].to_u256().as_u64();
        assert!(y0 > scale / 2, "dominant output should be > scale/2");
    }

    #[test]
    fn verified_hint_exp_matches_compute() {
        let scale: u64 = 1 << 18;
        let n = 3;
        let xs = [2i64 * scale as i64, 1 * scale as i64, 0];
        let mut inputs: Vec<F> = xs.iter().map(|&x| field(x)).collect();
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = vec![F::zero(); 2 * n + 1];
        softmax_verified_hint::<F>(&inputs, &mut outputs).unwrap();

        let max_q = field_to_i64(outputs[n]);
        for i in 0..n {
            let e_from_hint = outputs[n + 1 + i].to_u256().as_u64() as i64;
            let shifted = xs[i] - max_q;
            let e_expected = compute_exp_quantized(shifted, scale);
            assert_eq!(
                e_from_hint, e_expected,
                "exp mismatch for element {i}: hint={e_from_hint}, expected={e_expected}"
            );
        }
    }
}
