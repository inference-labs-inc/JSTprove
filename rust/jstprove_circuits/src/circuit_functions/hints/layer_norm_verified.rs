use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

pub const LAYER_NORM_VERIFIED_HINT_KEY: &str = "jstprove.layer_norm_verified_hint";

const EPSILON: f64 = 1e-5;

#[allow(
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
/// # Errors
/// Returns `Error::UserError` on arity mismatch or zero scale.
pub fn layer_norm_verified_hint<F: FieldArith>(
    inputs: &[F],
    outputs: &mut [F],
) -> Result<(), Error> {
    let total_out = outputs.len();
    if total_out < 4 {
        return Err(Error::UserError(format!(
            "layer_norm_verified_hint: expected n+2 outputs (n>=2), got {total_out}"
        )));
    }
    let n = total_out - 2;
    let expected_in = 3 * n + 1;
    if inputs.len() != expected_in {
        return Err(Error::UserError(format!(
            "layer_norm_verified_hint: expected {expected_in} inputs (3×{n}+1), got {}",
            inputs.len()
        )));
    }

    let p_half = F::MODULUS / 2;
    let decode_i64 = |x: F| -> i64 {
        let xu = x.to_u256();
        if xu > p_half {
            let neg_magnitude = F::MODULUS - xu;
            let max_i64 = U256::from(i64::MAX as u64);
            if neg_magnitude > max_i64 {
                i64::MIN
            } else {
                -(neg_magnitude.as_u64() as i64)
            }
        } else {
            let max_i64 = U256::from(i64::MAX as u64);
            if xu > max_i64 {
                i64::MAX
            } else {
                xu.as_u64() as i64
            }
        }
    };

    let encode_i64 = |v: i64| -> F {
        if v >= 0 {
            F::from_u256(U256::from(v as u64))
        } else {
            let mag = U256::from(v.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    };

    let scale_u64 = inputs[3 * n].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "layer_norm_verified_hint: scale is zero".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;
    let scale_sq = scale_f64 * scale_f64;

    let xs_f64: Vec<f64> = inputs[..n]
        .iter()
        .map(|&x| decode_i64(x) as f64 / scale_f64)
        .collect();
    let gammas_f64: Vec<f64> = inputs[n..2 * n]
        .iter()
        .map(|&g| decode_i64(g) as f64 / scale_f64)
        .collect();
    let betas_f64: Vec<f64> = inputs[2 * n..3 * n]
        .iter()
        .map(|&b| decode_i64(b) as f64 / scale_sq)
        .collect();

    let mean: f64 = xs_f64.iter().sum::<f64>() / n as f64;
    let var: f64 = xs_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let inv_std = 1.0 / (var + EPSILON).sqrt();

    for i in 0..n {
        let normalized = (xs_f64[i] - mean) * inv_std;
        let y_f = normalized * gammas_f64[i] + betas_f64[i];
        let y_scaled = y_f * scale_f64;
        let y_q: i64 = if y_scaled >= i64::MAX as f64 {
            i64::MAX
        } else if y_scaled <= i64::MIN as f64 {
            i64::MIN
        } else {
            y_scaled.round() as i64
        };
        outputs[i] = encode_i64(y_q);
    }

    let mean_q = (mean * scale_f64).round() as i64;
    outputs[n] = encode_i64(mean_q);

    let inv_std_q = (inv_std * scale_f64).round() as i64;
    outputs[n + 1] = encode_i64(inv_std_q);

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

    fn decode_signed(o: &F) -> i64 {
        let xu = o.to_u256();
        let p_half = F::MODULUS / 2;
        if xu > p_half {
            let mag = (F::MODULUS - xu).as_u64();
            -(mag as i64)
        } else {
            xu.as_u64() as i64
        }
    }

    #[test]
    fn verified_hint_outputs_mean_and_inv_std() {
        let scale: u64 = 1 << 18;
        let s = scale as i64;
        let n = 4;
        let xs = vec![s, -s, 0, 2 * s];
        let gammas = vec![s; n];
        let betas = vec![0i64; n];

        let mut inputs: Vec<F> = xs.iter().map(|&x| field(x)).collect();
        inputs.extend(gammas.iter().map(|&g| field(g)));
        inputs.extend(betas.iter().map(|&b| field(b)));
        inputs.push(F::from_u256(U256::from(scale)));

        let mut outputs = vec![F::zero(); n + 2];
        layer_norm_verified_hint::<F>(&inputs, &mut outputs).unwrap();

        let mean_q = decode_signed(&outputs[n]);
        let expected_mean = (1.0 + (-1.0) + 0.0 + 2.0) / 4.0;
        let expected_mean_q = (expected_mean * scale as f64).round() as i64;
        assert!(
            (mean_q - expected_mean_q).abs() <= 1,
            "mean_q={mean_q}, expected≈{expected_mean_q}"
        );

        let inv_std_q = decode_signed(&outputs[n + 1]);
        assert!(inv_std_q > 0, "inv_std should be positive, got {inv_std_q}");
    }

    #[test]
    fn verified_hint_y_sum_near_zero_with_zero_beta() {
        let scale: u64 = 1 << 18;
        let s = scale as i64;
        let n = 4;
        let xs = vec![s, -s, 0, 2 * s];
        let gammas = vec![s; n];
        let betas = vec![0i64; n];

        let mut inputs: Vec<F> = xs.iter().map(|&x| field(x)).collect();
        inputs.extend(gammas.iter().map(|&g| field(g)));
        inputs.extend(betas.iter().map(|&b| field(b)));
        inputs.push(F::from_u256(U256::from(scale)));

        let mut outputs = vec![F::zero(); n + 2];
        layer_norm_verified_hint::<F>(&inputs, &mut outputs).unwrap();

        let sum: i64 = outputs[..n].iter().map(|o| decode_signed(o)).sum();
        assert!(
            sum.abs() < s / 10,
            "sum of outputs should be near 0 with zero beta, got {sum}"
        );
    }
}
