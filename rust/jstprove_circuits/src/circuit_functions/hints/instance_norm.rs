use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const INSTANCE_NORM_HINT_KEY: &str = "jstprove.instance_norm_hint";

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc
)]
pub fn instance_norm_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() < 4 {
        return Err(Error::UserError(format!(
            "instance_norm_hint: expected at least 4 inputs \
             (spatial_values..., gamma_values..., beta_values..., scale), got {}",
            inputs.len()
        )));
    }

    let scale_u64 = inputs[inputs.len() - 1].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "instance_norm_hint: scale is zero".to_string(),
        ));
    }
    let alpha = scale_u64 as f64;

    let remaining = inputs.len() - 1;
    if remaining % 3 != 0 {
        return Err(Error::UserError(format!(
            "instance_norm_hint: (inputs.len()-1) must be divisible by 3 \
             (spatial + gamma + beta), got {remaining}"
        )));
    }
    let lane_size = remaining / 3;
    if outputs.len() != lane_size + 2 {
        return Err(Error::UserError(format!(
            "instance_norm_hint: expected {} outputs (lane_size + 2), got {}",
            lane_size + 2,
            outputs.len()
        )));
    }

    let x_vals: Vec<f64> = inputs[..lane_size]
        .iter()
        .map(|&v| field_to_i64(v) as f64)
        .collect();
    let gamma_vals: Vec<f64> = inputs[lane_size..2 * lane_size]
        .iter()
        .map(|&v| field_to_i64(v) as f64)
        .collect();
    let beta_vals: Vec<f64> = inputs[2 * lane_size..3 * lane_size]
        .iter()
        .map(|&v| field_to_i64(v) as f64)
        .collect();

    let n = lane_size as f64;
    let sum: f64 = x_vals.iter().sum();
    let mean = sum / n;
    let mean_q = mean.round();

    let variance: f64 = x_vals
        .iter()
        .map(|&v| (v - mean_q) * (v - mean_q))
        .sum::<f64>()
        / n;
    let std_dev = (variance + 1e-5 * alpha * alpha).sqrt();
    let inv_std = alpha / std_dev;
    let inv_std_q = inv_std.round();

    for i in 0..lane_size {
        let dev = x_vals[i] - mean_q;
        let norm_unscaled = dev * inv_std_q;
        let norm_q = (norm_unscaled / alpha).round();
        let y_real = norm_q * gamma_vals[i] + beta_vals[i];
        let y_q = (y_real / alpha).round() as i64;

        outputs[i] = if y_q >= 0 {
            F::from_u256(U256::from(y_q as u64))
        } else {
            let mag = U256::from(y_q.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        };
    }

    outputs[lane_size] = if mean_q >= 0.0 {
        F::from_u256(U256::from(mean_q as u64))
    } else {
        let mag = U256::from((-mean_q) as u64);
        F::from_u256(F::MODULUS - mag)
    };

    outputs[lane_size + 1] = F::from_u256(U256::from(inv_std_q as u64));
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
    fn instance_norm_basic() {
        let scale: u64 = 1 << 10;
        let alpha = scale as f64;
        let lane = 4;
        let x: Vec<i64> = vec![100, 200, 300, 400];
        let gamma: Vec<i64> = vec![scale as i64; lane];
        let beta: Vec<i64> = vec![0; lane];

        let mut inputs: Vec<F> = x.iter().map(|&v| field(v)).collect();
        inputs.extend(gamma.iter().map(|&v| field(v)));
        inputs.extend(beta.iter().map(|&v| field(v)));
        inputs.push(F::from_u256(U256::from(scale)));

        let mut outputs = vec![F::zero(); lane + 2];
        instance_norm_hint::<F>(&inputs, &mut outputs).unwrap();

        let mean_q = super::field_to_i64(outputs[lane]);
        assert!((mean_q as f64 - 250.0).abs() < alpha);
    }
}
