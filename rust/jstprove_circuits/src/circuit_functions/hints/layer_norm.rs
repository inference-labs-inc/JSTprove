// Hint function for the ONNX `LayerNormalization` operator.
//
// # ZK design
// LayerNormalization involves mean/variance computation, a square root, and an
// affine transformation — operations that cannot be expressed as polynomials
// over a finite field.  This hint performs the full computation **outside** the
// circuit (native f64) and injects all output elements as unconstrained
// witnesses.
//
// # Input layout
// `inputs[0..n-1]`   : quantised input elements `x_q[i]`  (scale α¹)
// `inputs[n..2n-1]`  : quantised gamma elements `γ_q[i]`  (scale α¹)
// `inputs[2n..3n-1]` : quantised beta elements  `β_q[i]`  (scale α²)
// `inputs[3n]`       : the scaling factor `scale = 2^scale_exponent`
// where `n = outputs.len()`.
//
// # Soundness limitation
// The hint alone adds no constraint.  Unlike Exp/Sigmoid/Softmax, no LogUp
// range check is applied because LayerNorm outputs can be negative.  The
// output variables propagate into downstream arithmetic layers (e.g. Gemm)
// whose constraints indirectly verify the computation through the overall
// output equality check.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

/// Hint key used to register and look up this function.
pub const LAYER_NORM_HINT_KEY: &str = "jstprove.layer_norm_hint";

/// Default epsilon used for variance stabilisation (ONNX LayerNorm default).
const EPSILON: f64 = 1e-5;

/// Hint function for `LayerNormalization` over a fixed-point integer lane.
///
/// # Inputs
/// - `inputs[0..n-1]`   : quantised input `x_q[i]` (scale α¹).
///   Values above `p/2` are treated as negative (two's-complement mod p).
/// - `inputs[n..2n-1]`  : quantised gamma `γ_q[i]` (scale α¹).
/// - `inputs[2n..3n-1]` : quantised beta `β_q[i]` (scale α²).
/// - `inputs[3n]`       : the scaling factor `scale = 2^scale_exponent`.
///
/// # Outputs
/// - `outputs[i]`: `round(y_f[i] * scale)` where
///   `y_f[i] = (x_f[i] - mean) / sqrt(var + ε) * γ_f[i] + β_f[i]`.
///   Negative outputs are encoded as field elements via two's-complement mod p.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != 3 * n + 1`.
#[allow(
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn layer_norm_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let n = outputs.len();
    let expected = 3 * n + 1;
    if inputs.len() != expected {
        return Err(Error::UserError(format!(
            "layer_norm_hint: expected {expected} inputs (3×n={n} + 1 scale), got {}",
            inputs.len()
        )));
    }

    let p_half = F::MODULUS / 2;

    // Decode a field element as a signed i64 using two's-complement convention.
    let decode_i64 = |x: F| -> i64 {
        let xu = x.to_u256();
        if xu > p_half {
            // Negative: -(p - xu)
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

    // Decode scale (always positive, fits in u64).
    let scale_u64 = inputs[3 * n].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "layer_norm_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;
    let scale_sq = scale_f64 * scale_f64;

    // Decode quantised inputs to real-valued f64.
    let xs_f64: Vec<f64> = inputs[..n]
        .iter()
        .map(|&x| decode_i64(x) as f64 / scale_f64)
        .collect();

    // gamma is quantised at α¹: γ_f[i] = γ_q[i] / scale
    let gammas_f64: Vec<f64> = inputs[n..2 * n]
        .iter()
        .map(|&g| decode_i64(g) as f64 / scale_f64)
        .collect();

    // beta is quantised at α²: β_f[i] = β_q[i] / scale²
    let betas_f64: Vec<f64> = inputs[2 * n..3 * n]
        .iter()
        .map(|&b| decode_i64(b) as f64 / scale_sq)
        .collect();

    // Compute mean.
    let mean: f64 = xs_f64.iter().sum::<f64>() / n as f64;

    // Compute variance.
    let var: f64 = xs_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    // Compute inverse standard deviation.
    let inv_std = 1.0 / (var + EPSILON).sqrt();

    for i in 0..n {
        let normalized = (xs_f64[i] - mean) * inv_std;
        let y_f = normalized * gammas_f64[i] + betas_f64[i];

        // Re-quantise: y_q = round(y_f * scale).  Can be negative.
        let y_scaled = y_f * scale_f64;
        let y_q: i64 = if y_scaled >= i64::MAX as f64 {
            i64::MAX
        } else if y_scaled <= i64::MIN as f64 {
            i64::MIN
        } else {
            y_scaled.round() as i64
        };

        // Encode y_q as a field element (two's-complement mod p for negatives).
        outputs[i] = if y_q >= 0 {
            F::from_u256(U256::from(y_q as u64))
        } else {
            let mag = U256::from(y_q.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        };
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

    fn run_layer_norm(xs: &[i64], gammas: &[i64], betas: &[i64], scale: u64) -> Vec<i64> {
        let n = xs.len();
        let mut inputs: Vec<F> = xs.iter().map(|&x| field(x)).collect();
        inputs.extend(gammas.iter().map(|&g| field(g)));
        inputs.extend(betas.iter().map(|&b| field(b)));
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = vec![F::zero(); n];
        layer_norm_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs.iter().map(decode_signed).collect()
    }

    #[test]
    fn layer_norm_wrong_input_count_returns_error() {
        let mut outputs = [F::zero(); 2];
        // 2 outputs → expects 7 inputs (3×2 + 1); give 5 → error
        let too_few: Vec<F> = vec![F::zero(); 5];
        assert!(layer_norm_hint::<F>(&too_few, &mut outputs).is_err());
    }

    #[test]
    fn layer_norm_zero_scale_returns_error() {
        let n = 2;
        let mut outputs = vec![F::zero(); n];
        // all zeros including scale
        let inputs = vec![F::zero(); 3 * n + 1];
        assert!(layer_norm_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn layer_norm_unit_gamma_zero_beta_output_sum_near_zero() {
        // With gamma=1 and beta=0, normalized outputs should sum near zero.
        let scale: u64 = 1 << 18;
        let s = scale as i64;
        // x = [1.0, -1.0, 0.0, 2.0] quantised at scale
        let xs = vec![s, -s, 0, 2 * s];
        // gamma = 1.0 (quantised at α¹ → scale)
        let gammas = vec![s, s, s, s];
        // beta = 0.0 (quantised at α² → 0)
        let betas = vec![0i64, 0, 0, 0];
        let outs = run_layer_norm(&xs, &gammas, &betas, scale);
        // Sum of layer-normed outputs is approximately 0 (sum of normalised values = 0).
        let sum: i64 = outs.iter().sum();
        assert!(
            sum.abs() < s / 10,
            "sum of layer_norm outputs {sum} should be close to 0"
        );
    }

    #[test]
    fn layer_norm_uniform_input_gives_zero_output_with_zero_beta() {
        // All inputs equal → variance ≈ 0 → normalized ≈ 0 → y ≈ 0 * gamma + 0 = 0.
        let scale: u64 = 1 << 18;
        let s = scale as i64;
        let xs = vec![s, s, s, s]; // uniform
        let gammas = vec![s, s, s, s];
        let betas = vec![0i64, 0, 0, 0];
        let outs = run_layer_norm(&xs, &gammas, &betas, scale);
        for (i, &o) in outs.iter().enumerate() {
            assert!(
                o.abs() < 10,
                "output[{i}] = {o} should be ~0 for uniform input"
            );
        }
    }
}
