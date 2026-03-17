// Hint function for the ONNX `Softmax` operator.
//
// # ZK design
// `Softmax` involves exponential and division operations that cannot be
// expressed as polynomials over a finite field. This hint performs the full
// computation **outside** the circuit (native f64) and injects all output
// elements as unconstrained witnesses.
//
// # Input layout
// `inputs[0..n-1]` : quantised input elements `x_q[i]`
// `inputs[n]`      : the scaling factor `scale = 2^scale_exponent`
// where `n = outputs.len()`.
//
// # Soundness limitation
// The hint alone adds no constraint. The `SoftmaxLayer` circuit pairs this
// hint with a LogUp range check that bounds each output to `[0, 2^n_bits)`,
// which ensures values stay within the quantised range but does NOT prove
// that `softmax(x)` was computed correctly. Full soundness would require a
// dedicated lookup proof for the exponential and a range-checked division —
// a planned future extension.

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key used to register and look up this function.
pub const SOFTMAX_HINT_KEY: &str = "jstprove.softmax_hint";

/// Hint function for `Softmax` over a fixed-point integer slice.
///
/// # Inputs
/// - `inputs[0..n-1]`: quantised input `x_q[i]` as field elements.  Values
///   above `p/2` are treated as negative (two's-complement mod p).
/// - `inputs[n]`: the scaling factor `scale = 2^scale_exponent` as a field
///   element (always a positive integer less than p/2).
///
/// # Outputs
/// - `outputs[i]`: `round(softmax(x_real)[i] * scale)`, clamped to
///   `[0, i64::MAX]`, stored as a field element.
///
/// # Errors
/// Returns [`Error::UserError`] when `inputs.len() != outputs.len() + 1`.
/// Out-of-range values are clamped, never an error.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn softmax_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let n = outputs.len();
    if inputs.len() != n + 1 {
        return Err(Error::UserError(format!(
            "softmax_hint: expected {} inputs (n={n} elements + 1 scale), got {}",
            n + 1,
            inputs.len()
        )));
    }

    let scale_u64 = inputs[n].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "softmax_hint: scale is zero; cannot de-quantise input".to_string(),
        ));
    }
    let scale_f64 = scale_u64 as f64;

    let xs_f64: Vec<f64> = inputs[..n]
        .iter()
        .map(|&x| field_to_i64(x) as f64 / scale_f64)
        .collect();

    // Numerically stable softmax: subtract max before computing exp.
    let max_x = xs_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs_f64.iter().map(|&x| (x - max_x).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();

    for i in 0..n {
        // Guard against degenerate sum (should not happen with finite inputs).
        let y_real = if sum_exp > 0.0 {
            exps[i] / sum_exp
        } else {
            1.0 / n as f64
        };

        // Re-quantise: y_q = round(y_real * scale), clamped to [0, i64::MAX].
        let y_scaled = y_real * scale_f64;
        let y_q: i64 = if y_scaled >= i64::MAX as f64 {
            i64::MAX
        } else if y_scaled < 0.0 {
            0
        } else {
            y_scaled.round() as i64
        };

        outputs[i] = F::from_u256(U256::from(y_q as u64));
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

    fn run_softmax(xs: &[i64], scale: u64) -> Vec<i64> {
        let mut inputs: Vec<F> = xs.iter().map(|&x| field(x)).collect();
        inputs.push(F::from_u256(U256::from(scale)));
        let mut outputs = vec![F::zero(); xs.len()];
        softmax_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs
            .iter()
            .map(|o| o.to_u256().as_u64() as i64)
            .collect()
    }

    #[test]
    fn softmax_hint_wrong_input_count_returns_error() {
        // 2 outputs → expects 3 inputs; give 2 → error
        let mut outputs = [F::zero(); 2];
        let too_few: Vec<F> = vec![F::zero(); 2]; // missing scale
        assert!(softmax_hint::<F>(&too_few, &mut outputs).is_err());
        // 4 inputs for 2 outputs → error
        let too_many: Vec<F> = vec![F::zero(); 4];
        assert!(softmax_hint::<F>(&too_many, &mut outputs).is_err());
    }

    #[test]
    fn softmax_hint_zero_scale_returns_error() {
        // 2 outputs → expects 3 inputs: [x_0, x_1, scale]; scale = 0
        let mut outputs = [F::zero(); 2];
        let inputs = vec![F::zero(), F::zero(), F::zero()]; // scale = 0
        assert!(softmax_hint::<F>(&inputs, &mut outputs).is_err());
    }

    #[test]
    fn softmax_uniform_input_sums_to_scale() {
        // For equal inputs softmax outputs should each be 1/n.
        let scale: u64 = 1 << 8; // 256
        let n = 4;
        let xs: Vec<i64> = vec![0i64; n]; // all zeros after quantisation
        let outs = run_softmax(&xs, scale);
        let sum: i64 = outs.iter().sum();
        // Each output should be round(0.25 * 256) = 64; sum = 256
        assert!(
            (sum - scale as i64).abs() <= n as i64,
            "sum {sum} not close to scale {scale}"
        );
    }

    #[test]
    fn softmax_large_winner_dominates() {
        // If one logit is much larger, its softmax output should be close to scale.
        let scale: u64 = 1 << 8; // 256
        let s = scale as i64;
        // x_real = [10, 0, 0]: first element dominates
        let xs = vec![10 * s, 0, 0];
        let outs = run_softmax(&xs, scale);
        assert!(
            outs[0] > scale as i64 / 2,
            "dominant logit output {} should be > scale/2",
            outs[0]
        );
        assert!(
            outs[1] < scale as i64 / 4,
            "non-dominant output {} should be small",
            outs[1]
        );
    }

    #[test]
    fn softmax_outputs_non_negative() {
        let scale: u64 = 1 << 8;
        let s = scale as i64;
        let xs = vec![-2 * s, 0, s, 3 * s];
        let outs = run_softmax(&xs, scale);
        for (i, &o) in outs.iter().enumerate() {
            assert!(o >= 0, "softmax output[{i}] = {o} is negative");
        }
    }
}
