// Hint function for the ONNX `GridSample` operator (bilinear mode).
//
// # ZK design
// Bilinear grid sampling computes a weighted sum of 4 corner pixel values:
//   y = sum(x_i * w_i)
// where x_i are input pixels at the 2×2 neighbourhood corners and w_i are
// bilinear interpolation weights derived from the fractional grid coordinates.
//
// All values are fixed-point at scale α = 2^scale_exponent:
//   x_i_q = round(x_i_real * α)
//   w_i_q = round(w_i_real * α)
//   y_q   = round(sum(x_i_q * w_i_q) / α)
//
// Out-of-bounds corners (zeros padding) have w_i_q = 0 and do not contribute.
//
// # Hint input layout
// `[x_corner_0, ..., x_corner_{n-1}, w_corner_0, ..., w_corner_{n-1}, scale]`
// where n = (inputs.len() - 1) / 2 (always 4 for standard bilinear).
//
// # Soundness caveat
// The hint computes the interpolation outside the circuit. Because bilinear
// weighted sums of signed pixel values can be negative, the output is a signed
// field element (two's complement mod p). The circuit currently applies a
// non-negative LogUp range check, so honest negative outputs are unprovable
// (liveness gap — full signed constraint is a planned future fix).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key for registration and look-up.
pub const GRIDSAMPLE_HINT_KEY: &str = "jstprove.gridsample_hint";

/// Hint function for bilinear grid-sample interpolation.
///
/// # Inputs (length = 2 * n_corners + 1)
/// - `inputs[0..n_corners]`: quantised corner pixel values `x_i_q` (field
///   elements, may encode negative integers in two's complement mod p).
/// - `inputs[n_corners..2*n_corners]`: quantised corner weights `w_i_q`
///   (always non-negative; values in `[0, scale]`; 0 for OOB corners).
/// - `inputs[2*n_corners]`: the scale factor `scale = 2^scale_exponent`.
///
/// # Output (length = 1)
/// - `outputs[0]`: `round(sum(x_i_q * w_i_q) / scale)`, clamped to
///   `[i64::MIN, i64::MAX]`, stored as a signed field element (two's complement
///   mod p: negative values encoded as `p - |y_q|`).
///
/// # Errors
/// Returns [`Error::UserError`] when input/output lengths are invalid or the
/// input count is even (expected 2*n+1).
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn gridsample_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.is_empty() || inputs.len() % 2 == 0 {
        return Err(Error::UserError(format!(
            "gridsample_hint: expected 2*n+1 inputs (n corner values, n weights, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "gridsample_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    // n_corners = (inputs.len() - 1) / 2
    let n = (inputs.len() - 1) / 2;
    if n == 0 {
        return Err(Error::UserError(
            "gridsample_hint: expected at least one corner/weight pair, got 0".to_string(),
        ));
    }

    let scale_u256 = inputs[2 * n].to_u256();
    if scale_u256 > U256::from(u64::MAX) {
        return Err(Error::UserError(format!(
            "gridsample_hint: scale value {scale_u256} exceeds u64::MAX"
        )));
    }
    let scale_u64 = scale_u256.as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "gridsample_hint: scale is zero; cannot compute interpolation".to_string(),
        ));
    }

    // Compute weighted sum with checked arithmetic to catch adversarial overflow.
    let mut sum_i128: i128 = 0;
    for i in 0..n {
        let x_i64 = field_to_i64(inputs[i]);
        let w_u256 = inputs[n + i].to_u256();
        if w_u256 > U256::from(u64::MAX) {
            return Err(Error::UserError(format!(
                "gridsample_hint: weight[{i}] value {w_u256} exceeds u64::MAX"
            )));
        }
        let w_u64 = w_u256.as_u64();
        let product = i128::from(x_i64)
            .checked_mul(i128::from(w_u64))
            .ok_or_else(|| {
                Error::UserError("gridsample_hint: overflow in weighted sum".to_string())
            })?;
        sum_i128 = sum_i128.checked_add(product).ok_or_else(|| {
            Error::UserError("gridsample_hint: overflow in weighted sum accumulation".to_string())
        })?;
    }

    // y_q = round(sum / scale), clamped to [i64::MIN, i64::MAX].
    let scale_i128 = i128::from(scale_u64);
    let half = scale_i128 / 2;
    let y_q: i64 = if sum_i128 >= 0 {
        let rounded = (sum_i128 + half) / scale_i128;
        if rounded > i128::from(i64::MAX) {
            i64::MAX
        } else {
            rounded as i64
        }
    } else {
        // Negative sum — compute signed rounding (symmetric half-away-from-zero).
        let rounded = (sum_i128 - half) / scale_i128;
        if rounded < i128::from(i64::MIN) {
            i64::MIN
        } else {
            rounded as i64
        }
    };

    // Reject negative results: the circuit applies a non-negative LogUp range
    // check, so a negative y_q would produce an unprovable witness.
    if y_q < 0 {
        return Err(Error::UserError(format!(
            "gridsample_hint: negative interpolation result {y_q} cannot be proven by the \
             circuit's non-negative range check (liveness gap; planned fix: signed range check)"
        )));
    }

    // Encode result as field element (two's complement for negatives).
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
    use ethnum::U256;
    use expander_compiler::field::BN254Fr;

    type F = BN254Fr;

    fn field_pos(n: u64) -> F {
        F::from_u256(U256::from(n))
    }

    fn run_hint(x_vals: &[i64], w_vals: &[u64], scale: u64) -> i64 {
        assert_eq!(x_vals.len(), w_vals.len());
        let n = x_vals.len();
        let mut inputs = Vec::with_capacity(2 * n + 1);
        for &x in x_vals {
            if x >= 0 {
                inputs.push(field_pos(x as u64));
            } else {
                let mag = U256::from(x.unsigned_abs());
                inputs.push(F::from_u256(F::MODULUS - mag));
            }
        }
        for &w in w_vals {
            inputs.push(field_pos(w));
        }
        inputs.push(field_pos(scale));
        let mut outputs = [F::zero()];
        gridsample_hint::<F>(&inputs, &mut outputs).unwrap();
        let v = outputs[0].to_u256();
        let p_half = F::MODULUS / 2;
        if v > p_half {
            // Use i128 to avoid panic when raw magnitude == 1<<63 (i64::MIN).
            let raw = (F::MODULUS - v).as_u64();
            (-(raw as i128)) as i64
        } else {
            v.as_u64() as i64
        }
    }

    #[test]
    fn single_corner_full_weight() {
        // One corner with weight = scale: output == input.
        let scale: u64 = 262144;
        let x_q: i64 = 500_000;
        let result = run_hint(&[x_q], &[scale], scale);
        assert_eq!(result, x_q);
    }

    #[test]
    fn four_corners_equal_weights() {
        // 4 corners with equal weights (0.25 each): output = average.
        let scale: u64 = 262144;
        let w: u64 = scale / 4;
        let x_vals = [100_000i64, 200_000, 300_000, 400_000];
        let w_vals = [w, w, w, w];
        let result = run_hint(&x_vals, &w_vals, scale);
        let expected = (x_vals.iter().sum::<i64>() + 2) / 4; // round(250_000)
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn oob_corner_zero_weight() {
        // Corner with weight 0 must not contribute even if its value is large.
        let scale: u64 = 262144;
        let result = run_hint(&[999_999i64, 1i64], &[scale, 0], scale);
        assert_eq!(result, 999_999);
    }

    #[test]
    fn even_input_count_returns_error() {
        let mut outputs = [F::zero()];
        assert!(gridsample_hint::<F>(&[F::zero(); 2], &mut outputs).is_err());
        assert!(gridsample_hint::<F>(&[F::zero(); 4], &mut outputs).is_err());
    }

    #[test]
    fn zero_scale_returns_error() {
        let inputs = [F::zero(); 3]; // n=1: x, w, scale=0
        let mut outputs = [F::zero()];
        assert!(gridsample_hint::<F>(&inputs, &mut outputs).is_err());
    }
}
