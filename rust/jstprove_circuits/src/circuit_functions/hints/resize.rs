// Hint function for the ONNX `Resize` operator (linear / bilinear mode).
//
// # ZK design
// Linear interpolation computes a weighted sum of corner values:
//   y = sum(x_i * w_i)
// where x_i are input elements at the floor/ceil corners and w_i are
// interpolation weights that sum to 1.
//
// All values are fixed-point at scale α = 2^scale_exponent:
//   x_i_q = round(x_i_real * α)
//   w_i_q = round(w_i_real * α)
//   y_q   = round(sum(x_i_q * w_i_q) / α)
//
// # Hint input layout
// `[x_corner_0, ..., x_corner_{n-1}, w_corner_0, ..., w_corner_{n-1}, scale]`
// where n = (inputs.len() - 1) / 2.
//
// # Soundness caveat
// The hint computes the interpolation outside the circuit; the circuit
// constrains the output via a LogUp range check (non-negative and bounded).

use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

/// Hint key for registration and look-up.
pub const RESIZE_HINT_KEY: &str = "jstprove.resize_hint";

/// Hint function for N-D linear (bilinear) resize interpolation.
///
/// # Inputs (length = 2 * n_corners + 1)
/// - `inputs[0..n_corners]`: quantised corner values `x_i_q` (field elements,
///   may encode negative integers in two's complement mod p).
/// - `inputs[n_corners..2*n_corners]`: quantised corner weights `w_i_q`
///   (always non-negative; values in [0, scale]).
/// - `inputs[2*n_corners]`: the scale factor `scale = 2^scale_exponent`.
///
/// # Output (length = 1)
/// - `outputs[0]`: `round(sum(x_i_q * w_i_q) / scale)`, clamped to
///   `[0, i64::MAX]`, stored as a field element.
///
/// # Errors
/// Returns [`Error::UserError`] when input/output lengths are invalid or
/// the input count is odd (expected 2*n+1).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn resize_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.is_empty() || inputs.len() % 2 == 0 {
        return Err(Error::UserError(format!(
            "resize_hint: expected 2*n+1 inputs (n corner values, n weights, scale), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "resize_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    // n_corners = (inputs.len() - 1) / 2
    let n = (inputs.len() - 1) / 2;
    let scale_u64: u64 = inputs[2 * n].to_u256().as_u64();
    if scale_u64 == 0 {
        return Err(Error::UserError(
            "resize_hint: scale is zero; cannot compute interpolation".to_string(),
        ));
    }

    // Compute weighted sum using i128 to avoid overflow.
    let mut sum_i128: i128 = 0;
    for i in 0..n {
        let x_i64 = field_to_i64(inputs[i]);
        let w_u256 = inputs[n + i].to_u256();
        let w_u64: u64 = w_u256.as_u64();
        sum_i128 += x_i64 as i128 * w_u64 as i128;
    }

    // y_q = round(sum / scale), clamped to [0, i64::MAX]
    let scale_i128 = scale_u64 as i128;
    let half = scale_i128 / 2;
    let y_q: i64 = if sum_i128 >= 0 {
        let rounded = (sum_i128 + half) / scale_i128;
        if rounded > i64::MAX as i128 {
            i64::MAX
        } else {
            rounded as i64
        }
    } else {
        // Negative sum — clamp to 0 (resize of non-negative activations).
        0
    };

    outputs[0] = F::from_u256(U256::from(y_q as u64));
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
        resize_hint::<F>(&inputs, &mut outputs).unwrap();
        outputs[0].to_u256().as_u64() as i64
    }

    #[test]
    fn nearest_equivalent_single_corner() {
        // With a single corner (weight = scale), output = input value.
        let scale: u64 = 256;
        let x_q: i64 = 1000;
        let result = run_hint(&[x_q], &[scale], scale);
        assert_eq!(result, x_q);
    }

    #[test]
    fn bilinear_midpoint() {
        // Two corners with equal weights (0.5 each): output = average.
        let scale: u64 = 256;
        let w: u64 = scale / 2; // 0.5 * scale
        let x0: i64 = 100;
        let x1: i64 = 200;
        let result = run_hint(&[x0, x1], &[w, w], scale);
        let expected = ((x0 + x1 + 1) / 2).max(0); // round(150) = 150
        assert!(
            (result - expected).abs() <= 1,
            "got {result}, expected ~{expected}"
        );
    }

    #[test]
    fn bilinear_floor_corner() {
        // Weight fully on floor corner: output = floor value.
        let scale: u64 = 256;
        let result = run_hint(&[400i64, 800i64], &[scale, 0], scale);
        assert_eq!(result, 400);
    }

    #[test]
    fn bilinear_ceil_corner() {
        // Weight fully on ceil corner: output = ceil value.
        let scale: u64 = 256;
        let result = run_hint(&[400i64, 800i64], &[0, scale], scale);
        assert_eq!(result, 800);
    }

    #[test]
    fn even_input_count_returns_error() {
        let mut outputs = [F::zero()];
        // 2 inputs is even — invalid
        assert!(resize_hint::<F>(&[F::zero(); 2], &mut outputs).is_err());
        // 4 inputs is even — invalid
        assert!(resize_hint::<F>(&[F::zero(); 4], &mut outputs).is_err());
    }

    #[test]
    fn zero_scale_returns_error() {
        let inputs = [F::zero(); 3]; // n=1: x, w, scale=0
        let mut outputs = [F::zero()];
        assert!(resize_hint::<F>(&inputs, &mut outputs).is_err());
    }
}
