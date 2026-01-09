use arith::Field;
use expander_compiler::frontend::Error;

pub fn exp_bucket_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let x_shifted = inputs[0].to_u256().as_u64();
    let scale_exponent = inputs[1].to_u256().as_u32() as usize;

    let (idx, out) = compute_exp_bucket(x_shifted, scale_exponent);

    outputs[0] = F::from(idx);
    outputs[1] = F::from(out as u32);

    Ok(())
}

fn compute_exp_bucket(x_shifted: u64, scale_exponent: usize) -> (u32, u64) {
    let scale = 1u64 << scale_exponent;
    let bucket_width = scale >> 4;

    let idx = if bucket_width == 0 {
        0u32
    } else {
        (x_shifted / bucket_width).min(255) as u32
    };

    let x_unscaled = (idx as f64 / 16.0) - 8.0;
    let exp_val = x_unscaled.exp();
    let exp_clamped = exp_val.min(2981.0);
    let out = (exp_clamped * scale as f64).round() as u64;

    (idx, out)
}

pub fn softmax_div_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let exp_val = inputs[0].to_u256().as_u64();
    let sum = inputs[1].to_u256().as_u64();
    let scale = inputs[2].to_u256().as_u64();

    let (quotient, remainder) = if sum == 0 {
        (0u64, 0u64)
    } else {
        let scaled_exp = exp_val * scale;
        let q = scaled_exp / sum;
        let r = scaled_exp % sum;
        (q, r)
    };

    outputs[0] = F::from(quotient as u32);
    outputs[1] = F::from(remainder as u32);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_bucket_center() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 8 * scale;

        let (idx, out) = compute_exp_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 128, "x=0 should map to bucket 128");
        let expected_exp = scale;
        assert_eq!(out, expected_exp, "exp(0) should be 1.0 * scale");
    }

    #[test]
    fn test_exp_bucket_negative() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 0;

        let (idx, out) = compute_exp_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 0, "x=-8 should map to bucket 0");
        let expected_exp = ((-8.0_f64).exp() * scale as f64).round() as u64;
        assert_eq!(out, expected_exp);
    }

    #[test]
    fn test_exp_bucket_positive() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 16 * scale - 1;

        let (idx, out) = compute_exp_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 255, "x=+8 should map to bucket 255");
    }

    #[test]
    fn test_exp_output_monotonic() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let bucket_width = scale >> 4;

        let mut prev_out = 0u64;
        for bucket in 0..=255u32 {
            let x_shifted = (bucket as u64) * bucket_width;
            let (_, out) = compute_exp_bucket(x_shifted, scale_exponent);
            assert!(out >= prev_out, "exp should be monotonically increasing");
            prev_out = out;
        }
    }
}
