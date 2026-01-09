use arith::Field;
use expander_compiler::frontend::Error;

pub fn sigmoid_bucket_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let x_shifted = inputs[0].to_u256().as_u64();
    let scale_exponent = inputs[1].to_u256().as_u32() as usize;

    let (idx, out) = compute_sigmoid_bucket(x_shifted, scale_exponent);

    outputs[0] = F::from(idx);
    outputs[1] = F::from(out);

    Ok(())
}

fn compute_sigmoid_bucket(x_shifted: u64, scale_exponent: usize) -> (u32, u32) {
    let scale = 1u64 << scale_exponent;
    let bucket_width = scale >> 4;

    let idx = if bucket_width == 0 {
        0u32
    } else {
        (x_shifted / bucket_width).min(255) as u32
    };

    let x_unscaled = (idx as f64 / 16.0) - 8.0;
    let sigmoid_val = 1.0 / (1.0 + (-x_unscaled).exp());
    let out = (sigmoid_val * scale as f64).round() as u32;

    (idx, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_bucket_center() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 8 * scale;

        let (idx, out) = compute_sigmoid_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 128, "x=0 should map to bucket 128");
        let expected_sigmoid = (0.5 * scale as f64).round() as u32;
        assert_eq!(out, expected_sigmoid, "sigmoid(0) should be 0.5 * scale");
    }

    #[test]
    fn test_sigmoid_bucket_negative() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 0;

        let (idx, out) = compute_sigmoid_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 0, "x=-8 should map to bucket 0");
        let expected_sigmoid = (1.0 / (1.0 + 8.0_f64.exp()) * scale as f64).round() as u32;
        assert_eq!(out, expected_sigmoid);
    }

    #[test]
    fn test_sigmoid_bucket_positive() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 16 * scale - 1;

        let (idx, out) = compute_sigmoid_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 255, "x=+8 should map to bucket 255");
        let x_unscaled: f64 = (255.0 / 16.0) - 8.0;
        let expected_sigmoid = (1.0 / (1.0 + (-x_unscaled).exp()) * scale as f64).round() as u32;
        assert_eq!(out, expected_sigmoid);
    }

    #[test]
    fn test_sigmoid_bucket_clamping_high() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let x_shifted = 20 * scale;

        let (idx, _) = compute_sigmoid_bucket(x_shifted, scale_exponent);

        assert_eq!(idx, 255, "values beyond 16*scale should clamp to idx=255");
    }

    #[test]
    fn test_sigmoid_bucket_boundaries() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let bucket_width = scale >> 4;

        for bucket in 0..=255u32 {
            let x_shifted = (bucket as u64) * bucket_width;
            let (idx, _) = compute_sigmoid_bucket(x_shifted, scale_exponent);
            assert_eq!(idx, bucket, "bucket {} boundary check failed", bucket);
        }
    }

    #[test]
    fn test_sigmoid_output_monotonic() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let bucket_width = scale >> 4;

        let mut prev_out = 0u32;
        for bucket in 0..=255u32 {
            let x_shifted = (bucket as u64) * bucket_width;
            let (_, out) = compute_sigmoid_bucket(x_shifted, scale_exponent);
            assert!(out >= prev_out, "sigmoid should be monotonically increasing");
            prev_out = out;
        }
    }

    #[test]
    fn test_sigmoid_output_range() {
        let scale_exponent = 12;
        let scale = 1u64 << scale_exponent;
        let bucket_width = scale >> 4;

        for bucket in 0..=255u32 {
            let x_shifted = (bucket as u64) * bucket_width;
            let (_, out) = compute_sigmoid_bucket(x_shifted, scale_exponent);
            assert!(
                out <= scale as u32,
                "sigmoid output should not exceed scale"
            );
        }
    }
}
