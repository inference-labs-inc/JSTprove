use arith::Field;
use expander_compiler::frontend::Error;

pub fn sigmoid_bucket_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let x_shifted = inputs[0].to_u256().as_u64();
    let scale_exponent = inputs[1].to_u256().as_u32() as usize;

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

    outputs[0] = F::from(idx);
    outputs[1] = F::from(out);

    Ok(())
}
