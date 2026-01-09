use arith::Field;
use expander_compiler::frontend::Error;

const SIGMOID_TABLE_SIZE: usize = 256;

pub fn sigmoid_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let x_raw = inputs[0].to_u256();
    let scale = inputs[1].to_u256().as_u64();
    let _scale_exponent = inputs[2].to_u256().as_u32();

    let x_signed = if x_raw.leading_zeros() == 0 {
        -((!x_raw).as_i128() + 1)
    } else {
        x_raw.as_i128()
    };

    let x_unscaled = x_signed as f64 / scale as f64;
    let x_clamped = x_unscaled.clamp(-8.0, 8.0);
    let idx = ((x_clamped + 8.0) * (SIGMOID_TABLE_SIZE as f64 / 16.0)).floor() as u32;
    let idx = idx.min((SIGMOID_TABLE_SIZE - 1) as u32);

    let x_for_sigmoid = (idx as f64 / (SIGMOID_TABLE_SIZE as f64 / 16.0)) - 8.0;
    let sigmoid_val = 1.0 / (1.0 + (-x_for_sigmoid).exp());
    let sigmoid_scaled = (sigmoid_val * scale as f64).round() as u32;

    outputs[0] = F::from(idx);
    outputs[1] = F::from(sigmoid_scaled);

    Ok(())
}
