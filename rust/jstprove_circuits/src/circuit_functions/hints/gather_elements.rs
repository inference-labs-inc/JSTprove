use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const GATHER_ELEMENTS_HINT_KEY: &str = "jstprove.gather_elements_hint";

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_errors_doc
)]
pub fn gather_elements_hint<F: FieldArith>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    if inputs.len() < 3 {
        return Err(Error::UserError(format!(
            "gather_elements_hint: expected at least 3 inputs (axis_size, index, data...), got {}",
            inputs.len()
        )));
    }
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "gather_elements_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }

    let axis_size = field_to_i64(inputs[0]) as usize;
    let data_len = inputs.len() - 2;
    let idx_raw = field_to_i64(inputs[1]);
    let resolved = if idx_raw < 0 {
        (idx_raw + axis_size as i64) as usize
    } else {
        idx_raw as usize
    };

    if resolved >= data_len {
        return Err(Error::UserError(format!(
            "gather_elements_hint: resolved index {resolved} out of bounds for data length {data_len}"
        )));
    }

    outputs[0] = inputs[2 + resolved];
    Ok(())
}
