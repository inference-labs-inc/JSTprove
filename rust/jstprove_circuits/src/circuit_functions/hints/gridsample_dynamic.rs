use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const GRIDSAMPLE_DYNAMIC_HINT_KEY: &str = "jstprove.gridsample_dynamic_hint";

#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::manual_midpoint,
    clippy::match_same_arms,
    clippy::missing_errors_doc
)]
pub fn gridsample_dynamic_hint<F: FieldArith>(
    inputs: &[F],
    outputs: &mut [F],
) -> Result<(), Error> {
    if outputs.len() != 1 {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: expected 1 output, got {}",
            outputs.len()
        )));
    }
    if inputs.len() < 8 {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: expected at least 8 inputs \
             (scale, h_in, w_in, align_corners, padding_mode, data..., x_norm, y_norm), got {}",
            inputs.len()
        )));
    }

    let scale_u256 = inputs[0].to_u256();
    if scale_u256 > U256::from(u64::MAX) || scale_u256 == U256::ZERO {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: invalid scale {scale_u256}"
        )));
    }
    let scale_u64 = scale_u256.as_u64();
    let alpha_f = scale_u64 as f64;

    let h_in = field_to_i64(inputs[1]) as usize;
    let w_in = field_to_i64(inputs[2]) as usize;
    let align_corners = field_to_i64(inputs[3]) != 0;
    let padding_mode_code = field_to_i64(inputs[4]);

    let data_size = h_in * w_in;
    let expected_len = 5 + data_size + 2;
    if inputs.len() != expected_len {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: expected {expected_len} inputs \
             (5 params + {data_size} data + 2 grid coords), got {}",
            inputs.len()
        )));
    }

    let data_offset = 5;
    let x_norm_q = field_to_i64(inputs[data_offset + data_size]);
    let y_norm_q = field_to_i64(inputs[data_offset + data_size + 1]);

    let x_norm = x_norm_q as f64 / alpha_f;
    let y_norm = y_norm_q as f64 / alpha_f;

    let unnorm = |norm: f64, size: usize, ac: bool| -> f64 {
        if ac {
            (norm + 1.0) / 2.0 * (size.saturating_sub(1) as f64)
        } else {
            (norm + 1.0) / 2.0 * size as f64 - 0.5
        }
    };

    let x_in = unnorm(x_norm, w_in, align_corners);
    let y_in = unnorm(y_norm, h_in, align_corners);

    let apply_pad = |coord: f64, size: usize| -> Option<f64> {
        match padding_mode_code {
            0 => {
                if coord < -0.5 || coord > (size as f64) - 0.5 {
                    None
                } else {
                    Some(coord.clamp(0.0, (size.saturating_sub(1)) as f64))
                }
            }
            1 => Some(coord.clamp(0.0, (size.saturating_sub(1)) as f64)),
            _ => Some(coord.clamp(0.0, (size.saturating_sub(1)) as f64)),
        }
    };

    let x_padded = apply_pad(x_in, w_in);
    let y_padded = apply_pad(y_in, h_in);

    let result: i64 = match (y_padded, x_padded) {
        (Some(y_p), Some(x_p)) => {
            let y_clamped = y_p.clamp(0.0, (h_in.saturating_sub(1)) as f64);
            let x_clamped = x_p.clamp(0.0, (w_in.saturating_sub(1)) as f64);

            let y_floor = y_clamped.floor();
            let x_floor = x_clamped.floor();
            let y_f = y_floor as usize;
            let x_f = x_floor as usize;
            let y_c = (y_clamped.ceil() as usize).min(h_in.saturating_sub(1));
            let x_c = (x_clamped.ceil() as usize).min(w_in.saturating_sub(1));

            let fy = y_clamped - y_floor;
            let fx = x_clamped - x_floor;

            let w00 = (1.0 - fy) * (1.0 - fx);
            let w01 = (1.0 - fy) * fx;
            let w10 = fy * (1.0 - fx);
            let w11 = fy * fx;

            let get_pixel =
                |h: usize, w: usize| -> i64 { field_to_i64(inputs[data_offset + h * w_in + w]) };

            let v00 = get_pixel(y_f, x_f) as f64;
            let v01 = get_pixel(y_f, x_c) as f64;
            let v10 = get_pixel(y_c, x_f) as f64;
            let v11 = get_pixel(y_c, x_c) as f64;

            let interp = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
            interp.round() as i64
        }
        _ => 0,
    };

    if result < 0 {
        let mag = U256::from(result.unsigned_abs());
        outputs[0] = F::from_u256(F::MODULUS - mag);
    } else {
        outputs[0] = F::from_u256(U256::from(result as u64));
    }

    Ok(())
}
