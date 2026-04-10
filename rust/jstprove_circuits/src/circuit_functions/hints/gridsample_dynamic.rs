use ethnum::U256;
use expander_compiler::field::FieldArith;
use expander_compiler::utils::error::Error;

use super::field_to_i64;

pub const GRIDSAMPLE_DYNAMIC_HINT_KEY: &str = "jstprove.gridsample_dynamic_hint";

const NUM_CORNERS: usize = 4;
pub const GRIDSAMPLE_DYNAMIC_OUTPUTS: usize = 1 + NUM_CORNERS + NUM_CORNERS;

fn i64_to_field<F: FieldArith>(v: i64) -> F {
    if v >= 0 {
        F::from_u256(U256::from(v.unsigned_abs()))
    } else {
        let mag = U256::from(v.unsigned_abs());
        F::from_u256(F::MODULUS - mag)
    }
}

#[allow(clippy::cast_precision_loss)]
fn reflect_coord(x: f64, size: usize, align_corners: bool) -> f64 {
    if size <= 1 {
        return 0.0;
    }
    let (lo, range) = if align_corners {
        (0.0f64, (size - 1) as f64)
    } else {
        (-0.5f64, size as f64)
    };
    let period = 2.0 * range;
    let mut rel = (x - lo).rem_euclid(period);
    if rel > range {
        rel = period - rel;
    }
    (rel + lo).clamp(0.0, (size - 1) as f64)
}

#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::manual_midpoint,
    clippy::missing_errors_doc
)]
pub fn gridsample_dynamic_hint<F: FieldArith>(
    inputs: &[F],
    outputs: &mut [F],
) -> Result<(), Error> {
    if outputs.len() != GRIDSAMPLE_DYNAMIC_OUTPUTS {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: expected {GRIDSAMPLE_DYNAMIC_OUTPUTS} outputs \
             (result + 4 corners + 4 weights), got {}",
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
    let max_scale = U256::from(i64::MAX as u64);
    if scale_u256 > max_scale || scale_u256 == U256::ZERO {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: scale {scale_u256} must be in [1, i64::MAX]"
        )));
    }
    let scale_u64 = scale_u256.as_u64();
    let alpha_f = scale_u64 as f64;

    let h_in_i64 = field_to_i64(inputs[1]);
    let w_in_i64 = field_to_i64(inputs[2]);
    if h_in_i64 <= 0 || w_in_i64 <= 0 {
        return Err(Error::UserError(format!(
            "gridsample_dynamic_hint: h_in={h_in_i64}, w_in={w_in_i64} must both be positive"
        )));
    }
    let h_in = h_in_i64 as usize;
    let w_in = w_in_i64 as usize;
    let align_corners = field_to_i64(inputs[3]) != 0;
    let padding_mode_code = field_to_i64(inputs[4]);

    let data_size = h_in.checked_mul(w_in).ok_or_else(|| {
        Error::UserError(format!(
            "gridsample_dynamic_hint: h_in * w_in overflow ({h_in} * {w_in})"
        ))
    })?;
    let expected_len = 5usize
        .checked_add(data_size)
        .and_then(|v| v.checked_add(2))
        .ok_or_else(|| {
            Error::UserError(format!(
                "gridsample_dynamic_hint: expected_len overflow for data_size={data_size}"
            ))
        })?;
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
            _ => Some(reflect_coord(coord, size, align_corners)),
        }
    };

    let x_padded = apply_pad(x_in, w_in);
    let y_padded = apply_pad(y_in, h_in);

    let zero_field = F::from_u256(U256::ZERO);

    match (y_padded, x_padded) {
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

            let w_f = [
                (1.0 - fy) * (1.0 - fx),
                (1.0 - fy) * fx,
                fy * (1.0 - fx),
                fy * fx,
            ];

            let get_pixel =
                |h: usize, w: usize| -> i64 { field_to_i64(inputs[data_offset + h * w_in + w]) };

            let corners_i64 = [
                get_pixel(y_f, x_f),
                get_pixel(y_f, x_c),
                get_pixel(y_c, x_f),
                get_pixel(y_c, x_c),
            ];

            let weights_q: [i64; NUM_CORNERS] = [
                (w_f[0] * scale_u64 as f64).round() as i64,
                (w_f[1] * scale_u64 as f64).round() as i64,
                (w_f[2] * scale_u64 as f64).round() as i64,
                (w_f[3] * scale_u64 as f64).round() as i64,
            ];

            let mut sum_i128: i128 = 0;
            for i in 0..NUM_CORNERS {
                sum_i128 += i128::from(corners_i64[i]) * i128::from(weights_q[i]);
            }

            let scale_i128 = i128::from(scale_u64);
            let half = scale_i128 / 2;
            let result: i64 = if sum_i128 >= 0 {
                let rounded = (sum_i128 + half) / scale_i128;
                rounded.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
            } else {
                let rounded = (sum_i128 - half) / scale_i128;
                rounded.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
            };

            outputs[0] = i64_to_field(result);
            for i in 0..NUM_CORNERS {
                outputs[1 + i] = i64_to_field(corners_i64[i]);
            }
            for i in 0..NUM_CORNERS {
                outputs[1 + NUM_CORNERS + i] = i64_to_field(weights_q[i]);
            }
        }
        _ => {
            for out in outputs.iter_mut() {
                *out = zero_field;
            }
        }
    }

    Ok(())
}
