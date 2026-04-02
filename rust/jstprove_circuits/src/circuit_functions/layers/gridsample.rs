// ONNX `GridSample` layer for ZK circuits.
//
// # ZK approach
// GridSample supports two interpolation modes:
//
// **Nearest**: Output element (n,c,h,w) is the input pixel nearest to the
// grid-specified coordinate. For compile-time-constant grids the sampling
// index for each output position is precomputed during `build()` and applied
// as a pure structural passthrough in `apply()` — no hint, no range check.
//
// **Bilinear**: Each output element is a convex combination of 4 input corner
// pixels (2×2 neighbourhood). The weighted sum is computed via
// `api.new_hint("jstprove.gridsample_hint", corners ++ weights ++ scale, 1)`
// and bounded by a LogUp range check.
//
// **Bicubic**: Rejected with a clear error — out of scope.
//
// # Coordinate transformation
// Grid values are normalised coordinates in [-1, 1]:
//   align_corners=1:  x_in = (x + 1) / 2 * (W_in - 1)
//   align_corners=0:  x_in = (x + 1) / 2 * W_in - 0.5
//
// # Padding modes
// Supported: zeros (default), border, reflection.
// - zeros:      out-of-bounds pixels contribute 0 (nearest → api.constant(0);
//               bilinear → corner weight forced to 0).
// - border:     clamp coordinate to [0, size-1].
// - reflection: reflect coordinate at the boundary.
//
// # Constant grid requirement
// The grid tensor MUST be a compile-time constant (model initializer).
// Dynamic grids are not supported; `build()` returns an error if the grid
// tensor cannot be found in `layer_context.w_and_b_map`.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    hints::gridsample::GRIDSAMPLE_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default, get_w_or_b},
    },
};

// -------- Internal types --------

/// Specification for one bilinear output element: 4 corners and their weights.
struct BilinearOutputSpec {
    /// Flat indices into the input tensor X for each of the 4 bilinear corners.
    corner_flat_indices: Vec<usize>,
    /// Quantised weights at α¹ for each corner (0 for OOB zeros corners).
    weights_q: Vec<i64>,
}

enum GridSampleInterp {
    /// Nearest-neighbour: `index_map[out_flat]` = `Some(in_flat)` (in-bounds)
    /// or `None` (zeros-padding out-of-bounds).
    Nearest { index_map: Vec<Option<usize>> },
    /// Bilinear: per-element 4-corner specs.
    Bilinear {
        per_output: Vec<BilinearOutputSpec>,
        n_bits: usize,
        scaling: u64,
    },
}

// -------- Struct --------

pub struct GridSampleLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    interp: GridSampleInterp,
}

// -------- Coordinate utilities --------

/// Convert a normalised grid coordinate in [-1, 1] to a continuous pixel
/// coordinate in input space.
#[allow(clippy::cast_precision_loss, clippy::manual_midpoint)]
fn unnormalize(norm: f64, size: usize, align_corners: bool) -> f64 {
    if align_corners {
        (norm + 1.0) / 2.0 * (size.saturating_sub(1) as f64)
    } else {
        (norm + 1.0) / 2.0 * size as f64 - 0.5
    }
}

/// Apply `reflection` padding to a continuous pixel coordinate.
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
    // Map relative to lo, take positive mod period, then fold.
    let mut rel = (x - lo).rem_euclid(period);
    if rel > range {
        rel = period - rel;
    }
    (rel + lo).clamp(0.0, (size - 1) as f64)
}

/// Resolve a pixel coordinate under the given padding mode; returns `None`
/// for out-of-bounds positions under `zeros` padding.
#[allow(clippy::cast_precision_loss)]
fn apply_padding_f(x: f64, size: usize, padding_mode: &str, align_corners: bool) -> Option<f64> {
    match padding_mode {
        "zeros" => {
            if x < -0.5 || x > (size as f64) - 0.5 {
                None
            } else {
                Some(x.clamp(0.0, (size.saturating_sub(1)) as f64))
            }
        }
        "reflection" => Some(reflect_coord(x, size, align_corners)),
        _ => Some(x.clamp(0.0, (size.saturating_sub(1)) as f64)),
    }
}

/// Nearest-neighbour rounding: round to nearest integer (ties go to +∞).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn nearest_px(x: f64, size: usize) -> usize {
    (x + 0.5)
        .floor()
        .clamp(0.0, (size.saturating_sub(1)) as f64) as usize
}

/// Returns `(floor_idx, ceil_idx, weight_floor, weight_ceil)` for a continuous
/// pixel coordinate `x` in a dimension of size `size`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn bilinear_corners_1d(x: f64, size: usize) -> (usize, usize, f64, f64) {
    let x_clamped = x.clamp(0.0, (size.saturating_sub(1)) as f64);
    let floor_f = x_clamped.floor();
    let floor_idx = floor_f as usize;
    let ceil_idx = (x_clamped.ceil() as usize).min(size.saturating_sub(1));
    let frac = x_clamped - floor_f;
    (floor_idx, ceil_idx, 1.0 - frac, frac)
}

// -------- Build helpers --------

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn build_nearest_index_map(
    grid_flat: &[i64],
    alpha_f: f64,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    padding_mode: &str,
    align_corners: bool,
) -> Vec<Option<usize>> {
    let total_out = n * c * h_out * w_out;
    let mut index_map = Vec::with_capacity(total_out);

    for bn in 0..n {
        for bc in 0..c {
            for bh in 0..h_out {
                for bw in 0..w_out {
                    // Grid stores (x_norm, y_norm) at [bn, bh, bw, 0/1].
                    let sp = bn * h_out * w_out + bh * w_out + bw;
                    let x_norm = grid_flat[sp * 2] as f64 / alpha_f;
                    let y_norm = grid_flat[sp * 2 + 1] as f64 / alpha_f;

                    let x_in = unnormalize(x_norm, w_in, align_corners);
                    let y_in = unnormalize(y_norm, h_in, align_corners);

                    let entry = apply_padding_f(y_in, h_in, padding_mode, align_corners)
                        .and_then(|y_p| {
                            apply_padding_f(x_in, w_in, padding_mode, align_corners)
                                .map(|x_p| (nearest_px(y_p, h_in), nearest_px(x_p, w_in)))
                        })
                        .map(|(h_px, w_px)| {
                            bn * c * h_in * w_in + bc * h_in * w_in + h_px * w_in + w_px
                        });

                    index_map.push(entry);
                }
            }
        }
    }

    index_map
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::cast_precision_loss
)]
fn build_bilinear_per_output(
    grid_flat: &[i64],
    alpha_f: f64,
    scaling: u64,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    padding_mode: &str,
    align_corners: bool,
) -> Vec<BilinearOutputSpec> {
    let total_out = n * c * h_out * w_out;
    let mut per_output = Vec::with_capacity(total_out);

    for bn in 0..n {
        for bc in 0..c {
            for bh in 0..h_out {
                for bw in 0..w_out {
                    let sp = bn * h_out * w_out + bh * w_out + bw;
                    let x_norm = grid_flat[sp * 2] as f64 / alpha_f;
                    let y_norm = grid_flat[sp * 2 + 1] as f64 / alpha_f;

                    let x_in = unnormalize(x_norm, w_in, align_corners);
                    let y_in = unnormalize(y_norm, h_in, align_corners);

                    // 2-D bilinear: 4 corners in (H, W).
                    let (h_f, h_c, wh_f, wh_c) = bilinear_corners_1d(y_in, h_in);
                    let (w_f, w_c, ww_f, ww_c) = bilinear_corners_1d(x_in, w_in);

                    // Corner ordering: (h_f,w_f), (h_f,w_c), (h_c,w_f), (h_c,w_c).
                    let corners_hw = [(h_f, w_f), (h_f, w_c), (h_c, w_f), (h_c, w_c)];
                    let weights_f = [wh_f * ww_f, wh_f * ww_c, wh_c * ww_f, wh_c * ww_c];

                    let mut corner_flat_indices = Vec::with_capacity(4);
                    let mut weights_q = Vec::with_capacity(4);

                    for (i, &(ch, cw)) in corners_hw.iter().enumerate() {
                        // For "zeros" padding, check if the unnormalized grid coordinate
                        // is OOB. bilinear_corners_1d already clamps h_f/h_c/w_f/w_c to
                        // valid ranges, so the per-corner bounds checks are redundant.
                        let in_bounds = if padding_mode == "zeros" {
                            apply_padding_f(y_in, h_in, "zeros", align_corners).is_some()
                                && apply_padding_f(x_in, w_in, "zeros", align_corners).is_some()
                        } else {
                            true
                        };

                        let flat = bn * c * h_in * w_in + bc * h_in * w_in + ch * w_in + cw;
                        corner_flat_indices.push(flat);

                        let wq = if in_bounds {
                            (weights_f[i] * scaling as f64).round() as i64
                        } else {
                            0
                        };
                        weights_q.push(wq);
                    }

                    per_output.push(BilinearOutputSpec {
                        corner_flat_indices,
                        weights_q,
                    });
                }
            }
        }
    }

    per_output
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GridSampleLayer {
    #[allow(clippy::cast_sign_loss)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::GridSample, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::GridSample,
            name: x_name.clone(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::GridSample,
            msg: "input tensor X is not contiguous".to_string(),
        })?;

        let out_flat = match &self.interp {
            GridSampleInterp::Nearest { index_map } => {
                // Structural passthrough: select by precomputed index or inject zero.
                let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
                index_map
                    .iter()
                    .map(|entry| match entry {
                        Some(idx) => data_flat[*idx],
                        None => zero_var,
                    })
                    .collect::<Vec<_>>()
            }

            GridSampleInterp::Bilinear {
                per_output,
                n_bits,
                scaling,
            } => {
                let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(*scaling)));

                let mut out_vars = Vec::with_capacity(per_output.len());
                for spec in per_output {
                    // Build hint inputs: [x_corners..., w_corners..., scale]
                    let mut hint_inputs: Vec<Variable> = spec
                        .corner_flat_indices
                        .iter()
                        .map(|&idx| data_flat[idx])
                        .collect();

                    let w_vars: Vec<Variable> = spec
                        .weights_q
                        .iter()
                        .map(|&w| {
                            let fv = if w >= 0 {
                                CircuitField::<C>::from_u256(U256::from(w as u64))
                            } else {
                                let mag = U256::from(w.unsigned_abs());
                                CircuitField::<C>::from_u256(CircuitField::<C>::MODULUS - mag)
                            };
                            api.constant(fv)
                        })
                        .collect();
                    hint_inputs.extend_from_slice(&w_vars);
                    hint_inputs.push(scale_var);

                    let hint_out = api.new_hint(GRIDSAMPLE_HINT_KEY, &hint_inputs, 1);
                    let y = hint_out[0];
                    logup_ctx.range_check::<C, Builder>(api, y, *n_bits)?;
                    out_vars.push(y);
                }

                out_vars
            }
        };

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::GridSample,
                    msg: format!("gridsample output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // Guard: exactly two inputs (X and grid).
        if layer.inputs.len() < 2 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::GridSample,
                param: format!(
                    "expected at least 2 inputs (X, grid), got {}",
                    layer.inputs.len()
                ),
            }
            .into());
        }

        // Guard: exactly one output.
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::GridSample,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GridSample,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        // Validate output is 4-D: [N, C, H_out, W_out].
        if output_shape.len() != 4 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::GridSample,
                msg: format!(
                    "expected 4-D output [N,C,H_out,W_out], got {}D",
                    output_shape.len()
                ),
            }
            .into());
        }

        let x_name = &layer.inputs[0];
        let input_shape = layer_context
            .shapes_map
            .get(x_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GridSample,
                msg: format!("missing input shape for X '{x_name}'"),
            })?
            .clone();

        if input_shape.len() != 4 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::GridSample,
                msg: format!("X must be 4-D [N,C,H_in,W_in], got {}D", input_shape.len()),
            }
            .into());
        }

        let [n, c, h_in, w_in] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        let [h_out, w_out] = [output_shape[2], output_shape[3]];

        // Read the constant grid tensor (quantised at α¹ by the quantizer).
        let grid_name = &layer.inputs[1];
        let grid_array: ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, grid_name).map_err(|e| LayerError::Other {
                layer: LayerKind::GridSample,
                msg: format!(
                    "failed to read grid tensor '{grid_name}': {e}; \
                        GridSample requires a compile-time constant (initializer) grid"
                ),
            })?;

        let grid_flat = grid_array
            .as_slice()
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GridSample,
                msg: "grid tensor is not contiguous".to_string(),
            })?;

        // Compute alpha from scale_exponent and use it to decode quantised grid.
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::GridSample,
                msg: format!(
                    "scale_exponent {} too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;
        let alpha_f = scaling as f64;

        // Read string attributes.
        let params = extract_params(layer).ok();

        let mode: String = params
            .as_ref()
            .and_then(|p| {
                get_param_or_default(&layer.name, "mode", p, Some(&"bilinear".to_string())).ok()
            })
            .unwrap_or_else(|| "bilinear".to_string());

        let padding_mode: String = params
            .as_ref()
            .and_then(|p| {
                get_param_or_default(&layer.name, "padding_mode", p, Some(&"zeros".to_string()))
                    .ok()
            })
            .unwrap_or_else(|| "zeros".to_string());

        let align_corners_i: i64 = params
            .as_ref()
            .and_then(|p| get_param_or_default(&layer.name, "align_corners", p, Some(&0i64)).ok())
            .unwrap_or(0);
        let align_corners = align_corners_i != 0;

        if mode == "bicubic" {
            return Err(LayerError::Other {
                layer: LayerKind::GridSample,
                msg: "bicubic interpolation mode is not supported in ZK circuits".to_string(),
            }
            .into());
        }

        let interp = if mode == "nearest" {
            let index_map = build_nearest_index_map(
                grid_flat,
                alpha_f,
                n,
                c,
                h_in,
                w_in,
                h_out,
                w_out,
                &padding_mode,
                align_corners,
            );
            GridSampleInterp::Nearest { index_map }
        } else {
            // bilinear (default)
            let n_bits = layer_context.n_bits_for(&layer.name);
            let per_output = build_bilinear_per_output(
                grid_flat,
                alpha_f,
                scaling,
                n,
                c,
                h_in,
                w_in,
                h_out,
                w_out,
                &padding_mode,
                align_corners,
            );
            GridSampleInterp::Bilinear {
                per_output,
                n_bits,
                scaling,
            }
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            interp,
        }))
    }
}
