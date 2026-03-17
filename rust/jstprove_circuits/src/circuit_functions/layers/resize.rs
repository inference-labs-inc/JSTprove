// ONNX `Resize` layer for ZK circuits.
//
// # ZK approach
// Resize supports two interpolation modes:
//
// **Nearest**: Output element i maps to input element j via a compile-time
// index map precomputed from the coordinate transformation formula. This is a
// purely structural operation (no hint, no range check) — elements are
// selected by indexing, analogous to Gather.
//
// **Linear**: Each output element is a convex combination of up to 2^k input
// corner values (k = number of dimensions being resized). The weighted sum is
// computed via `api.new_hint("jstprove.resize_hint", corners ++ weights ++ scale, 1)`
// and bounded by a LogUp range check.
//
// **Cubic**: Rejected with a clear error — out of scope.
//
// # Coordinate transformation modes
// Supported: half_pixel (default), asymmetric, align_corners,
// pytorch_half_pixel, tf_half_pixel_for_nn.
//
// # Nearest rounding modes
// Supported: round_prefer_floor (default), round_prefer_ceil, floor, ceil.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    hints::resize::RESIZE_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
};

// -------- Internal types --------

/// Compile-time specification for one linear-mode output element.
struct LinearOutputSpec {
    /// Flat indices into the input tensor for each interpolation corner.
    corner_flat_indices: Vec<usize>,
    /// Quantised weights for each corner (at α¹ = scale).
    weights_q: Vec<i64>,
}

enum ResizeInterp {
    /// Precomputed flat index map: `index_map[out_flat] = in_flat`.
    Nearest { index_map: Vec<usize> },
    /// Per-output-element corner specs for linear interpolation.
    Linear {
        per_output: Vec<LinearOutputSpec>,
        n_bits: usize,
        scaling: u64,
    },
}

// -------- Struct --------

pub struct ResizeLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    interp: ResizeInterp,
}

// -------- Coordinate utilities --------

/// Map output coordinate to continuous input coordinate using the specified
/// coordinate transformation mode.
fn output_to_input_coord(out_idx: usize, in_size: usize, out_size: usize, mode: &str) -> f64 {
    let o = out_idx as f64;
    let in_f = in_size as f64;
    let out_f = out_size as f64;
    match mode {
        "half_pixel" => (o + 0.5) * in_f / out_f - 0.5,
        "asymmetric" => o * in_f / out_f,
        "align_corners" => {
            if out_size <= 1 {
                0.0
            } else {
                o * (in_f - 1.0) / (out_f - 1.0)
            }
        }
        "pytorch_half_pixel" => {
            if out_size > 1 {
                (o + 0.5) * in_f / out_f - 0.5
            } else {
                0.0
            }
        }
        "tf_half_pixel_for_nn" => (o + 0.5) * in_f / out_f,
        _ => unreachable!(
            "coordinate_transformation_mode '{mode}' should have been rejected in build()"
        ),
    }
}

/// Apply nearest-neighbour rounding to a continuous coordinate.
fn apply_nearest_rounding(x: f64, in_size: usize, nearest_mode: &str) -> Result<usize, LayerError> {
    if in_size == 0 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Resize,
            msg: "apply_nearest_rounding: source axis has size 0".to_string(),
        });
    }
    let idx: i64 = match nearest_mode {
        "round_prefer_floor" => {
            let floor = x.floor();
            if x - floor == 0.5 {
                floor as i64
            } else {
                x.round() as i64
            }
        }
        "round_prefer_ceil" => {
            let floor = x.floor();
            if x - floor == 0.5 {
                x.ceil() as i64
            } else {
                x.round() as i64
            }
        }
        "floor" => x.floor() as i64,
        "ceil" => x.ceil() as i64,
        _ => unreachable!("nearest_mode '{nearest_mode}' should have been rejected in build()"),
    };
    Ok(idx.clamp(0, in_size as i64 - 1) as usize)
}

/// Return (floor_idx, ceil_idx, weight_floor, weight_ceil) for a continuous
/// coordinate `x` into a dimension of size `in_size`.
fn linear_corners(x: f64, in_size: usize) -> Result<(usize, usize, f64, f64), LayerError> {
    if in_size == 0 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Resize,
            msg: "linear_corners: source axis has size 0".to_string(),
        });
    }
    let x_clamped = x.clamp(0.0, (in_size.saturating_sub(1)) as f64);
    let floor_f = x_clamped.floor();
    let ceil_f = x_clamped.ceil();
    let floor_idx = floor_f as usize;
    let ceil_idx = (ceil_f as usize).min(in_size - 1);
    let frac = x_clamped - floor_f;
    Ok((floor_idx, ceil_idx, 1.0 - frac, frac))
}

/// Convert a flat index to per-dimension coordinates (C-order / row-major).
fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            coords[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
    coords
}

/// Convert per-dimension coordinates to a flat index (C-order / row-major).
fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += coords[i] * stride;
        stride *= shape[i];
    }
    flat
}

// -------- Build helpers --------

fn build_nearest_index_map(
    input_shape: &[usize],
    output_shape: &[usize],
    coord_mode: &str,
    nearest_mode: &str,
) -> Result<Vec<usize>, LayerError> {
    let total_out: usize = output_shape.iter().product();
    let rank = output_shape.len();
    let mut index_map = Vec::with_capacity(total_out);

    for out_flat in 0..total_out {
        let out_coords = unravel_index(out_flat, output_shape);
        let mut in_coords = vec![0usize; rank];
        for d in 0..rank {
            let x =
                output_to_input_coord(out_coords[d], input_shape[d], output_shape[d], coord_mode);
            in_coords[d] = apply_nearest_rounding(x, input_shape[d], nearest_mode)?;
        }
        index_map.push(ravel_index(&in_coords, input_shape));
    }

    Ok(index_map)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn build_linear_per_output(
    input_shape: &[usize],
    output_shape: &[usize],
    coord_mode: &str,
    scaling: u64,
) -> Result<Vec<LinearOutputSpec>, LayerError> {
    let rank = output_shape.len();
    // Dimensions where the size changes — these require interpolation.
    let resize_dims: Vec<usize> = (0..rank)
        .filter(|&d| input_shape[d] != output_shape[d])
        .collect();
    let n_corners = 1usize << resize_dims.len();
    let total_out: usize = output_shape.iter().product();
    let mut per_output = Vec::with_capacity(total_out);

    for out_flat in 0..total_out {
        let out_coords = unravel_index(out_flat, output_shape);

        // Per resize-dim: (floor_idx, ceil_idx, w_floor, w_ceil).
        let dim_info: Vec<(usize, usize, f64, f64)> = resize_dims
            .iter()
            .map(|&d| {
                let x = output_to_input_coord(
                    out_coords[d],
                    input_shape[d],
                    output_shape[d],
                    coord_mode,
                );
                linear_corners(x, input_shape[d])
            })
            .collect::<Result<Vec<_>, LayerError>>()?;

        let mut corner_flat_indices = Vec::with_capacity(n_corners);
        let mut corner_weights_f = Vec::with_capacity(n_corners);

        for mask in 0..n_corners {
            let mut in_coords = out_coords.clone();
            let mut w = 1.0f64;
            for (i, &d) in resize_dims.iter().enumerate() {
                let (floor_idx, ceil_idx, w_floor, w_ceil) = dim_info[i];
                if mask & (1 << i) == 0 {
                    in_coords[d] = floor_idx;
                    w *= w_floor;
                } else {
                    in_coords[d] = ceil_idx;
                    w *= w_ceil;
                }
            }
            corner_flat_indices.push(ravel_index(&in_coords, input_shape));
            corner_weights_f.push(w);
        }

        let weights_q: Vec<i64> = corner_weights_f
            .iter()
            .map(|&w| (w * scaling as f64).round() as i64)
            .collect();

        per_output.push(LinearOutputSpec {
            corner_flat_indices,
            weights_q,
        });
    }

    Ok(per_output)
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ResizeLayer {
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Resize, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Resize,
            name: x_name.to_string(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Resize,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        let out_flat = match &self.interp {
            ResizeInterp::Nearest { index_map } => {
                // Structural passthrough: select elements by precomputed index.
                index_map
                    .iter()
                    .map(|&idx| data_flat[idx])
                    .collect::<Vec<_>>()
            }
            ResizeInterp::Linear {
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
                            // Bilinear weights are derived from `(w_real * scale).round()` where
                            // w_real ∈ [0, 1], so weights_q is always non-negative.
                            debug_assert!(w >= 0, "bilinear weight must be non-negative, got {w}");
                            let fv = CircuitField::<C>::from_u256(U256::from(w as u64));
                            api.constant(fv)
                        })
                        .collect();
                    hint_inputs.extend_from_slice(&w_vars);
                    hint_inputs.push(scale_var);

                    let hint_out = api.new_hint(RESIZE_HINT_KEY, &hint_inputs, 1);
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
                    layer: LayerKind::Resize,
                    msg: format!("resize output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // Guard: at least one input (data tensor X).
        if layer.inputs.is_empty() {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Resize,
                param: "data input X".to_string(),
            }
            .into());
        }

        // Guard: exactly one output.
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Resize,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Resize,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let input_name = layer.inputs.first().unwrap();
        let input_shape = layer_context
            .shapes_map
            .get(input_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Resize,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        if input_shape.len() != output_shape.len() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Resize,
                msg: format!(
                    "input rank {} != output rank {}",
                    input_shape.len(),
                    output_shape.len()
                ),
            }
            .into());
        }

        // Read string attributes (defaults per ONNX spec).
        let params = extract_params(layer).ok();
        let default_mode = "nearest".to_string();
        let default_coord = "half_pixel".to_string();
        let default_nearest = "round_prefer_floor".to_string();

        let mode: String = params
            .as_ref()
            .and_then(|p| get_param_or_default(&layer.name, "mode", p, Some(&default_mode)).ok())
            .unwrap_or(default_mode);

        let coord_mode: String = params
            .as_ref()
            .and_then(|p| {
                get_param_or_default(
                    &layer.name,
                    "coordinate_transformation_mode",
                    p,
                    Some(&default_coord),
                )
                .ok()
            })
            .unwrap_or(default_coord);

        let nearest_mode: String = params
            .as_ref()
            .and_then(|p| {
                get_param_or_default(&layer.name, "nearest_mode", p, Some(&default_nearest)).ok()
            })
            .unwrap_or(default_nearest);

        // Validate attribute values against the allowed ONNX sets.
        const VALID_MODES: &[&str] = &["nearest", "linear", "bilinear", "trilinear", "cubic"];
        if !VALID_MODES.contains(&mode.as_str()) {
            return Err(LayerError::Other {
                layer: LayerKind::Resize,
                msg: format!(
                    "layer '{}': Resize mode '{}' is not in the allowed set {:?}",
                    layer.name, mode, VALID_MODES
                ),
            }
            .into());
        }
        const VALID_COORD_MODES: &[&str] = &[
            "half_pixel",
            "asymmetric",
            "align_corners",
            "pytorch_half_pixel",
            "tf_half_pixel_for_nn",
        ];
        if !VALID_COORD_MODES.contains(&coord_mode.as_str()) {
            return Err(LayerError::Other {
                layer: LayerKind::Resize,
                msg: format!(
                    "layer '{}': Resize coordinate_transformation_mode '{}' is not in \
                     the allowed set {:?}",
                    layer.name, coord_mode, VALID_COORD_MODES
                ),
            }
            .into());
        }
        const VALID_NEAREST_MODES: &[&str] =
            &["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"];
        if !VALID_NEAREST_MODES.contains(&nearest_mode.as_str()) {
            return Err(LayerError::Other {
                layer: LayerKind::Resize,
                msg: format!(
                    "layer '{}': Resize nearest_mode '{}' is not in the allowed set {:?}",
                    layer.name, nearest_mode, VALID_NEAREST_MODES
                ),
            }
            .into());
        }

        if mode == "cubic" {
            return Err(LayerError::Other {
                layer: LayerKind::Resize,
                msg: "cubic interpolation mode is not supported in ZK circuits".to_string(),
            }
            .into());
        }

        let interp = if mode == "linear" || mode == "bilinear" || mode == "trilinear" {
            let scaling: u64 =
                1u64.checked_shl(circuit_params.scale_exponent)
                    .ok_or_else(|| LayerError::Other {
                        layer: LayerKind::Resize,
                        msg: format!(
                            "scale_exponent {} is too large to shift u64",
                            circuit_params.scale_exponent
                        ),
                    })?;
            let n_bits = layer_context.n_bits_for(&layer.name);
            let per_output =
                build_linear_per_output(&input_shape, &output_shape, &coord_mode, scaling)?;
            ResizeInterp::Linear {
                per_output,
                n_bits,
                scaling,
            }
        } else {
            // Default: nearest neighbour.
            let index_map =
                build_nearest_index_map(&input_shape, &output_shape, &coord_mode, &nearest_mode)?;
            ResizeInterp::Nearest { index_map }
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            interp,
        }))
    }
}
