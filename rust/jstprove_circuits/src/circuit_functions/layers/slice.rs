// ONNX `Slice` layer for ZK circuits.
//
// # ZK approach
// Slice extracts a sub-tensor from the input by selecting ranges along
// specified axes.  Because the circuit works in fixed-point quantised integers
// and field elements carry no type or layout information, Slice is a purely
// structural operation: the output element at each flat position is the input
// element at a precomputed flat position.
//
// A compile-time index map `index_map[out_flat] = in_flat` is built during
// `build()` from the `starts`, `ends`, `axes`, and `steps` inputs (all must
// be compile-time constants).  At circuit evaluation time `apply()` simply
// selects from the input by index — no hint, no range check.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
    },
};

// -------- Struct --------

pub struct SliceLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Output shape after slicing.
    output_shape: Vec<usize>,
    /// `index_map[out_flat] = in_flat` — precomputed at build time.
    index_map: Vec<usize>,
}

// -------- Index helpers (same as Transpose) --------

/// Convert a flat index to per-dimension coordinates (C-order / row-major).
fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
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
fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += coords[i] * stride;
        stride *= shape[i];
    }
    flat
}

/// Build the compile-time index map for a Slice.
///
/// `index_map[out_flat] = in_flat` maps every output position back to the
/// corresponding input position given the slice parameters.
fn build_index_map(
    input_shape: &[usize],
    starts: &[i64],
    _ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    output_shape: &[usize],
) -> Vec<usize> {
    let rank = input_shape.len();
    let total_out: usize = output_shape.iter().product();

    // Per-axis slice start and step (resolved to non-negative usizes).
    let mut axis_start = vec![0usize; rank];
    let mut axis_step = vec![1usize; rank];
    for (i, &axis_raw) in axes.iter().enumerate() {
        let axis = if axis_raw < 0 {
            (rank as i64 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        let dim = input_shape[axis] as i64;
        let s = starts.get(i).copied().unwrap_or(0);
        let s = if s < 0 { dim + s } else { s };
        axis_start[axis] = s.clamp(0, dim) as usize;
        axis_step[axis] = steps.get(i).copied().unwrap_or(1).max(1) as usize;
    }

    let mut index_map = Vec::with_capacity(total_out);
    for out_flat in 0..total_out {
        let out_coords = unravel(out_flat, output_shape);
        let in_coords: Vec<usize> = (0..rank)
            .map(|ax| axis_start[ax] + out_coords[ax] * axis_step[ax])
            .collect();
        index_map.push(ravel(&in_coords, input_shape));
    }
    index_map
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SliceLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Slice, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Slice,
            name: x_name.to_string(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Slice,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        // Structural passthrough: select elements by the precomputed index map.
        let out_flat: Vec<Variable> = self.index_map.iter().map(|&idx| data_flat[idx]).collect();

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Slice,
                    msg: format!("slice output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Slice,
                param: "data input".to_string(),
            })?;

        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Slice,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Slice,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let output_shape = layer_context
            .shapes_map
            .get(output_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Slice,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        // Helper: read a required i64 weight tensor.
        let read_i64 = |name: &String, field: &str| -> Result<Vec<i64>, CircuitError> {
            let arr: ndarray::ArrayD<i64> =
                get_w_or_b(layer_context.w_and_b_map, name).map_err(|e| LayerError::Other {
                    layer: LayerKind::Slice,
                    msg: format!("failed to read {field} tensor '{name}': {e}"),
                })?;
            Ok(arr
                .as_slice()
                .ok_or_else(|| LayerError::Other {
                    layer: LayerKind::Slice,
                    msg: format!("{field} tensor '{name}' is not contiguous"),
                })?
                .to_vec())
        };

        // starts — required, input[1]
        let starts_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Slice,
                param: "starts input".to_string(),
            })?;
        let starts = read_i64(starts_name, "starts")?;

        // ends — required, input[2]
        let ends_name = layer
            .inputs
            .get(2)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Slice,
                param: "ends input".to_string(),
            })?;
        let ends = read_i64(ends_name, "ends")?;

        // axes — optional, input[3]; default [0, 1, ..., len(starts)-1]
        let axes: Vec<i64> = layer
            .inputs
            .get(3)
            .filter(|n| !n.is_empty())
            .map(|n| read_i64(n, "axes"))
            .transpose()?
            .unwrap_or_else(|| (0..starts.len() as i64).collect());

        // steps — optional, input[4]; default all 1s
        let steps: Vec<i64> = layer
            .inputs
            .get(4)
            .filter(|n| !n.is_empty())
            .map(|n| read_i64(n, "steps"))
            .transpose()?
            .unwrap_or_else(|| vec![1i64; starts.len()]);

        let index_map = build_index_map(&input_shape, &starts, &ends, &axes, &steps, &output_shape);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            index_map,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_map_1d_basic() {
        // Shape [10], slice [2:7] → shape [5], elements at 2,3,4,5,6
        let map = build_index_map(&[10], &[2], &[7], &[0], &[1], &[5]);
        assert_eq!(map, vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn index_map_1d_step2() {
        // Shape [8], slice [0:8:2] → shape [4], elements at 0,2,4,6
        let map = build_index_map(&[8], &[0], &[8], &[0], &[2], &[4]);
        assert_eq!(map, vec![0, 2, 4, 6]);
    }

    #[test]
    fn index_map_2d_axis1() {
        // Shape [3, 4], slice axis=1 [1:3] → shape [3, 2]
        // Row 0: cols 1,2 → flat 1,2 ; Row 1: 5,6 ; Row 2: 9,10
        let map = build_index_map(&[3, 4], &[1], &[3], &[1], &[1], &[3, 2]);
        assert_eq!(map, vec![1, 2, 5, 6, 9, 10]);
    }

    #[test]
    fn index_map_identity() {
        // Full slice is identity
        let map = build_index_map(&[2, 3], &[0, 0], &[2, 3], &[0, 1], &[1, 1], &[2, 3]);
        let expected: Vec<usize> = (0..6).collect();
        assert_eq!(map, expected);
    }

    #[test]
    fn index_map_negative_start() {
        // Shape [5], slice [-3:] → elements 2,3,4 → shape [3]
        let map = build_index_map(&[5], &[-3], &[5], &[0], &[1], &[3]);
        assert_eq!(map, vec![2, 3, 4]);
    }
}
