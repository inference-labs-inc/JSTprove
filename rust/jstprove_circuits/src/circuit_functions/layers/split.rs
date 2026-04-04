// ONNX `Split` layer for ZK circuits.
//
// # ZK approach
// Split divides a tensor along an axis into multiple sub-tensors, one per
// output.  In the fixed-point quantised integer domain this is purely
// structural: each output element at flat position `k` is the input element
// at a precomputed flat position stored in that output's `index_map`.
//
// A compile-time `index_map` is built for every output during `build()`.
// At circuit evaluation time `apply_multi()` selects variables by index — no
// arithmetic, no hint, no range check.
//
// `apply()` is not reachable because the execution loop in `onnx.rs` always
// calls `apply_multi()`.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_param_or_default,
};

// -------- Struct --------

pub struct SplitLayer {
    input_name: String,
    /// Per-output: (output_name, index_map, output_shape).
    outputs: Vec<(String, Vec<usize>, Vec<usize>)>,
}

// -------- Index helpers --------

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

fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += coords[i] * stride;
        stride *= shape[i];
    }
    flat
}

/// Build the compile-time index map for one Split output slice.
///
/// `offset` is the starting index along `axis` in the input tensor.
fn build_split_index_map(
    input_shape: &[usize],
    axis: usize,
    offset: usize,
    output_shape: &[usize],
) -> Vec<usize> {
    let total_out: usize = output_shape.iter().product();
    let mut index_map = Vec::with_capacity(total_out);
    for out_flat in 0..total_out {
        let mut in_coords = unravel(out_flat, output_shape);
        in_coords[axis] += offset;
        index_map.push(ravel(&in_coords, input_shape));
    }
    index_map
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SplitLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // The execution loop always calls apply_multi; apply is unreachable.
        unreachable!("SplitLayer::apply called directly; use apply_multi")
    }

    fn apply_multi(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<Vec<(String, ArrayD<Variable>)>, CircuitError> {
        let x = input
            .get(&self.input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Split,
                name: self.input_name.clone(),
            })?;

        let data_flat = x.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Split,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        let mut results = Vec::with_capacity(self.outputs.len());
        for (name, index_map, shape) in &self.outputs {
            let out_flat: Vec<Variable> = index_map.iter().map(|&idx| data_flat[idx]).collect();
            let out_array = ArrayD::from_shape_vec(IxDyn(shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Split,
                    msg: format!("split output reshape failed: {e}"),
                }
            })?;
            results.push((name.clone(), out_array));
        }
        Ok(results)
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
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
                layer: LayerKind::Split,
                param: "data input".to_string(),
            })?
            .clone();

        let input_shape = layer_context
            .shapes_map
            .get(&input_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Split,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let rank = input_shape.len();

        // Read axis attribute (default 0), normalise negative values.
        let axis_raw: i64 = layer
            .params
            .as_ref()
            .and_then(|p| get_param_or_default::<i64>(&layer.name, "axis", p, Some(&0i64)).ok())
            .unwrap_or(0);

        let axis = if axis_raw < 0 {
            let a = rank as i64 + axis_raw;
            if a < 0 {
                return Err(LayerError::Other {
                    layer: LayerKind::Split,
                    msg: format!("axis {axis_raw} out of range for rank {rank}"),
                }
                .into());
            }
            a as usize
        } else {
            axis_raw as usize
        };

        if axis >= rank {
            return Err(LayerError::Other {
                layer: LayerKind::Split,
                msg: format!("axis {axis} >= rank {rank}"),
            }
            .into());
        }

        if layer.outputs.is_empty() {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Split,
                param: "outputs".to_string(),
            }
            .into());
        }

        // Derive split sizes from the already-inferred output shapes.
        // This works regardless of whether sizes came from inputs[1], the
        // `split` attribute, or equal-split inference.
        let mut offset = 0usize;
        let mut outputs = Vec::with_capacity(layer.outputs.len());
        for out_name in &layer.outputs {
            let out_shape = layer_context
                .shapes_map
                .get(out_name)
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Split,
                    msg: format!("missing output shape for '{out_name}'"),
                })?
                .clone();

            let index_map = build_split_index_map(&input_shape, axis, offset, &out_shape);
            offset += out_shape[axis];
            outputs.push((out_name.clone(), index_map, out_shape));
        }

        Ok(Box::new(Self {
            input_name,
            outputs,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_index_map_equal_1d() {
        // Shape [4], axis=0, split into 2×[2]: offsets 0 and 2.
        let map0 = build_split_index_map(&[4], 0, 0, &[2]);
        let map1 = build_split_index_map(&[4], 0, 2, &[2]);
        assert_eq!(map0, vec![0, 1]);
        assert_eq!(map1, vec![2, 3]);
    }

    #[test]
    fn split_index_map_2d_axis1() {
        // Shape [2, 4], axis=1, split into [2,2] and [2,2].
        let input_shape = &[2, 4];
        let map0 = build_split_index_map(input_shape, 1, 0, &[2, 2]);
        let map1 = build_split_index_map(input_shape, 1, 2, &[2, 2]);
        // Row 0: cols 0,1 → flat 0,1; Row 1: cols 0,1 → flat 4,5
        assert_eq!(map0, vec![0, 1, 4, 5]);
        // Row 0: cols 2,3 → flat 2,3; Row 1: cols 2,3 → flat 6,7
        assert_eq!(map1, vec![2, 3, 6, 7]);
    }

    #[test]
    fn split_index_map_2d_unequal_axis1() {
        // Shape [1, 4], axis=1, split into [1,1] and [1,3].
        let input_shape = &[1, 4];
        let map0 = build_split_index_map(input_shape, 1, 0, &[1, 1]);
        let map1 = build_split_index_map(input_shape, 1, 1, &[1, 3]);
        assert_eq!(map0, vec![0]);
        assert_eq!(map1, vec![1, 2, 3]);
    }
}
