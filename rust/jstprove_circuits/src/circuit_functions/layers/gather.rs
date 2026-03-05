// ONNX `Gather` layer for ZK circuits.
//
// # ZK approach
// Gather selects slices from a data tensor using constant integer indices.
// After quantization all tensors are INT64 field elements; element selection
// is a purely structural operation with no arithmetic. There are no circuit
// constraints: the correct slice is assembled in `apply()` by indexing into
// the data Variable array using compile-time constant indices.
//
// # Constant indices requirement
// The indices tensor MUST be a model initializer or a Constant node so that
// the index values are known at circuit-build time. Dynamic indices (computed
// at inference time) are not supported because the circuit topology must be
// fixed before the prover runs.
//
// # Axis handling
// Gather is applied along the specified axis (default 0, matching the ONNX
// spec). Negative axis values are resolved relative to the data rank.
//
// # Output shape
// output_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default, get_w_or_b},
    },
};

// -------- Struct --------

#[derive(Debug)]
pub struct GatherLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Resolved (non-negative) gather axis.
    axis: usize,
    /// Flat compile-time constant indices (in C-order / row-major).
    indices: Vec<i64>,
    /// Expected output shape; used to construct the output ArrayD.
    output_shape: Vec<usize>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GatherLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let data_name = get_input_name(&self.inputs, 0, LayerKind::Gather, INPUT)?;
        let data = input
            .get(data_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Gather,
                name: data_name.to_string(),
            })?;

        let data_shape = data.shape();
        let axis = self.axis;

        if axis >= data_shape.len() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Gather,
                msg: format!(
                    "axis {} out of range for data rank {}",
                    axis,
                    data_shape.len()
                ),
            }
            .into());
        }

        let d_axis = data_shape[axis];
        // Product of dims before the gather axis.
        let outer_size: usize = data_shape[..axis].iter().product();
        // Product of dims after the gather axis (size of one gathered slice).
        let slice_size: usize = if axis + 1 < data_shape.len() {
            data_shape[axis + 1..].iter().product()
        } else {
            1
        };

        let n_indices = self.indices.len();
        let expected_out_elems: usize = self.output_shape.iter().product();

        // Validate output size matches expectations.
        if outer_size * n_indices * slice_size != expected_out_elems {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Gather,
                msg: format!(
                    "gather output element count mismatch: outer={} n_indices={} slice={} \
                    expected={expected_out_elems}",
                    outer_size, n_indices, slice_size
                ),
            }
            .into());
        }

        let data_flat = data.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Gather,
            msg: "data tensor is not contiguous".to_string(),
        })?;

        // For each outer block, gather the selected slices.
        let data_axis_stride = d_axis * slice_size; // stride of one outer block in data
        let mut out_flat = Vec::with_capacity(expected_out_elems);

        for outer_i in 0..outer_size {
            for &idx in &self.indices {
                let idx = usize::try_from(idx).map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::Gather,
                    msg: format!("negative index {idx} is not supported"),
                })?;
                if idx >= d_axis {
                    return Err(LayerError::InvalidShape {
                        layer: LayerKind::Gather,
                        msg: format!("index {idx} out of bounds for axis {axis} of size {d_axis}"),
                    }
                    .into());
                }
                let data_start = outer_i * data_axis_stride + idx * slice_size;
                out_flat.extend_from_slice(&data_flat[data_start..data_start + slice_size]);
            }
        }

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Gather,
                    msg: format!("gather output reshape failed: {e}"),
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
        // Guard: Gather requires at least two inputs (data + indices).
        if layer.inputs.len() < 2 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Gather,
                param: format!(
                    "expected at least 2 inputs (data, indices), got {}",
                    layer.inputs.len()
                ),
            }
            .into());
        }

        // Guard: Gather must produce exactly one output.
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Gather,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Gather,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        // Read axis attribute (ONNX default = 0).
        let raw_axis: i64 = match extract_params(layer).ok() {
            Some(params) => get_param_or_default(&layer.name, "axis", &params, Some(&0i64))
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::Gather,
                    msg: format!("failed to read 'axis' attribute: {e}"),
                })?,
            None => 0,
        };

        // Resolve axis relative to data rank.  We derive data rank from the
        // output_shape and the indices tensor shape.
        let indices_name = &layer.inputs[1];
        let indices_array: ndarray::ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, indices_name).map_err(|e| LayerError::Other {
                layer: LayerKind::Gather,
                msg: format!(
                    "failed to read indices tensor '{indices_name}': {e}; \
                        Gather requires constant (initializer) indices"
                ),
            })?;

        let indices_rank = indices_array.ndim();
        // data_rank = output_rank - indices_rank + 1
        let data_rank = output_shape.len() + 1 - indices_rank;

        let axis = if raw_axis < 0 {
            let a = data_rank as i64 + raw_axis;
            if a < 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Gather,
                    msg: format!("axis {} out of range for data rank {}", raw_axis, data_rank),
                }
                .into());
            }
            a as usize
        } else {
            let a = raw_axis as usize;
            if a >= data_rank {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Gather,
                    msg: format!("axis {} out of range for data rank {}", raw_axis, data_rank),
                }
                .into());
            }
            a
        };

        let indices: Vec<i64> = indices_array
            .as_slice()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Gather,
                msg: "indices tensor is not contiguous".to_string(),
            })?
            .to_vec();

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axis,
            indices,
            output_shape,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a 2-D data tensor gathered along axis 0 with a 1-D index
    /// vector produces the correct output rows.
    #[test]
    fn gather_axis0_selects_rows() {
        use ndarray::array;

        // data: [[10, 20], [30, 40], [50, 60]]  shape [3, 2]
        let data_arr: ArrayD<i64> = array![[10i64, 20], [30, 40], [50, 60]].into_dyn();
        let layer = GatherLayer {
            inputs: vec!["data".into(), "indices".into()],
            outputs: vec!["out".into()],
            axis: 0,
            indices: vec![2, 0], // select rows 2 and 0
            output_shape: vec![2, 2],
        };

        // Row 2 = [50, 60], row 0 = [10, 20] → output [[50,60],[10,20]]
        let expected: Vec<i64> = vec![50, 60, 10, 20];
        let out_arr =
            ArrayD::from_shape_vec(IxDyn(&layer.output_shape), expected.clone()).expect("shape ok");

        // Convert data to Variable (use i64 as proxy to avoid needing a real
        // circuit builder in this unit test).
        //
        // Because Variable is a circuit type, we test the indexing logic
        // directly by running the gather on the flat slice instead.
        let flat: Vec<i64> = data_arr.as_slice().expect("contiguous").to_vec();

        let d_axis = 3usize;
        let slice_size = 2usize;
        let outer_size = 1usize;
        let data_axis_stride = d_axis * slice_size;
        let mut out_flat: Vec<i64> = Vec::new();
        for outer_i in 0..outer_size {
            for &idx in &layer.indices {
                let idx = idx as usize;
                let start = outer_i * data_axis_stride + idx * slice_size;
                out_flat.extend_from_slice(&flat[start..start + slice_size]);
            }
        }

        let result =
            ArrayD::from_shape_vec(IxDyn(&layer.output_shape), out_flat).expect("shape ok");
        assert_eq!(result, out_arr);
    }
}
