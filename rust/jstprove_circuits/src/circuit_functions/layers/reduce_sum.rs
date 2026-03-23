// ONNX `ReduceSum` layer for ZK circuits.
//
// # ZK approach
// ReduceSum computes the sum along specified axes. Unlike ReduceMean,
// addition IS expressible as a low-degree polynomial, so we implement this
// using direct circuit operations (`api.add`).
//
// # Supported attributes
// - `axes`     (default: all axes): axes along which to reduce.
// - `keepdims` (default 1): if 1, keep reduced dims as size 1; if 0, remove them.
// - `noop_with_empty_axes` (default 0): if 1 and axes is empty, return input unchanged.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

// -------- Struct --------

#[derive(Debug)]
pub struct ReduceSumLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    axes: Vec<usize>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReduceSumLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = self
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ReduceSum,
                param: "input tensor".to_string(),
            })?;

        let x_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ReduceSum,
                name: input_name.clone(),
            })?;

        let rank = self.input_shape.len();

        // Handle noop case.
        if self.noop_with_empty_axes && self.axes.is_empty() {
            return Ok((self.outputs.clone(), x_input.clone()));
        }

        let input_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::ReduceSum,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        let total_out: usize = self.output_shape.iter().product();
        let mut out_storage: Vec<Variable> = Vec::with_capacity(total_out);

        // For each output position, sum the corresponding input elements.
        for out_flat in 0..total_out {
            // Compute output coordinates.
            let out_coords = unravel(out_flat, &self.output_shape);

            // Find all input positions that map to this output.
            // We iterate over the reduction axes.
            let input_indices = collect_reduction_inputs(
                &out_coords,
                &self.axes,
                self.keepdims,
                &self.input_shape,
                rank,
            );

            // Sum all contributing input variables.
            let mut acc = input_flat[input_indices[0]];
            for &idx in &input_indices[1..] {
                acc = api.add(acc, input_flat[idx]);
            }
            out_storage.push(acc);
        }

        let result =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_storage).map_err(|_| {
                LayerError::InvalidShape {
                    layer: LayerKind::ReduceSum,
                    msg: format!(
                        "ReduceSum: cannot reshape result into shape {:?}",
                        self.output_shape
                    ),
                }
            })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
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
                layer: LayerKind::ReduceSum,
                param: "input tensor".to_string(),
            })?;
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ReduceSum,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ReduceSum,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ReduceSum,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let rank = input_shape.len();

        let keepdims = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("keepdims".into()) {
                            if let rmpv::Value::Integer(i) = v {
                                i.as_i64()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
            .unwrap_or(1)
            != 0;

        let noop_with_empty_axes = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("noop_with_empty_axes".into()) {
                            if let rmpv::Value::Integer(i) = v {
                                i.as_i64()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
            .unwrap_or(0)
            != 0;

        // Parse axes from input[1] (opset 13+) or attribute.
        let axes: Vec<usize> = {
            let from_input: Option<Vec<i64>> = layer.inputs.get(1).and_then(|axes_name| {
                get_w_or_b(layer_context.w_and_b_map, axes_name)
                    .ok()
                    .map(|arr| arr.as_slice().unwrap_or(&[]).to_vec())
            });

            let raw: Option<Vec<i64>> = from_input.or_else(|| {
                layer.params.as_ref().and_then(|p| {
                    if let rmpv::Value::Map(m) = p {
                        m.iter().find_map(|(k, v)| {
                            if k == &rmpv::Value::String("axes".into()) {
                                match v {
                                    rmpv::Value::Array(arr) => arr
                                        .iter()
                                        .map(|x| {
                                            if let rmpv::Value::Integer(i) = x {
                                                i.as_i64()
                                            } else {
                                                None
                                            }
                                        })
                                        .collect(),
                                    rmpv::Value::Integer(i) => i.as_i64().map(|x| vec![x]),
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                })
            });

            match raw {
                Some(v) => {
                    let mut axes: Vec<usize> = v
                        .iter()
                        .map(|&a| {
                            let a = if a < 0 { a + rank as i64 } else { a };
                            if a < 0 || a as usize >= rank {
                                return Err(LayerError::InvalidShape {
                                    layer: LayerKind::ReduceSum,
                                    msg: format!("axis {a} out of range for rank {rank}"),
                                }
                                .into());
                            }
                            Ok(a as usize)
                        })
                        .collect::<Result<Vec<usize>, CircuitError>>()?;
                    axes.sort_unstable();
                    axes.dedup();
                    axes
                }
                None => {
                    if noop_with_empty_axes {
                        vec![]
                    } else {
                        (0..rank).collect()
                    }
                }
            }
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axes,
            keepdims,
            noop_with_empty_axes,
            input_shape,
            output_shape,
        }))
    }
}

// -------- Helpers --------

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

/// Given output coordinates, enumerate all input flat indices that contribute
/// to this output via the reduction axes.
pub(crate) fn collect_reduction_inputs(
    out_coords: &[usize],
    axes: &[usize],
    keepdims: bool,
    input_shape: &[usize],
    rank: usize,
) -> Vec<usize> {
    // Build the mapping from output coord axis to input coord axis.
    // If keepdims=true: out_coords[i] maps to input_shape[i] (reduced axes have coord 0).
    // If keepdims=false: non-reduced output dims are packed.
    let input_base: Vec<usize> = if keepdims {
        // out_coords has same rank as input; reduced dims are 0.
        out_coords.to_vec()
    } else {
        // out_coords has fewer dims; re-expand by inserting 0 for reduced axes.
        let mut in_coords = vec![0usize; rank];
        let mut out_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for ax in 0..rank {
            if axes.contains(&ax) {
                in_coords[ax] = 0; // will be iterated
            } else {
                in_coords[ax] = out_coords[out_idx];
                out_idx += 1;
            }
        }
        in_coords
    };

    // Iterate all combinations of reduction axes.
    let reduction_sizes: Vec<usize> = axes.iter().map(|&ax| input_shape[ax]).collect();
    let total_reduction: usize = reduction_sizes.iter().product();

    (0..total_reduction)
        .map(|r| {
            // Decode r into per-reduction-axis coordinates.
            let mut r_rem = r;
            let mut in_coords = input_base.clone();
            for (i, &ax) in axes.iter().enumerate().rev() {
                let sz = reduction_sizes[i];
                in_coords[ax] = r_rem % sz;
                r_rem /= sz;
            }
            ravel(&in_coords, input_shape)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unravel_ravel_roundtrip() {
        let shape = [2usize, 3, 4];
        for flat in 0..24 {
            let coords = unravel(flat, &shape);
            assert_eq!(ravel(&coords, &shape), flat);
        }
    }

    #[test]
    fn collect_reduction_inputs_reduce_all_axes_keepdims() {
        // input [2, 3], reduce axes [0,1], keepdims=true → output [1, 1]
        // Every input element (6 total) contributes to output[0,0].
        let input_shape = [2usize, 3];
        let axes = [0, 1];
        let out_coords = [0usize, 0];
        let indices = collect_reduction_inputs(&out_coords, &axes, true, &input_shape, 2);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn collect_reduction_inputs_reduce_axis0_keepdims() {
        // input [3, 4], reduce axis [0], keepdims=true → output [1, 4]
        // out[0, j] collects input[0,j], input[1,j], input[2,j]
        let input_shape = [3usize, 4];
        let axes = [0];
        for j in 0..4 {
            let out_coords = [0usize, j];
            let indices = collect_reduction_inputs(&out_coords, &axes, true, &input_shape, 2);
            let expected: Vec<usize> = (0..3).map(|i| i * 4 + j).collect();
            assert_eq!(indices, expected, "j={j}");
        }
    }

    #[test]
    fn collect_reduction_inputs_reduce_axis1_no_keepdims() {
        // input [2, 3], reduce axis [1], keepdims=false → output [2]
        // out[i] collects input[i, 0..3]
        let input_shape = [2usize, 3];
        let axes = [1];
        for i in 0..2 {
            let out_coords = [i];
            let indices = collect_reduction_inputs(&out_coords, &axes, false, &input_shape, 2);
            let expected: Vec<usize> = (0..3).map(|j| i * 3 + j).collect();
            assert_eq!(indices, expected, "i={i}");
        }
    }

    #[test]
    fn collect_reduction_inputs_single_element() {
        // input [1, 1], reduce both axes, keepdims=true → output [1, 1]
        let input_shape = [1usize, 1];
        let axes = [0, 1];
        let indices = collect_reduction_inputs(&[0, 0], &axes, true, &input_shape, 2);
        assert_eq!(indices, vec![0]);
    }
}
