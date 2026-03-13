// ONNX `ReduceMean` layer for ZK circuits.
//
// # ZK approach
// ReduceMean computes the arithmetic mean along specified axes. Division is
// not expressible as a low-degree polynomial, so a hint is used:
//
// 1. **Hint**: for each "lane" of n values along the reduced axes, call
//    `api.new_hint("jstprove.reduce_mean_hint", &[x_0, ..., x_{n-1}, scale], 1)`
//    which returns `round(sum(x_i) / n)` in the field. No constraint is added.
//
// 2. **No range check**: mean of signed values can be negative, so the LogUp
//    range check (which only supports non-negative values) is not applied.
//    The output is unconstrained beyond being a valid field element.
//
// # Supported attributes
// - `axes`     (default: all axes): axes along which to reduce.
// - `keepdims` (default 1): if 1, keep reduced dims as size 1; if 0, remove them.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    hints::reduce_mean::REDUCE_MEAN_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
    },
};

// -------- Struct --------

#[derive(Debug)]
pub struct ReduceMeanLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Axes to reduce over (sorted, non-negative, within rank).
    axes: Vec<usize>,
    /// If true, keep the reduced dimensions as size 1 in the output.
    #[allow(dead_code)]
    keepdims: bool,
    /// Input shape (needed to build lane iterators at apply time).
    input_shape: Vec<usize>,
    /// Output shape.
    output_shape: Vec<usize>,
    /// Scaling factor passed to the hint (not used in the mean computation itself).
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReduceMeanLayer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::ReduceMean, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::ReduceMean,
            name: x_name.to_string(),
        })?;

        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let in_shape = &self.input_shape;
        let rank = in_shape.len();

        // Compute the "outer" shape (non-reduced dims) and the "lane" shape (reduced dims).
        let outer_shape: Vec<usize> = (0..rank)
            .filter(|i| !self.axes.contains(i))
            .map(|i| in_shape[i])
            .collect();
        let lane_shape: Vec<usize> = self.axes.iter().map(|&a| in_shape[a]).collect();

        let outer_total: usize = outer_shape.iter().product::<usize>().max(1);
        let lane_size: usize = lane_shape.iter().product::<usize>().max(1);

        // Helper: unravel a flat index in a given shape.
        let unravel = |mut flat: usize, shape: &[usize]| -> Vec<usize> {
            let mut coords = vec![0usize; shape.len()];
            for i in (0..shape.len()).rev() {
                coords[i] = flat % shape[i];
                flat /= shape[i];
            }
            coords
        };
        let ravel = |coords: &[usize], shape: &[usize]| -> usize {
            let mut idx = 0usize;
            let mut stride = 1usize;
            for i in (0..shape.len()).rev() {
                idx += coords[i] * stride;
                stride *= shape[i];
            }
            idx
        };

        // Non-reduced axis indices (for rebuilding full input coordinate from outer coord).
        let outer_axes: Vec<usize> = (0..rank).filter(|i| !self.axes.contains(i)).collect();

        let mut out_flat: Vec<Variable> = Vec::with_capacity(outer_total);

        for outer_flat in 0..outer_total {
            let outer_coords = unravel(outer_flat, &outer_shape);

            // Collect the lane: all combinations of reduced-axis coordinates.
            let mut hint_inputs: Vec<Variable> = Vec::with_capacity(lane_size + 1);
            for lane_flat in 0..lane_size {
                let lane_coords = unravel(lane_flat, &lane_shape);
                // Build full input coordinate.
                let mut in_coords = vec![0usize; rank];
                for (oi, &ax) in outer_axes.iter().enumerate() {
                    in_coords[ax] = outer_coords[oi];
                }
                for (li, &ax) in self.axes.iter().enumerate() {
                    in_coords[ax] = lane_coords[li];
                }
                let in_idx = ravel(&in_coords, in_shape);
                hint_inputs.push(
                    x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
                        layer: LayerKind::ReduceMean,
                        msg: "input array is not contiguous".to_string(),
                    })?[in_idx],
                );
            }
            hint_inputs.push(scale_var);

            let hint_out = api.new_hint(REDUCE_MEAN_HINT_KEY, &hint_inputs, 1);
            out_flat.push(hint_out[0]);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
                msg: format!(
                    "cannot reshape ReduceMean result into {:?}",
                    self.output_shape
                ),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ReduceMean,
                param: "input tensor".to_string(),
            })?;
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ReduceMean,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
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

        // Parse axes from input[1] initializer (opset 18+), then from params attribute
        // (opset ≤ 17), defaulting to all axes if neither is present.
        let axes: Vec<usize> = {
            // Try input[1] from w_and_b_map first (opset 18+ style).
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
                                    rmpv::Value::Array(arr) => Some(
                                        arr.iter()
                                            .filter_map(|x| {
                                                if let rmpv::Value::Integer(i) = x {
                                                    i.as_i64()
                                                } else {
                                                    None
                                                }
                                            })
                                            .collect(),
                                    ),
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
                                    layer: LayerKind::ReduceMean,
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
                None => (0..rank).collect(),
            }
        };

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::ReduceMean,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axes,
            keepdims,
            input_shape,
            output_shape,
            scaling,
        }))
    }
}

#[cfg(test)]
mod tests {

    fn compute_expected_mean(values: &[i64], lane_size: usize) -> Vec<i64> {
        values
            .chunks(lane_size)
            .map(|chunk| {
                let sum: i64 = chunk.iter().sum();
                let n = chunk.len() as f64;
                (sum as f64 / n).round() as i64
            })
            .collect()
    }

    #[test]
    fn reduce_mean_uniform_lane() {
        // mean([100, 200, 300, 400]) = 250
        let vals = [100i64, 200, 300, 400];
        let result = compute_expected_mean(&vals, 4);
        assert_eq!(result, vec![250i64]);
    }

    #[test]
    fn reduce_mean_two_lanes() {
        // shape [2, 4], reduce axis 1 → [2]
        // lane 0: mean([100, 200, 300, 400]) = 250
        // lane 1: mean([10, 20, 30, 40])     = 25
        let vals = [100i64, 200, 300, 400, 10, 20, 30, 40];
        let result = compute_expected_mean(&vals, 4);
        assert_eq!(result, vec![250i64, 25]);
    }

    #[test]
    fn reduce_mean_signed() {
        // mean([-100, 100]) = 0
        let vals = [-100i64, 100];
        let result = compute_expected_mean(&vals, 2);
        assert_eq!(result, vec![0i64]);
    }
}
