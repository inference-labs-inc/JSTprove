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

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

// -------- Struct --------

// Fields populated by `build` are kept for when a sound mean constraint is
// eventually implemented. `apply` always returns an error in the meantime.
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReduceMeanLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Axes to reduce over (sorted, non-negative, within rank).
    axes: Vec<usize>,
    /// If true, keep the reduced dimensions as size 1 in the output.
    keepdims: bool,
    /// Input shape.
    input_shape: Vec<usize>,
    /// Output shape.
    output_shape: Vec<usize>,
    /// Scaling factor `2^scale_exponent`.
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReduceMeanLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // ReduceMean is not soundly implementable via hint alone: the hint
        // produces an unconstrained output that a malicious prover can set to
        // any value. Full soundness requires a verifiable mean constraint
        // (e.g., sum check + range proof), which is not yet implemented.
        Err(LayerError::Other {
            layer: LayerKind::ReduceMean,
            msg: "ReduceMean is not yet supported in the Expander backend: soundly \
                  proving the mean relation requires constraints that are not yet \
                  implemented."
                .to_string(),
        }
        .into())
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

        // Validate that the output shape from shapes_map matches what the
        // axes+keepdims combination implies from the input shape.
        let expected_output_shape: Vec<usize> = if keepdims {
            (0..rank)
                .map(|i| if axes.contains(&i) { 1 } else { input_shape[i] })
                .collect()
        } else {
            (0..rank)
                .filter(|i| !axes.contains(i))
                .map(|i| input_shape[i])
                .collect()
        };
        if output_shape != expected_output_shape {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
                msg: format!(
                    "output shape {output_shape:?} from shapes_map does not match \
                     expected {expected_output_shape:?} \
                     (input={input_shape:?}, axes={axes:?}, keepdims={keepdims})"
                ),
            }
            .into());
        }

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
