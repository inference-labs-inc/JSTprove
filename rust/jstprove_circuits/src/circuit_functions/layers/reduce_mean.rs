use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{
        LayerError, LayerKind,
        layer_ops::LayerOp,
        reduce_sum::{collect_reduction_inputs, unravel},
    },
    utils::onnx_model::get_w_or_b,
};

#[derive(Debug)]
pub struct ReduceMeanLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    axes: Vec<usize>,
    keepdims: bool,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    scaling: u64,
    lane_size: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReduceMeanLayer {
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = self
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ReduceMean,
                param: "input tensor".to_string(),
            })?;

        let x_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ReduceMean,
                name: input_name.clone(),
            })?;

        if x_input.shape() != self.input_shape.as_slice() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
                msg: format!(
                    "runtime input shape {:?} does not match expected {:?}",
                    x_input.shape(),
                    self.input_shape
                ),
            }
            .into());
        }

        let rank = self.input_shape.len();
        let input_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::ReduceMean,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        let n = self.lane_size;
        let half_n = n / 2;
        let total_out: usize = self.output_shape.iter().product();

        let n_const = api.constant(CircuitField::<C>::from_u256(U256::from(n as u64)));
        let bias = self.scaling * n as u64;
        let bias_const = api.constant(CircuitField::<C>::from_u256(U256::from(
            bias + half_n as u64,
        )));
        let bias_div_n = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let remainder_bits = usize::BITS as usize - n.leading_zeros() as usize + 1;

        let mut out_storage: Vec<Variable> = Vec::with_capacity(total_out);

        for out_flat in 0..total_out {
            let out_coords = unravel(out_flat, &self.output_shape);
            let input_indices = collect_reduction_inputs(
                &out_coords,
                &self.axes,
                self.keepdims,
                &self.input_shape,
                rank,
            );

            let sum = if input_indices.is_empty() {
                api.constant(CircuitField::<C>::from_u256(U256::from(0u64)))
            } else {
                let mut acc = input_flat[input_indices[0]];
                for &idx in &input_indices[1..] {
                    acc = api.add(acc, input_flat[idx]);
                }
                acc
            };

            let shifted_sum = api.add(sum, bias_const);
            let shifted_q = api.unconstrained_int_div(shifted_sum, n_const);
            let remainder = api.unconstrained_mod(shifted_sum, n_const);

            let rhs_mul = api.mul(n_const, shifted_q);
            let rhs = api.add(rhs_mul, remainder);
            api.assert_is_equal(shifted_sum, rhs);

            logup_ctx
                .range_check::<C, Builder>(api, remainder, remainder_bits)
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::ReduceMean,
                    msg: format!("remainder range check: {e}"),
                })?;

            let mean = api.sub(shifted_q, bias_div_n);
            out_storage.push(mean);
        }

        let result =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_storage).map_err(|_| {
                LayerError::InvalidShape {
                    layer: LayerKind::ReduceMean,
                    msg: format!("cannot reshape result into shape {:?}", self.output_shape),
                }
            })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
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
                    "output shape {output_shape:?} does not match expected {expected_output_shape:?} \
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

        let lane_size: usize = axes.iter().map(|&a| input_shape[a]).product();
        if lane_size == 0 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
                msg: "reduction lane size is zero".to_string(),
            }
            .into());
        }

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axes,
            keepdims,
            input_shape,
            output_shape,
            scaling,
            lane_size,
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
        let vals = [100i64, 200, 300, 400];
        let result = compute_expected_mean(&vals, 4);
        assert_eq!(result, vec![250i64]);
    }

    #[test]
    fn reduce_mean_two_lanes() {
        let vals = [100i64, 200, 300, 400, 10, 20, 30, 40];
        let result = compute_expected_mean(&vals, 4);
        assert_eq!(result, vec![250i64, 25]);
    }

    #[test]
    fn reduce_mean_signed() {
        let vals = [-100i64, 100];
        let result = compute_expected_mean(&vals, 2);
        assert_eq!(result, vec![0i64]);
    }
}
