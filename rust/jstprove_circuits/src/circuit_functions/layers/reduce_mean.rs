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
    n_bits: usize,
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
        let total_out: usize = self.output_shape.iter().product();

        let n_const = api.constant(CircuitField::<C>::from_u256(U256::from(n as u64)));
        let half_n = n / 2;
        let half_n_const = api.constant(CircuitField::<C>::from_u256(U256::from(half_n as u64)));
        let remainder_bits = n.next_power_of_two().trailing_zeros() as usize;
        let remainder_bits = remainder_bits.max(1);

        let n_bits_u32 = u32::try_from(self.n_bits).unwrap_or(u32::MAX);
        let offset = U256::from(1u64) << (n_bits_u32 - 1);
        let offset_var = api.constant(CircuitField::<C>::from_u256(offset));

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

            let biased_sum = api.add(sum, half_n_const);
            let mean_q = api.unconstrained_int_div(biased_sum, n_const);
            let remainder = api.unconstrained_mod(biased_sum, n_const);

            let rhs = api.mul(n_const, mean_q);
            let rhs = api.add(rhs, remainder);
            api.assert_is_equal(biased_sum, rhs);

            logup_ctx
                .range_check::<C, Builder>(api, remainder, remainder_bits)
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::ReduceMean,
                    msg: format!("remainder range check: {e}"),
                })?;

            if !n.is_power_of_two() {
                let n_minus_one =
                    api.constant(CircuitField::<C>::from_u256(U256::from(n as u64 - 1)));
                let upper = api.sub(n_minus_one, remainder);
                logup_ctx
                    .range_check::<C, Builder>(api, upper, remainder_bits)
                    .map_err(|e| LayerError::Other {
                        layer: LayerKind::ReduceMean,
                        msg: format!("remainder upper bound: {e}"),
                    })?;
            }

            let mean_shifted = api.add(mean_q, offset_var);
            logup_ctx
                .range_check::<C, Builder>(api, mean_shifted, self.n_bits)
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::ReduceMean,
                    msg: format!("mean output range check: {e}"),
                })?;

            out_storage.push(mean_q);
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

        let lane_size: usize = axes.iter().map(|&a| input_shape[a]).product();
        if lane_size == 0 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ReduceMean,
                msg: "reduction lane size is zero".to_string(),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        if n_bits == 0 || n_bits > 128 {
            return Err(LayerError::Other {
                layer: LayerKind::ReduceMean,
                msg: format!("n_bits={n_bits} out of supported range 1..=128"),
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
            n_bits,
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

    #[test]
    fn reduce_mean_rounding() {
        // sum=7, n=3, 7/3 = 2.333 → round → 2
        let vals = [1i64, 2, 4];
        let result = compute_expected_mean(&vals, 3);
        assert_eq!(result, vec![2i64]);

        // sum=8, n=3, 8/3 = 2.667 → round → 3
        let vals = [1i64, 3, 4];
        let result = compute_expected_mean(&vals, 3);
        assert_eq!(result, vec![3i64]);

        // sum=5, n=2, 5/2 = 2.5 → round → 2 (banker's rounds to even? no, f64 rounds to 3)
        // Actually f64 2.5.round() = 3 in Rust (round half away from zero)
        let vals = [2i64, 3];
        let result = compute_expected_mean(&vals, 2);
        assert_eq!(result, vec![3i64]);
    }

    #[test]
    fn reduce_mean_tolerance_boundary() {
        // Exact division: sum=12, n=4, mean=3, remainder=0
        let vals = [1i64, 2, 4, 5];
        let result = compute_expected_mean(&vals, 4);
        assert_eq!(result, vec![3i64]);

        // Off-by-one from exact: sum=13, n=4, mean=3.25→3, remainder=1 ≤ floor(4/2)=2 ✓
        let vals = [1i64, 3, 4, 5];
        let result = compute_expected_mean(&vals, 4);
        assert_eq!(result, vec![3i64]);

        // Mixed sign, non-exact: sum=-1, n=3, mean=-0.333→0
        let vals = [-2i64, 0, 1];
        let result = compute_expected_mean(&vals, 3);
        assert_eq!(result, vec![0i64]);
    }
}
