use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, ShiftRangeContext, constrained_max},
    layers::{LayerError, LayerKind, layer_ops::LayerOp, reduce_sum::collect_reduction_inputs},
    utils::onnx_model::get_w_or_b,
};

#[derive(Debug)]
pub struct ReduceMaxLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    axes: Vec<usize>,
    keepdims: bool,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    n_bits: usize,
}

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

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReduceMaxLayer {
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
                layer: LayerKind::ReduceMax,
                param: "input tensor".to_string(),
            })?;

        let x_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ReduceMax,
                name: input_name.clone(),
            })?;

        if x_input.shape() != self.input_shape.as_slice() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ReduceMax,
                msg: format!(
                    "ReduceMax runtime input shape {:?} does not match expected {:?}",
                    x_input.shape(),
                    self.input_shape
                ),
            }
            .into());
        }

        let rank = self.input_shape.len();

        let input_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::ReduceMax,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        let shift_exponent = self.n_bits.saturating_sub(1);
        let shift_ctx =
            ShiftRangeContext::new::<C, Builder>(api, LayerKind::ReduceMax, shift_exponent)?;

        let total_out: usize = self.output_shape.iter().product();
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

            if input_indices.is_empty() {
                return Err(LayerError::Other {
                    layer: LayerKind::ReduceMax,
                    msg: "ReduceMax: empty reduction lane".to_string(),
                }
                .into());
            }

            let lane_vars: Vec<Variable> = input_indices.iter().map(|&i| input_flat[i]).collect();
            let max_var = constrained_max(api, &shift_ctx, logup_ctx, &lane_vars)?;
            out_storage.push(max_var);
        }

        let result =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_storage).map_err(|_| {
                LayerError::InvalidShape {
                    layer: LayerKind::ReduceMax,
                    msg: format!(
                        "ReduceMax: cannot reshape result into shape {:?}",
                        self.output_shape
                    ),
                }
            })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_lines
    )]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        if layer.inputs.is_empty() {
            return Err(LayerError::Other {
                layer: LayerKind::ReduceMax,
                msg: "ReduceMax expects at least 1 input".to_string(),
            }
            .into());
        }
        if layer.outputs.len() != 1 {
            return Err(LayerError::Other {
                layer: LayerKind::ReduceMax,
                msg: format!(
                    "ReduceMax expects exactly 1 output, got {}",
                    layer.outputs.len()
                ),
            }
            .into());
        }

        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ReduceMax,
                param: "input tensor".to_string(),
            })?;
        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ReduceMax,
                msg: format!("missing input shape for '{input_name}'"),
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
            let from_input: Option<Vec<i64>> = if let Some(axes_name) = layer.inputs.get(1) {
                let arr = get_w_or_b(layer_context.w_and_b_map, axes_name).map_err(|e| {
                    LayerError::Other {
                        layer: LayerKind::ReduceMax,
                        msg: format!("invalid/missing axes input for ReduceMax '{axes_name}': {e}"),
                    }
                })?;

                if arr.ndim() != 1 {
                    return Err(LayerError::InvalidShape {
                        layer: LayerKind::ReduceMax,
                        msg: format!(
                            "axes tensor '{axes_name}' must be 1D, got shape {:?}",
                            arr.shape()
                        ),
                    }
                    .into());
                }

                let slice = arr.as_slice().ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::ReduceMax,
                    msg: format!("axes tensor '{axes_name}' is not contiguous"),
                })?;

                Some(slice.to_vec())
            } else {
                None
            };

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
                    if v.is_empty() {
                        (0..rank).collect()
                    } else {
                        let mut axes: Vec<usize> = v
                            .iter()
                            .map(|&a| {
                                let a = if a < 0 { a + rank as i64 } else { a };
                                if a < 0 || a as usize >= rank {
                                    return Err(LayerError::InvalidShape {
                                        layer: LayerKind::ReduceMax,
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
                }
                None => (0..rank).collect(),
            }
        };

        let computed_output_shape: Vec<usize> = if keepdims {
            input_shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                .collect()
        } else {
            input_shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
                .collect()
        };

        let n_bits = layer_context.n_bits_for(&layer.name);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axes,
            keepdims,
            input_shape,
            output_shape: computed_output_shape,
            n_bits,
        }))
    }
}
