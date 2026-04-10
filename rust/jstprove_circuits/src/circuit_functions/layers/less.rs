use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        onnx_model::get_optional_w_or_b,
        tensor_ops::{broadcast_two_arrays, load_array_constants_or_get_inputs},
    },
};

enum LessMode {
    Precomputed {
        result: Vec<bool>,
        output_shape: Vec<usize>,
    },
    Runtime {
        initializer_a: Option<ArrayD<i64>>,
        initializer_b: Option<ArrayD<i64>>,
    },
}

pub struct LessLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    mode: LessMode,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LessLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        match &self.mode {
            LessMode::Precomputed {
                result,
                output_shape,
            } => {
                let zero = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
                let one = api.constant(CircuitField::<C>::from_u256(U256::from(1u64)));

                let out_flat: Vec<Variable> =
                    result.iter().map(|&v| if v { one } else { zero }).collect();

                let out_array =
                    ArrayD::from_shape_vec(IxDyn(output_shape), out_flat).map_err(|e| {
                        LayerError::InvalidShape {
                            layer: LayerKind::Less,
                            msg: format!("Less output reshape failed: {e}"),
                        }
                    })?;

                Ok((self.outputs.clone(), out_array))
            }
            LessMode::Runtime {
                initializer_a,
                initializer_b,
            } => {
                let a_name = &self.inputs[0];
                let b_name = &self.inputs[1];

                let a_arr = load_array_constants_or_get_inputs(
                    api,
                    input,
                    a_name,
                    initializer_a,
                    LayerKind::Less,
                )?;
                let b_arr = load_array_constants_or_get_inputs(
                    api,
                    input,
                    b_name,
                    initializer_b,
                    LayerKind::Less,
                )?;

                let (a_bc, b_bc) = broadcast_two_arrays(&a_arr, &b_arr)?;

                let a_flat = a_bc.as_slice().ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Less,
                    msg: "A tensor is not contiguous after broadcast".to_string(),
                })?;
                let b_flat = b_bc.as_slice().ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Less,
                    msg: "B tensor is not contiguous after broadcast".to_string(),
                })?;

                let out_flat: Vec<Variable> = a_flat
                    .iter()
                    .zip(b_flat.iter())
                    .map(|(&a_val, &b_val)| api.gt(b_val, a_val))
                    .collect();

                let out_array =
                    ArrayD::from_shape_vec(IxDyn(a_bc.shape()), out_flat).map_err(|e| {
                        LayerError::InvalidShape {
                            layer: LayerKind::Less,
                            msg: format!("Less output reshape failed: {e}"),
                        }
                    })?;

                Ok((self.outputs.clone(), out_array))
            }
        }
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let a_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Less,
                param: "A input".to_string(),
            })?;
        let b_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Less,
                param: "B input".to_string(),
            })?;

        let a_opt = get_optional_w_or_b(layer_context, a_name).map_err(|e| LayerError::Other {
            layer: LayerKind::Less,
            msg: format!("Less input A '{a_name}': {e}"),
        })?;
        let b_opt = get_optional_w_or_b(layer_context, b_name).map_err(|e| LayerError::Other {
            layer: LayerKind::Less,
            msg: format!("Less input B '{b_name}': {e}"),
        })?;

        let mode = if let (Some(a_data), Some(b_data)) = (&a_opt, &b_opt) {
            let output_name =
                layer
                    .outputs
                    .first()
                    .ok_or_else(|| LayerError::MissingParameter {
                        layer: LayerKind::Less,
                        param: "output tensor".to_string(),
                    })?;

            let output_shape = layer_context
                .shapes_map
                .get(output_name.as_str())
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Less,
                    msg: format!("missing output shape for '{output_name}'"),
                })?
                .clone();

            let a_bc = a_data
                .broadcast(IxDyn(&output_shape))
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Less,
                    msg: format!(
                        "cannot broadcast A {:?} to {output_shape:?}",
                        a_data.shape()
                    ),
                })?
                .to_owned();
            let b_bc = b_data
                .broadcast(IxDyn(&output_shape))
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Less,
                    msg: format!(
                        "cannot broadcast B {:?} to {output_shape:?}",
                        b_data.shape()
                    ),
                })?
                .to_owned();

            let a_flat = a_bc.as_slice().ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Less,
                msg: "A tensor is not contiguous after broadcast".to_string(),
            })?;
            let b_flat = b_bc.as_slice().ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Less,
                msg: "B tensor is not contiguous after broadcast".to_string(),
            })?;

            let result: Vec<bool> = a_flat
                .iter()
                .zip(b_flat.iter())
                .map(|(a, b)| a < b)
                .collect();

            LessMode::Precomputed {
                result,
                output_shape,
            }
        } else {
            LessMode::Runtime {
                initializer_a: a_opt,
                initializer_b: b_opt,
            }
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            mode,
        }))
    }
}
