use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

pub struct EqualLayer {
    #[allow(dead_code)]
    inputs: Vec<String>,
    outputs: Vec<String>,
    result: Vec<bool>,
    output_shape: Vec<usize>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for EqualLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let zero = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let one = api.constant(CircuitField::<C>::from_u256(U256::from(1u64)));

        let out_flat: Vec<Variable> = self
            .result
            .iter()
            .map(|&v| if v { one } else { zero })
            .collect();

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Equal,
                    msg: format!("Equal output reshape failed: {e}"),
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
        let a_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Equal,
                param: "A input".to_string(),
            })?;
        let b_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Equal,
                param: "B input".to_string(),
            })?;

        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Equal,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Equal,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let a_data: ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, a_name).map_err(|e| LayerError::Other {
                layer: LayerKind::Equal,
                msg: format!("Equal input A '{a_name}' must be a compile-time constant: {e}"),
            })?;

        let b_data: ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, b_name).map_err(|e| LayerError::Other {
                layer: LayerKind::Equal,
                msg: format!("Equal input B '{b_name}' must be a compile-time constant: {e}"),
            })?;

        let a_flat = a_data.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Equal,
            msg: "A tensor is not contiguous".to_string(),
        })?;
        let b_flat = b_data.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Equal,
            msg: "B tensor is not contiguous".to_string(),
        })?;

        let result: Vec<bool> = a_flat
            .iter()
            .zip(b_flat.iter())
            .map(|(a, b)| a == b)
            .collect();

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            result,
            output_shape,
        }))
    }
}
