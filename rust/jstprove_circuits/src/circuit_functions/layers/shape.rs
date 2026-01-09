use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

#[derive(Debug)]
pub struct ShapeLayer {
    name: String,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ShapeLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Shape, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Shape,
                name: input_name.clone(),
            })?;

        let shape_values: Vec<Variable> = layer_input
            .shape()
            .iter()
            .map(|&dim| api.constant(dim as u32))
            .collect();

        let out = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[shape_values.len()]),
            shape_values,
        )
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Shape,
            msg: format!("Failed to create shape array: {e}"),
        })?;

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (_, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Shape,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            input_shape: expected_shape,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
