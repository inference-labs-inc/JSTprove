use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{AXIS, INPUT},
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param_or_default},
        shaping::onnx_flatten,
    },
};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct FlattenLayer {
    name: String,
    axis: usize,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for FlattenLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let reshape_axis = self.axis;
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Flatten, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Flatten,
                name: input_name.clone(),
            })?
            .clone();

        let out = onnx_flatten(layer_input.clone(), reshape_axis)?;

        Ok((self.outputs.clone(), out.clone()))
    }
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Flatten,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;
        let flatten = Self {
            name: layer.name.clone(),
            axis: get_param_or_default(&layer.name, AXIS, &params, Some(&1))?,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };
        Ok(Box::new(flatten))
    }
}
