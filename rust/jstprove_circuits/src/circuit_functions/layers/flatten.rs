use std::collections::HashMap;

use ndarray::ArrayD;
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::{constants::AXIS, onnx_model::{extract_params_and_expected_shape, get_param_or_default}, shaping::onnx_flatten}};

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
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let reshape_axis = self.axis.clone();
        let layer_input = input.get(&self.inputs[0])
        .ok_or_else(|| panic!("Missing input {}", self.inputs[0].clone())).unwrap()
    .clone();

        let out = onnx_flatten(layer_input.clone(), reshape_axis);

        Ok((self.outputs.clone(), out.clone()))
    }
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer);
        let flatten = Self{
            name: layer.name.clone(),
            axis: get_param_or_default(&layer.name, AXIS, &params, Some(&1)),
            input_shape: expected_shape.to_vec(),
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
        };
        Ok(Box::new(flatten))
    }
}
