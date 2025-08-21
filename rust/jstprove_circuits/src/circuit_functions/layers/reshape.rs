use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::{layer_ops::LayerOp, LayerError, LayerKind}, utils::{onnx_model::{extract_params_and_expected_shape, get_param_or_default}, shaping::infer_reshape_shape}, CircuitError};


// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReshapeLayer {
    name: String,
    shape: Vec<isize>,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}
// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReshapeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), CircuitError> {
        let reshape_shape = self.shape.clone();
        let layer_input = input.get(&self.inputs[0])
        .ok_or_else(|| panic!("Missing input {}", self.inputs[0].clone())).unwrap()
    .clone();
        let inferred_shape = infer_reshape_shape(layer_input.len(), &reshape_shape)?;

        let out = layer_input.into_shape_with_order(IxDyn(&inferred_shape))
            .map_err(|_| LayerError::InvalidShape{layer: LayerKind::Reshape, msg: format!("Cannot reshape into {:?}", inferred_shape)})?;

        Ok((self.outputs.clone(), out.clone()))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let shape_name = layer.inputs.get(1)
        .ok_or_else(|| panic!("Missing input shape name")).unwrap()
        .clone();
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer).unwrap();
        let output_shape = layer_context.shapes_map.get(&layer.outputs.to_vec()[0]);
        let output_shape_isize = output_shape.map(|v| v.iter().map(|&x| x as isize).collect());


        let shape: Vec<isize> = get_param_or_default(&layer.name, &shape_name, &params, output_shape_isize.as_ref())?;

        let reshape = Self{
            name: layer.name.clone(),
            input_shape: expected_shape.to_vec(),
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
            shape: shape
        };
        Ok(Box::new(reshape))
    }
}
