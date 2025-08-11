use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::onnx_model::get_param_or_default};


// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReshapeLayer {
    name: String,
    shape: Vec<usize>,
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
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let reshape_shape = self.shape.clone();
        let layer_input = input.get(&self.inputs[0]).unwrap();
        let out = &layer_input.clone()
            .into_shape_with_order(IxDyn(&reshape_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions.");

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
        let shape_name = layer.inputs[1].clone();
        let params = layer.params.clone().unwrap();
        
        let expected_shape = match layer_context.shapes_map.get(&layer.inputs[0]){
            Some(input_shape) => input_shape,
            None => panic!("Error getting output shape for layer {}", layer.name)
        };
        let output_shape = layer_context.shapes_map.get(&layer.outputs.to_vec()[0]);
        let reshape = Self{
            name: layer.name.clone(),
            input_shape: expected_shape.to_vec(),
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
            shape: get_param_or_default(&layer.name, &shape_name, &params, output_shape)
        };
        Ok(Box::new(reshape))
    }
}