use std::collections::HashMap;

use ndarray::ArrayD;
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::{LayerBuilder, LayerOp}, utils::{onnx_model::get_param_or_default, shaping::onnx_flatten}};

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
impl FlattenLayer{
    pub fn new(
        name: String,
        axis: usize,
        input_shape: Vec<usize>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name,
            axis: axis,
            input_shape: input_shape,
            inputs: inputs,
            outputs: outputs,
        }
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for FlattenLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let reshape_axis = self.axis.clone();
        let layer_input = input.get(&self.inputs[0]).unwrap();

        let out = onnx_flatten(layer_input.clone(), reshape_axis);

        Ok((self.outputs.clone(), out.clone()))
    }
}


impl<C: Config, Builder: RootAPI<C>> LayerBuilder<C, Builder> for FlattenLayer{
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::layers::layer_ops::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {
        let params = layer.params.clone().unwrap();
                
        let expected_shape = match layer_context.shapes_map.get(&layer.inputs[0]){
            Some(input_shape) => input_shape,
            None => panic!("Error getting output shape for layer {}", layer.name)
        };
        let flatten = FlattenLayer::new(
            layer.name.clone(),
            get_param_or_default(&layer.name, &"axis", &params, Some(&1)),
            expected_shape.to_vec(),
            layer.inputs.to_vec(),
            layer.outputs.to_vec(),
        );
        Ok(Box::new(flatten))
    }
}