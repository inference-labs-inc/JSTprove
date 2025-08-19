use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use serde_json::Value;
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::{constants::VALUE, onnx_model::get_param}};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ConstantLayer {
    name: String,
    value: Value,
    outputs: Vec<String>,
}

// -------- Implementations --------

// TODO remove constants from python side. Incorporate into the layer that uses it instead
impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantLayer {
    // Passthrough
    fn apply(
        &self,
        api: &mut Builder,
        _input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {

        Ok((self.outputs.clone(), ArrayD::from_shape_vec(IxDyn(&[1]), vec![api.constant(0)]).unwrap()))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {
        let constant = Self {
            name: layer.name.clone(),
            value: get_param(&layer.name, VALUE, &layer.params.clone().unwrap()),
            outputs: layer.outputs.to_vec()
        };

        Ok(Box::new(constant))
    }
}
