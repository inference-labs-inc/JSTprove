use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
};

#[derive(Debug)]
pub struct ConstantLayer {
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantLayer {
    // Passthrough
    fn apply(
        &self,
        api: &mut Builder,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let arr = ArrayD::from_shape_vec(IxDyn(&[1]), vec![api.constant(0)]).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::Constant,
                msg: e.to_string(),
            }
        })?;

        Ok((self.outputs.clone(), arr))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let constant = Self {
            outputs: layer.outputs.clone(),
        };

        Ok(Box::new(constant))
    }
}
