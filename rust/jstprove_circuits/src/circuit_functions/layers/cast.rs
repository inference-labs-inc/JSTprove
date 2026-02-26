use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param},
    },
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct CastLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    to_type: i64, // ONNX TensorProto.DataType code; stored for debug/validation
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for CastLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Cast, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Cast,
                name: input_name.clone(),
            })?
            .clone();
        // Cast is a no-op in the ZK circuit: the quantizer has already resolved
        // types; field elements are typeless.
        Ok((self.outputs.clone(), layer_input))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let to_type: i64 =
            get_param(&layer.name, "to", &params).map_err(|_| LayerError::MissingParameter {
                layer: LayerKind::Cast,
                param: "to".to_string(),
            })?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            to_type,
        }))
    }
}
