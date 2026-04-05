use std::collections::HashMap;

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
};

const GROUP_NORM_UNSUPPORTED: &str = "GroupNormalization is not yet supported in the circuit backend; \
     use InstanceNormalization or LayerNormalization instead.";

pub struct GroupNormLayer;

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GroupNormLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        Err(LayerError::Other {
            layer: LayerKind::GroupNormalization,
            msg: GROUP_NORM_UNSUPPORTED.to_string(),
        }
        .into())
    }

    fn build(
        _layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        Err(LayerError::Other {
            layer: LayerKind::GroupNormalization,
            msg: GROUP_NORM_UNSUPPORTED.to_string(),
        }
        .into())
    }
}
