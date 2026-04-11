use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::tensor_ops::load_circuit_constant,
};

#[derive(Debug)]
pub struct ConstantLayer {
    outputs: Vec<String>,
    values: ArrayD<i64>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let elements: Vec<Variable> = self
            .values
            .iter()
            .map(|&v| load_circuit_constant::<C, Builder>(api, v))
            .collect();
        let arr = ArrayD::from_shape_vec(self.values.raw_dim(), elements).map_err(|e| {
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
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let out_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Constant,
                param: "output tensor".to_string(),
            })?;

        let values = layer_context
            .get_constant(out_name)
            .cloned()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Constant,
                msg: format!(
                    "Constant node output '{out_name}' has no tensor payload in constants_map; \
                     the ONNX slice must include the Constant node's value as an initializer"
                ),
            })?;

        Ok(Box::new(Self {
            outputs: layer.outputs.clone(),
            values,
        }))
    }
}
