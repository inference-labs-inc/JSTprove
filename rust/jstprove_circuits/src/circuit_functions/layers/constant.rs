use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::tensor_ops::load_circuit_constant,
};

#[derive(Debug)]
pub struct ConstantLayer {
    outputs: Vec<String>,
    values: Option<ArrayD<i64>>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let arr = match &self.values {
            Some(vals) => {
                let elements: Vec<Variable> = vals
                    .iter()
                    .map(|&v| load_circuit_constant::<C, Builder>(api, v))
                    .collect();
                ArrayD::from_shape_vec(vals.raw_dim(), elements).map_err(|e| {
                    LayerError::InvalidShape {
                        layer: LayerKind::Constant,
                        msg: e.to_string(),
                    }
                })?
            }
            None => ArrayD::from_shape_vec(IxDyn(&[1]), vec![api.constant(0)]).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Constant,
                    msg: e.to_string(),
                }
            })?,
        };

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
        let values = layer
            .outputs
            .first()
            .and_then(|out_name| layer_context.get_constant(out_name))
            .cloned();

        Ok(Box::new(Self {
            outputs: layer.outputs.clone(),
            values,
        }))
    }
}
