use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct NotLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for NotLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Not, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Not,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let one_var = api.constant(CircuitField::<C>::from_u256(U256::from(1u64)));

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            api.assert_is_bool(x);
            let y = api.sub(one_var, x);
            out_storage.push(y);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::Not,
                msg: format!("NotLayer: cannot reshape result into shape {shape:?}: {e}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
