use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name, tensor_ops::broadcast_two_arrays},
};

#[derive(Debug)]
pub struct AndLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for AndLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let a_name = get_input_name(&self.inputs, 0, LayerKind::And, INPUT)?;
        let b_name = get_input_name(&self.inputs, 1, LayerKind::And, INPUT)?;

        let a_input = input.get(a_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::And,
            name: a_name.to_string(),
        })?;
        let b_input = input.get(b_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::And,
            name: b_name.to_string(),
        })?;

        let (a_bc, b_bc) = broadcast_two_arrays(a_input, b_input)?;

        let shape = a_bc.shape().to_vec();
        let a_flat = a_bc.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::And,
            msg: "A tensor is not contiguous after broadcast".to_string(),
        })?;
        let b_flat = b_bc.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::And,
            msg: "B tensor is not contiguous after broadcast".to_string(),
        })?;

        let mut out_storage: Vec<Variable> = Vec::with_capacity(a_flat.len());

        for (&a, &b) in a_flat.iter().zip(b_flat.iter()) {
            let y = api.mul(a, b);
            out_storage.push(y);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::And,
                msg: format!("AndLayer: cannot reshape result into shape {shape:?}"),
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
