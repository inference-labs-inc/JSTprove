use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, Axis};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::extract_params_and_expected_shape,
};

#[derive(Debug)]
pub struct ConcatLayer {
    name: String,
    axis: isize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConcatLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let mut arrays: Vec<ArrayD<Variable>> = Vec::new();

        for input_name in &self.inputs {
            if let Some(arr) = input.get(input_name) {
                arrays.push(arr.clone());
            }
        }

        if arrays.is_empty() {
            return Err(CircuitError::from(LayerError::MissingInput {
                layer: LayerKind::Concat,
                name: self.inputs.first().cloned().unwrap_or_default(),
            }));
        }

        let first = &arrays[0];
        let ndim = first.ndim();
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        let out = ndarray::concatenate(Axis(axis), &views).map_err(|e| LayerError::Other {
            layer: LayerKind::Concat,
            msg: format!("Concatenation failed: {e}"),
        })?;

        Ok((self.outputs.clone(), out))
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
                layer: LayerKind::Concat,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let axis: isize = params
            .get("axis")
            .and_then(|v| v.as_i64())
            .map(|v| v as isize)
            .unwrap_or(0);

        Ok(Box::new(Self {
            name: layer.name.clone(),
            axis,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
