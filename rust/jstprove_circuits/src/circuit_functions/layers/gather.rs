use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

#[derive(Debug)]
pub struct GatherLayer {
    name: String,
    axis: isize,
    indices: Vec<i64>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GatherLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Gather, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Gather,
                name: input_name.clone(),
            })?
            .clone();

        let ndim = layer_input.ndim();
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let axis_len = layer_input.shape()[axis] as i64;

        let normalized_indices: Vec<usize> = self
            .indices
            .iter()
            .map(|&idx| {
                if idx < 0 {
                    (axis_len + idx) as usize
                } else {
                    idx as usize
                }
            })
            .collect();

        let out = layer_input.select(ndarray::Axis(axis), &normalized_indices);

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
        let (params, _) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Gather,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;

        let axis: isize = params
            .get("axis")
            .and_then(|v| v.as_i64())
            .map(|v| v as isize)
            .unwrap_or(0);

        let indices: Vec<i64> = params
            .get("indices")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_default();

        Ok(Box::new(Self {
            name: layer.name.clone(),
            axis,
            indices,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
