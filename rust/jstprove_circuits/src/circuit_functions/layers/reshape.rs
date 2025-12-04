use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{INPUT, INPUT_SHAPE},
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param_or_default},
        shaping::infer_reshape_shape,
    },
};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReshapeLayer {
    name: String,
    shape: Vec<isize>,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}
// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReshapeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let reshape_shape = self.shape.clone();
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Conv, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Conv,
                name: input_name.clone(),
            })?
            .clone();

        let inferred_shape = infer_reshape_shape(layer_input.len(), &reshape_shape)?;

        let out = layer_input
            .into_shape_with_order(IxDyn(&inferred_shape))
            .map_err(|_| LayerError::InvalidShape {
                layer: LayerKind::Reshape,
                msg: format!("Cannot reshape into {inferred_shape:?}"),
            })?;

        Ok((self.outputs.clone(), out.clone()))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let shape_name = get_input_name(&layer.inputs, 1, LayerKind::Reshape, INPUT_SHAPE)?;
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Reshape,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;
        let output_shape = layer_context.shapes_map.get(&layer.outputs.clone()[0]);
        let output_shape_isize: Option<Vec<isize>> = output_shape.map(|v| {
            v.iter()
                .filter_map(|&x| x.try_into().ok()) // convert usize -> isize, ignore if it fails
                .collect()
        });

        let shape: Vec<isize> = get_param_or_default(
            &layer.name,
            shape_name,
            &params,
            output_shape_isize.as_ref(),
        )?;

        let reshape = Self {
            name: layer.name.clone(),
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            shape,
        };
        Ok(Box::new(reshape))
    }
}
