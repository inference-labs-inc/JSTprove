use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::layers::math::matrix_addition;
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        graph_pattern_matching::PatternRegistry,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct AddLayer {
    name: String,
    // weights: ArrayD<i64>, //This should be an optional field
    optimization_pattern: PatternRegistry,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for AddLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let a_name = get_input_name(&self.inputs, 0, LayerKind::Add, INPUT)?;
        let b_name = get_input_name(&self.inputs, 1, LayerKind::Add, INPUT)?;

        let a_input = input
            .get(&a_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Add,
                name: a_name.clone(),
            })?;

        let b_input = input
            .get(&b_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Add,
                name: b_name.clone(),
            })?
            .clone();

        // let mut weights_array = load_array_constants(api, &self.weights)
        //     .into_dimensionality::<Ix2>()
        //     .map_err(|_| LayerError::InvalidShape {
        //         layer: LayerKind::Add,
        //         msg: format!("Expected 2D weights array for layer {}", self.name),
        //     })?;

        // Matrix multiplication and bias addition
        let result = matrix_addition(api, a_input, b_input, LayerKind::Add)?;
        Ok((self.outputs.clone(), result))
    }
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (_params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Add,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;
        let add = Self {
            name: layer.name.clone(),
            // weights: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[1])?,
            optimization_pattern,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };
        Ok(Box::new(add))
    }
}
