use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::layers::math::matrix_addition;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::{
    broadcast_two_arrays, load_array_constants_or_get_inputs,
};
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
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
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

        let a_input = load_array_constants_or_get_inputs(
            api,
            &input,
            a_name,
            &self.initializer_a,
            LayerKind::Add,
        )?;

        let b_input = load_array_constants_or_get_inputs(
            api,
            &input,
            b_name,
            &self.initializer_b,
            LayerKind::Add,
        )?;

        let (a_bc, b_bc) = broadcast_two_arrays(&a_input, &b_input)?;

        // Matrix multiplication and bias addition
        let result = matrix_addition(api, &a_bc, b_bc, LayerKind::Add)?;
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

        let initializer_a = get_optional_w_or_b(layer_context, &layer.inputs[0]);
        let initializer_b = get_optional_w_or_b(layer_context, &layer.inputs[1]);

        let add = Self {
            name: layer.name.clone(),
            // weights: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[1])?,
            optimization_pattern,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
        };
        Ok(Box::new(add))
    }
}
