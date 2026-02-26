use std::collections::HashMap;

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::gadgets::linear_algebra::{matrix_addition, matrix_subtraction};
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::{
    broadcast_two_arrays, load_array_constants_or_get_inputs,
};
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct BinaryArithLayer {
    kind: LayerKind,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for BinaryArithLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let a_name = get_input_name(&self.inputs, 0, self.kind.clone(), INPUT)?;
        let b_name = get_input_name(&self.inputs, 1, self.kind.clone(), INPUT)?;

        let a_input = load_array_constants_or_get_inputs(
            api,
            input,
            a_name,
            &self.initializer_a,
            self.kind.clone(),
        )?;

        let b_input = load_array_constants_or_get_inputs(
            api,
            input,
            b_name,
            &self.initializer_b,
            self.kind.clone(),
        )?;

        let (a_bc, b_bc) = broadcast_two_arrays(&a_input, &b_input)?;

        let result = match self.kind {
            LayerKind::Add => matrix_addition(api, &a_bc, b_bc, LayerKind::Add)?,
            LayerKind::Sub => matrix_subtraction(api, &a_bc, b_bc, LayerKind::Sub)?,
            _ => unreachable!("BinaryArithLayer only supports Add and Sub"),
        };
        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let a_name = get_input_name(&layer.inputs, 0, LayerKind::Add, INPUT)?;
        let b_name = get_input_name(&layer.inputs, 1, LayerKind::Add, INPUT)?;
        let initializer_a = get_optional_w_or_b(layer_context, a_name)?;
        let initializer_b = get_optional_w_or_b(layer_context, b_name)?;

        let kind = LayerKind::try_from(layer.op_type.as_str())?;

        Ok(Box::new(Self {
            kind,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
        }))
    }
}
