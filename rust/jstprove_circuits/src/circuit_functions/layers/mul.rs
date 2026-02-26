use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::gadgets::linear_algebra::matrix_hadamard_product;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::quantization::{MaybeRescaleParams, maybe_rescale};
use crate::circuit_functions::utils::tensor_ops::{
    broadcast_two_arrays, load_array_constants_or_get_inputs,
};
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT, graph_pattern_matching::PatternRegistry, onnx_model::get_input_name,
    },
};

#[derive(Debug)]
pub struct MulLayer {
    name: String,
    optimization_pattern: PatternRegistry,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
    scaling: u64,
    v_plus_one: usize,
    is_rescale: bool,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for MulLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let is_relu = matches!(self.optimization_pattern, PatternRegistry::MulRelu);
        let a_name = get_input_name(&self.inputs, 0, LayerKind::Mul, INPUT)?;
        let b_name = get_input_name(&self.inputs, 1, LayerKind::Mul, INPUT)?;

        let a_input = load_array_constants_or_get_inputs(
            api,
            input,
            a_name,
            &self.initializer_a,
            LayerKind::Mul,
        )?;

        let b_input = load_array_constants_or_get_inputs(
            api,
            input,
            b_name,
            &self.initializer_b,
            LayerKind::Mul,
        )?;

        let (a_bc, b_bc) = broadcast_two_arrays(&a_input, &b_input)?;

        let result = matrix_hadamard_product(api, &a_bc, b_bc, LayerKind::Mul)?;
        let out = maybe_rescale(
            api,
            result,
            &MaybeRescaleParams {
                is_rescale: self.is_rescale,
                scaling_exponent: self.scaling,
                n_bits: self.v_plus_one,
                is_relu,
                layer_kind: LayerKind::Mul,
                layer_name: self.name.clone(),
            },
        )?;
        Ok((self.outputs.clone(), out))
    }
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let a_name = get_input_name(&layer.inputs, 0, LayerKind::Mul, INPUT)?;
        let b_name = get_input_name(&layer.inputs, 1, LayerKind::Mul, INPUT)?;
        let initializer_a = get_optional_w_or_b(layer_context, a_name)?;
        let initializer_b = get_optional_w_or_b(layer_context, b_name)?;

        let mul = Self {
            name: layer.name.clone(),
            optimization_pattern,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
            scaling: circuit_params.scale_exponent.into(),
            is_rescale,
            v_plus_one: layer_context.n_bits_for(&layer.name),
        };
        Ok(Box::new(mul))
    }
}
