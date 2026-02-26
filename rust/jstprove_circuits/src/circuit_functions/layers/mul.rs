use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::gadgets::linear_algebra::matrix_hadamard_product;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::quantization::rescale_array;
use crate::circuit_functions::utils::tensor_ops::{
    broadcast_two_arrays, load_array_constants_or_get_inputs,
};
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
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

        // Matrix hadammard product with optional rescaling
        let result = matrix_hadamard_product(api, &a_bc, b_bc, LayerKind::Mul)?;
        if self.is_rescale {
            let k = usize::try_from(self.scaling).map_err(|_| LayerError::Other {
                layer: LayerKind::Mul,
                msg: "Cannot convert scaling to usize".to_string(),
            })?;
            let s = self.v_plus_one.checked_sub(1).ok_or_else(|| {
                LayerError::InvalidParameterValue {
                    layer: LayerKind::Mul,
                    layer_name: self.name.clone(),
                    param_name: "v_plus_one".to_string(),
                    value: self.v_plus_one.to_string(),
                }
            })?;
            let out_array =
                rescale_array(api, result, k, s, is_relu).map_err(|e| LayerError::Other {
                    layer: LayerKind::Mul,
                    msg: format!("Rescale failed: {e}"),
                })?;
            return Ok((self.outputs.clone(), out_array));
        }
        Ok((self.outputs.clone(), result))
    }
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let a_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Mul,
                name: "input A".to_string(),
            })?;
        let b_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Mul,
                name: "input B".to_string(),
            })?;
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
