use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

use crate::circuit_functions::gadgets::linear_algebra::{matrix_addition, matrix_hadamard_product};
use crate::circuit_functions::utils::tensor_ops::{
    broadcast_two_arrays, reshape_channel_vector_for_broadcast,
};
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        graph_pattern_matching::PatternRegistry,
        onnx_model::{get_input_name, get_w_or_b},
        quantization::rescale_array,
        tensor_ops::load_array_constants_or_get_inputs,
    },
};
/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct BatchnormLayer {
    name: String,
    index: usize,
    weights: Option<ArrayD<i64>>,
    bias: Option<ArrayD<i64>>,
    is_rescale: bool,
    v_plus_one: usize,
    optimization_pattern: PatternRegistry,
    scaling: u64,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementation --------
impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for BatchnormLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // TODO can add bias check as an optional step.
        let is_relu = matches!(self.optimization_pattern, PatternRegistry::BatchnormRelu);
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Batchnorm, INPUT)?;

        let layer_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Batchnorm,
                name: input_name.to_string(),
            })?
            .clone();

        let w_name = get_input_name(&self.inputs, 1, LayerKind::Batchnorm, "weights")?;
        let mul_raw = load_array_constants_or_get_inputs(
            api,
            &input,
            w_name,
            &self.weights,
            LayerKind::Batchnorm,
        )?;
        let b_name = get_input_name(&self.inputs, 2, LayerKind::Batchnorm, "bias")?;
        let add_raw = load_array_constants_or_get_inputs(
            api,
            &input,
            b_name,
            &self.bias,
            LayerKind::Batchnorm,
        )?;

        let mul_input = reshape_channel_vector_for_broadcast(&mul_raw, &layer_input)?;
        let add_input = reshape_channel_vector_for_broadcast(&add_raw, &layer_input)?;

        // Matrix hadammard product with optional rescaling
        let (a_mul_bc, b_mul_bc) = broadcast_two_arrays(&layer_input, &mul_input)?;
        let mul_out = matrix_hadamard_product(api, &a_mul_bc, b_mul_bc, LayerKind::Batchnorm)?;

        // Matrix addition with optional rescaling
        let (a_add_bc, b_add_bc) = broadcast_two_arrays(&mul_out, &add_input)?;
        let result = matrix_addition(api, &a_add_bc, b_add_bc, LayerKind::Batchnorm)?;

        if self.is_rescale {
            let k = usize::try_from(self.scaling).map_err(|_| LayerError::Other {
                layer: LayerKind::Batchnorm,
                msg: "Cannot convert scaling to usize".to_string(),
            })?;
            let s = self.v_plus_one.checked_sub(1).ok_or_else(|| {
                LayerError::InvalidParameterValue {
                    layer: LayerKind::Batchnorm,
                    layer_name: self.name.clone(),
                    param_name: "v_plus_one".to_string(),
                    value: self.v_plus_one.to_string(),
                }
            })?;
            let out_array =
                rescale_array(api, result, k, s, is_relu).map_err(|e| LayerError::Other {
                    layer: LayerKind::Batchnorm,
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
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (weights, bias) = if layer_context.weights_as_inputs {
            (None, None)
        } else {
            (
                Some(get_w_or_b(
                    &layer_context.w_and_b_map,
                    get_input_name(&layer.inputs, 1, LayerKind::Batchnorm, "weights")?,
                )?),
                Some(get_w_or_b(
                    &layer_context.w_and_b_map,
                    get_input_name(&layer.inputs, 2, LayerKind::Batchnorm, "bias")?,
                )?),
            )
        };

        let batchnorm = Self {
            name: layer.name.clone(),
            index,
            weights,
            bias,
            is_rescale,
            v_plus_one: layer_context.n_bits,
            optimization_pattern,
            scaling: circuit_params.scale_exponent.into(), // TODO: Becomes scaling_in?
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };
        Ok(Box::new(batchnorm))
    }
}
