use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::gadgets::LogupRangeCheckContext;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::load_array_constants_or_get_inputs;
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

#[derive(Debug)]
pub struct DivLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
    n_bits: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for DivLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let a_name = get_input_name(&self.inputs, 0, LayerKind::Div, INPUT)?;

        let a_input = load_array_constants_or_get_inputs(
            api,
            &input,
            a_name,
            &self.initializer_a,
            LayerKind::Div,
        )?;

        let divisor_const = self.initializer_b.as_ref().ok_or_else(|| LayerError::Other {
            layer: LayerKind::Div,
            msg: "Div requires constant divisor (initializer_b)".to_string(),
        })?;

        let divisor_val = divisor_const.iter().next().copied().unwrap_or(1) as u32;
        if divisor_val == 0 {
            return Err(CircuitError::from(LayerError::Other {
                layer: LayerKind::Div,
                msg: "Division by zero".to_string(),
            }));
        }

        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let divisor_var = api.constant(divisor_val);
        let shape = a_input.shape().to_vec();

        let results: Result<Vec<Variable>, CircuitError> = a_input
            .iter()
            .map(|dividend| {
                let quotient = api.unconstrained_int_div(*dividend, divisor_val);
                let remainder = api.unconstrained_mod(*dividend, divisor_val);

                let prod = api.mul(divisor_var, quotient);
                let reconstructed = api.add(prod, remainder);
                api.assert_is_equal(*dividend, reconstructed);

                let remainder_bits = self.n_bits.saturating_sub(1);
                if remainder_bits > 0 {
                    let _ = logup_ctx.range_check::<C, Builder>(api, remainder, remainder_bits);
                }

                Ok(quotient)
            })
            .collect();

        logup_ctx.finalize::<C, Builder>(api);

        let out = ArrayD::from_shape_vec(shape, results?).map_err(|e| LayerError::Other {
            layer: LayerKind::Div,
            msg: format!("Failed to reshape result: {e}"),
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
        let (_, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Div,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let initializer_a = get_optional_w_or_b(layer_context, &layer.inputs[0])?;
        let initializer_b = get_optional_w_or_b(layer_context, &layer.inputs[1])?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
            n_bits: layer_context.n_bits,
        }))
    }
}
