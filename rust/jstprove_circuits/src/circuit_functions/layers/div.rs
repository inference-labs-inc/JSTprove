use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use ethnum::U256;
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::gadgets::range_check::LogupRangeCheckContext;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::load_array_constants_or_get_inputs;
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        graph_pattern_matching::PatternRegistry,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

const SHIFT_BITS: usize = 20;

#[allow(dead_code)]
#[derive(Debug)]
pub struct DivLayer {
    name: String,
    optimization_pattern: PatternRegistry,
    input_shape: Vec<usize>,
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

        let divisor_constants = self.initializer_b.as_ref().ok_or_else(|| LayerError::Other {
            layer: LayerKind::Div,
            msg: "Div requires constant divisor (initializer_b). Dynamic divisors not supported."
                .to_string(),
        })?;

        let a_shape = a_input.shape().to_vec();

        let broadcasted_divisors = divisor_constants
            .broadcast(IxDyn(&a_shape))
            .ok_or_else(|| LayerError::ShapeMismatch {
                layer: LayerKind::Div,
                expected: a_shape.clone(),
                got: divisor_constants.shape().to_vec(),
                var_name: "divisor".to_string(),
            })?
            .to_owned();

        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        for &d in broadcasted_divisors.iter() {
            if d <= 0 {
                return Err(LayerError::Other {
                    layer: LayerKind::Div,
                    msg: format!("Divisor must be positive, got {d}"),
                }
                .into());
            }
        }

        let result: Vec<Variable> = a_input
            .iter()
            .zip(broadcasted_divisors.iter())
            .map(|(&dividend, &divisor_val)| {
                let divisor_u32 = divisor_val as u32;

                let base_shift: u64 = 1u64 << SHIFT_BITS;
                let shift = base_shift * u64::from(divisor_u32);
                let shift_var = api.constant(CircuitField::<C>::from_u256(U256::from(shift)));
                let shifted_dividend = api.add(dividend, shift_var);

                let shifted_quotient = api.unconstrained_int_div(shifted_dividend, divisor_u32);
                let remainder = api.unconstrained_mod(shifted_dividend, divisor_u32);

                let divisor_var = api.constant(divisor_u32);
                let product = api.mul(divisor_var, shifted_quotient);
                let reconstructed = api.add(product, remainder);
                api.assert_is_equal(shifted_dividend, reconstructed);

                let divisor_minus_one = api.constant(divisor_u32 - 1);
                let remainder_bound = api.sub(divisor_minus_one, remainder);
                let divisor_bits = (32 - divisor_u32.leading_zeros()) as usize;
                logup_ctx
                    .range_check::<C, Builder>(api, remainder_bound, divisor_bits)
                    .expect("Range check failed: remainder >= divisor");

                let base_shift_var =
                    api.constant(CircuitField::<C>::from_u256(U256::from(base_shift)));
                let quotient_floor = api.sub(shifted_quotient, base_shift_var);

                let is_neg = api.unconstrained_lesser(shifted_dividend, shift_var);
                let one = api.constant(1u32);
                let is_rem_zero = api.is_zero(remainder);
                let has_rem = api.sub(one, is_rem_zero);
                let correction = api.mul(is_neg, has_rem);

                let quotient_trunc = api.add(quotient_floor, correction);

                let neg_diff = api.sub(shift_var, shifted_dividend);
                let pos_diff = api.sub(shifted_dividend, shift_var);

                let neg_check = api.mul(neg_diff, is_neg);
                let is_not_neg = api.sub(one, is_neg);
                let pos_check = api.mul(pos_diff, is_not_neg);

                logup_ctx
                    .range_check::<C, Builder>(api, neg_check, SHIFT_BITS + 4)
                    .expect("Range check failed for negative diff");
                logup_ctx
                    .range_check::<C, Builder>(api, pos_check, SHIFT_BITS + 4)
                    .expect("Range check failed for positive diff");

                quotient_trunc
            })
            .collect();

        logup_ctx.finalize::<C, Builder>(api);

        let output =
            ArrayD::from_shape_vec(IxDyn(&a_shape), result).map_err(|_| LayerError::InvalidShape {
                layer: LayerKind::Div,
                msg: "Failed to build result array".to_string(),
            })?;

        Ok((self.outputs.clone(), output))
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
                layer: LayerKind::Div,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;

        let initializer_a = get_optional_w_or_b(layer_context, &layer.inputs[0])?;
        let initializer_b = get_optional_w_or_b(layer_context, &layer.inputs[1])?;

        let div = Self {
            name: layer.name.clone(),
            optimization_pattern,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
            n_bits: layer_context.n_bits,
        };
        Ok(Box::new(div))
    }
}
