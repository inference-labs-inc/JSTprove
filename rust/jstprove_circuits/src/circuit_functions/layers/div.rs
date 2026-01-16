use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use ethnum::U256;
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::gadgets::euclidean_algebra::div_pos_integer_pow2_constant;
use crate::circuit_functions::gadgets::range_check::LogupRangeCheckContext;
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::load_array_constants_or_get_inputs;
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct DivLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
    scaling: u64,
    v_plus_one: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for DivLayer {
    #[allow(clippy::too_many_lines)]
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

        let divisor_constants = self
            .initializer_b
            .as_ref()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Div,
                msg:
                    "Div requires constant divisor (initializer_b). Dynamic divisors not supported."
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

        for &d in &broadcasted_divisors {
            if d <= 0 || d > i64::from(u32::MAX) {
                return Err(LayerError::Other {
                    layer: LayerKind::Div,
                    msg: format!("Divisor must be in range [1, 2^32), got {d}"),
                }
                .into());
            }
            // Power-of-two check
            if (d & (d - 1)) != 0 {
                return Err(LayerError::Other {
                    layer: LayerKind::Div,
                    msg: format!("Divisor must be a power of 2, got {d}"),
                }
                .into());
            }
        }
        let k = usize::try_from(self.scaling).map_err(|_| LayerError::Other {
            layer: LayerKind::Div,
            msg: "Cannot convert scaling to usize".to_string(),
        })?;
        let s =
            self.v_plus_one
                .checked_sub(1)
                .ok_or_else(|| LayerError::InvalidParameterValue {
                    layer: LayerKind::Div,
                    layer_name: self.name.clone(),
                    param_name: "v_plus_one".to_string(),
                    value: self.v_plus_one.to_string(),
                })?;
        let context =
            crate::circuit_functions::utils::quantization::RescalingContext::new(api, k, s)?;

        let mut div_cache: HashMap<u32, Variable> = HashMap::new();
        let mut shift_cache: HashMap<u32, Variable> = HashMap::new();
        let mut scaled_shift_cache: HashMap<u64, Variable> = HashMap::new();

        let result: Result<Vec<Variable>, CircuitError> = a_input
            .iter()
            .zip(&broadcasted_divisors)
            .map(|(&dividend, &divisor_val)| {
                if divisor_val == 1 {
                    return Ok(dividend);
                }
                let div_u32 = u32::try_from(divisor_val)
                    .map_err(|_| CircuitError::Other("divisor is negative or too large".into()))?;
                let div = *div_cache
                    .entry(div_u32)
                    .or_insert_with(|| api.constant(div_u32));

                let remainder_bits = div_u32.trailing_zeros() as usize;

                let shift_amount = u32::try_from(context.shift_exponent).map_err(|_| {
                    crate::circuit_functions::utils::RescaleError::ShiftExponentTooLargeError {
                        exp: context.scaling_exponent,
                        type_name: "u32",
                    }
                })?;

                let shift_ = 1u32.checked_shl(shift_amount).ok_or(
                    crate::circuit_functions::utils::RescaleError::ShiftExponentTooLargeError {
                        exp: context.scaling_exponent,
                        type_name: "u32",
                    },
                )?;
                let shift = *shift_cache
                    .entry(shift_)
                    .or_insert_with(|| api.constant(shift_));
                let scaled_shift_ = u64::from(shift_) * u64::from(div_u32);
                let scaled_shift = *scaled_shift_cache.entry(scaled_shift_).or_insert_with(|| {
                    api.constant(CircuitField::<C>::from_u256(U256::from(scaled_shift_)))
                });

                let out = div_pos_integer_pow2_constant(
                    api,
                    &mut logup_ctx,
                    dividend,
                    div,
                    scaled_shift,
                    remainder_bits,
                    context.shift_exponent,
                    shift,
                )?;

                Ok(out)
            })
            .collect();
        let result = result?;

        logup_ctx.finalize::<C, Builder>(api);

        let output = ArrayD::from_shape_vec(IxDyn(&a_shape), result).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Div,
                msg: "Failed to build result array".to_string(),
            }
        })?;

        Ok((self.outputs.clone(), output))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let dividend_name = get_input_name(&layer.inputs, 0, LayerKind::Div, INPUT)?;
        let divisor_name = get_input_name(&layer.inputs, 1, LayerKind::Div, INPUT)?;
        let initializer_a = get_optional_w_or_b(layer_context, dividend_name)?;
        let initializer_b = get_optional_w_or_b(layer_context, divisor_name)?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
            v_plus_one: layer_context.n_bits,
            scaling: circuit_params.scale_exponent.into(),
        }))
    }
}
