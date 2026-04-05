use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, ShiftRangeContext, constrained_relu},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param_or_default},
    },
};

use crate::circuit_functions::gadgets::euclidean_division::div_pos_integer_pow2_constant;
use crate::circuit_functions::utils::quantization::RescalingContext;

#[derive(Debug)]
pub struct LeakyReluLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    alpha_q: i64,
    #[allow(dead_code)]
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LeakyReluLayer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::LeakyRelu, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::LeakyRelu,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();

        let shift_exponent = self.n_bits.checked_sub(1).ok_or_else(|| {
            CircuitError::Other("LeakyReLU: n_bits must be at least 1".to_string())
        })?;
        let shift_ctx =
            ShiftRangeContext::new::<C, Builder>(api, LayerKind::LeakyRelu, shift_exponent)?;

        let alpha_q_var = if self.alpha_q >= 0 {
            api.constant(CircuitField::<C>::from_u256(U256::from(
                self.alpha_q as u64,
            )))
        } else {
            let mag = U256::from(self.alpha_q.unsigned_abs());
            api.constant(CircuitField::<C>::from_u256(
                CircuitField::<C>::MODULUS - mag,
            ))
        };

        let rescale_ctx =
            RescalingContext::new::<C, Builder>(api, self.scale_exponent as usize, shift_exponent)?;

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let relu_x = constrained_relu::<C, Builder>(api, &shift_ctx, logup_ctx, x)?;

            let neg_part = api.sub(x, relu_x);

            let product = api.mul(alpha_q_var, neg_part);

            let alpha_neg = div_pos_integer_pow2_constant(
                api,
                logup_ctx,
                product,
                rescale_ctx.scaling_factor,
                rescale_ctx.scaled_shift,
                rescale_ctx.scaling_exponent,
                rescale_ctx.shift_exponent,
                rescale_ctx.shift,
            )?;

            let y = api.add(relu_x, alpha_neg);

            out_storage.push(y);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::LeakyRelu,
                msg: format!("LeakyReluLayer: cannot reshape result into shape {shape:?}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::LeakyRelu,
                name: "input X".to_string(),
            })?;

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::LeakyRelu,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::LeakyRelu,
                msg: format!(
                    "scale_exponent {} >= 32: scaling as f64 loses precision in alpha_q calculation",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::LeakyRelu,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let alpha_f64: f64 = get_param_or_default(&layer.name, "alpha", &params, Some(&0.01_f64))
            .map_err(|e| LayerError::Other {
            layer: LayerKind::LeakyRelu,
            msg: format!("failed to read 'alpha' attribute: {e}"),
        })?;

        let alpha_q = (alpha_f64 * scaling as f64).round() as i64;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits,
            alpha_q,
            scaling,
            scale_exponent: circuit_params.scale_exponent,
        }))
    }
}
