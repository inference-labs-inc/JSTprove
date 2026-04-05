use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{ShiftRangeContext, constrained_clip, range_check::LogupRangeCheckContext},
    hints::hardswish::HARDSWISH_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct HardSwishLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    #[allow(dead_code)]
    n_bits: usize,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for HardSwishLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::HardSwish, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::HardSwish,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let alpha = self.scaling;
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(alpha)));

        let three_alpha = 3u64 * alpha;
        let three_alpha_var = api.constant(CircuitField::<C>::from_u256(U256::from(three_alpha)));

        let six_alpha = 6u64 * alpha;
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let six_alpha_var = api.constant(CircuitField::<C>::from_u256(U256::from(six_alpha)));
        let six_var = api.constant(CircuitField::<C>::from_u256(U256::from(6u64)));

        let clip_shift_exponent = self.scale_exponent as usize + 3;
        let shift_ctx =
            ShiftRangeContext::new::<C, Builder>(api, LayerKind::HardSwish, clip_shift_exponent)?;

        let cross_tolerance = 6u64 * alpha;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
        let two_cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            2 * cross_tolerance,
        )));
        let cross_tol_bits = (2 * cross_tolerance).next_power_of_two().trailing_zeros() as usize;

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let hint_out = api.new_hint(HARDSWISH_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            let x_plus_3 = api.add(x, three_alpha_var);
            let clipped = constrained_clip(
                api,
                &shift_ctx,
                logup_ctx,
                x_plus_3,
                Some(zero_var),
                Some(six_alpha_var),
            )?;
            let product = api.mul(x, clipped);

            let six_y_alpha = api.mul(six_var, y);
            let six_y_alpha = api.mul(six_y_alpha, scale_var);

            let diff = api.sub(six_y_alpha, product);
            let shifted = api.add(diff, cross_tol_var);
            logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;
            let upper = api.sub(two_cross_tol_var, shifted);
            logup_ctx.range_check::<C, Builder>(api, upper, cross_tol_bits)?;

            out_storage.push(y);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::HardSwish,
                msg: format!("HardSwishLayer: cannot reshape result into shape {shape:?}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        if layer.inputs.len() != 1 {
            return Err(LayerError::Other {
                layer: LayerKind::HardSwish,
                msg: format!(
                    "HardSwish expects exactly 1 input, got {}",
                    layer.inputs.len()
                ),
            }
            .into());
        }

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::HardSwish,
                msg: format!(
                    "scale_exponent {} >= 32: overflows u64 in HardSwish verification",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::HardSwish,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits,
            scaling,
            scale_exponent: circuit_params.scale_exponent,
        }))
    }
}
