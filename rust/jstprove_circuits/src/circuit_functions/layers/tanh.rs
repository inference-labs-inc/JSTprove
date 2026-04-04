use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{DecomposedExpLookup, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::{exp::compute_exp_quantized, tanh::TANH_HINT_KEY},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct TanhLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TanhLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Tanh, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Tanh,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale = self.scaling;
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale)));
        let cross_tolerance = 3u64 * scale;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
        let two_cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            2 * cross_tolerance,
        )));
        let cross_tol_bits = (2 * cross_tolerance).next_power_of_two().trailing_zeros() as usize;

        let lookup_bits = function_lookup_bits(self.scale_exponent);
        let mut decomposed = DecomposedExpLookup::build::<C, Builder>(
            api,
            lookup_bits,
            scale,
            compute_exp_quantized,
        );

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let hint_out = api.new_hint(TANH_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            let y_plus_scale = api.add(y, scale_var);
            logup_ctx.range_check::<C, Builder>(api, y_plus_scale, self.n_bits + 1)?;
            let scale_minus_y = api.sub(scale_var, y);
            logup_ctx.range_check::<C, Builder>(api, scale_minus_y, self.n_bits + 1)?;

            let two_x = api.add(x, x);
            let exp_2x = decomposed.verify_exp::<C, Builder>(api, logup_ctx, two_x)?;

            let denom = api.add(exp_2x, scale_var);
            let lhs = api.mul(y, denom);
            let rhs = api.sub(exp_2x, scale_var);
            let rhs = api.mul(rhs, scale_var);
            let diff = api.sub(lhs, rhs);
            let shifted = api.add(diff, cross_tol_var);
            logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;
            let upper_shifted = api.sub(two_cross_tol_var, shifted);
            logup_ctx.range_check::<C, Builder>(api, upper_shifted, cross_tol_bits)?;

            out_storage.push(y);
        }

        if !x_input.is_empty() {
            decomposed.finalize::<C, Builder>(api);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Tanh,
                msg: format!("TanhLayer: cannot reshape result into shape {shape:?}"),
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
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Tanh,
                name: "input X".to_string(),
            })?;

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::Tanh,
                msg: format!(
                    "scale_exponent {} >= 32: scale^2 overflows u64 in Tanh cross-multiplication",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Tanh,
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
