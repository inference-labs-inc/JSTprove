use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{DecomposedExpLookup, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::{exp::compute_exp_quantized, gelu::GELU_HINT_KEY},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
};

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
const GELU_COEF: f64 = 0.044_715;

#[derive(Debug)]
pub struct GeluLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GeluLayer {
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Gelu, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Gelu,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale = self.scaling;
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale)));
        let scale_sq_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale * scale)));
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));

        let coef_q = (GELU_COEF * scale as f64).round() as u64;
        let coef_var = api.constant(CircuitField::<C>::from_u256(U256::from(coef_q)));

        let outer_coef_q = (2.0 * SQRT_2_OVER_PI * scale as f64).round() as u64;
        let outer_coef_var = api.constant(CircuitField::<C>::from_u256(U256::from(outer_coef_q)));

        let cross_tolerance = 5u64 * scale;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
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
            let hint_out = api.new_hint(GELU_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            let x_sq = api.mul(x, x);
            let x_sq_div_alpha = api.unconstrained_int_div(x_sq, scale_var);
            let x_sq_rem = api.unconstrained_mod(x_sq, scale_var);
            let x_sq_recon = api.mul(x_sq_div_alpha, scale_var);
            let x_sq_recon = api.add(x_sq_recon, x_sq_rem);
            api.assert_is_equal(x_sq_recon, x_sq);
            logup_ctx.range_check::<C, Builder>(api, x_sq_rem, self.scale_exponent as usize)?;

            let x_cubed_a2 = api.mul(x_sq_div_alpha, x);

            let coef_x3 = api.mul(coef_var, x_cubed_a2);
            let coef_x3_div = api.unconstrained_int_div(coef_x3, scale_sq_var);
            let coef_x3_rem = api.unconstrained_mod(coef_x3, scale_sq_var);
            let coef_x3_recon = api.mul(coef_x3_div, scale_sq_var);
            let coef_x3_recon = api.add(coef_x3_recon, coef_x3_rem);
            api.assert_is_equal(coef_x3_recon, coef_x3);
            logup_ctx.range_check::<C, Builder>(
                api,
                coef_x3_rem,
                2 * self.scale_exponent as usize,
            )?;

            let inner = api.add(x, coef_x3_div);

            let z_a2 = api.mul(outer_coef_var, inner);
            let z = api.unconstrained_int_div(z_a2, scale_var);
            let z_rem = api.unconstrained_mod(z_a2, scale_var);
            let z_recon = api.mul(z, scale_var);
            let z_recon = api.add(z_recon, z_rem);
            api.assert_is_equal(z_recon, z_a2);
            logup_ctx.range_check::<C, Builder>(api, z_rem, self.scale_exponent as usize)?;

            let neg_z = api.sub(zero_var, z);
            let exp_neg_z = decomposed.verify_exp::<C, Builder>(api, logup_ctx, neg_z)?;

            let denom = api.add(scale_var, exp_neg_z);
            let sigmoid_times_denom = api.mul(scale_var, denom);

            let x_times_scale = api.mul(x, scale_var);
            let lhs = api.mul(y, denom);
            let diff = api.sub(lhs, x_times_scale);
            let shifted = api.add(diff, cross_tol_var);
            logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;

            let _ = sigmoid_times_denom;

            out_storage.push(y);
        }

        if !x_input.is_empty() {
            decomposed.finalize::<C, Builder>(api);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Gelu,
                msg: format!("GeluLayer: cannot reshape result into shape {shape:?}"),
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
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Gelu,
                name: "input X".to_string(),
            })?;

        let params = extract_params(layer).ok();
        let default_approximate = "none".to_string();
        let approximate: String = params
            .as_ref()
            .and_then(|p| {
                get_param_or_default(&layer.name, "approximate", p, Some(&default_approximate)).ok()
            })
            .unwrap_or(default_approximate);

        if approximate != "tanh" {
            return Err(LayerError::Other {
                layer: LayerKind::Gelu,
                msg: format!(
                    "Gelu approximate='{approximate}' is not supported in the Expander backend: \
                     only approximate='tanh' is implemented."
                ),
            }
            .into());
        }

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::Gelu,
                msg: format!(
                    "scale_exponent {} >= 32: scale^2 overflows u64 in GELU polynomial verification",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Gelu,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            scaling,
            scale_exponent: circuit_params.scale_exponent,
        }))
    }
}
