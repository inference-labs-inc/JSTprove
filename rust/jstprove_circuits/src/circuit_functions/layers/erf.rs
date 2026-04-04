use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{DecomposedExpLookup, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::{
        erf::{ERF_ABS_HINT_KEY, ERF_HINT_KEY},
        exp::compute_exp_quantized,
    },
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

const P_COEF: f64 = 0.327_591_1;
const A1: f64 = 0.254_829_592;
const A3: f64 = 1.421_413_741;
const A5: f64 = 1.061_405_429;
const ABS_A2: f64 = 0.284_496_736;
const ABS_A4: f64 = 1.453_152_027;

#[derive(Debug)]
pub struct ErfLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    scale_exponent: u32,
}

#[allow(clippy::too_many_lines)]
fn erf_div_rem<C: Config, B: RootAPI<C>>(
    api: &mut B,
    logup_ctx: &mut LogupRangeCheckContext,
    num: Variable,
    den: Variable,
    rem_bits: usize,
) -> Result<Variable, CircuitError> {
    let q = api.unconstrained_int_div(num, den);
    let r = api.unconstrained_mod(num, den);
    let recon = api.mul(q, den);
    let recon = api.add(recon, r);
    api.assert_is_equal(recon, num);
    logup_ctx.range_check::<C, B>(api, r, rem_bits)?;
    Ok(q)
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ErfLayer {
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
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Erf, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Erf,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale = self.scaling;
        let se = self.scale_exponent as usize;
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale)));
        let scale_sq_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale * scale)));
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let one_var = api.constant(CircuitField::<C>::from_u256(U256::from(1u64)));

        let p_q = (P_COEF * scale as f64).round() as u64;
        let a1_q = (A1 * scale as f64).round() as u64;
        let a3_q = (A3 * scale as f64).round() as u64;
        let a5_q = (A5 * scale as f64).round() as u64;
        let abs_a2_q = (ABS_A2 * scale as f64).round() as u64;
        let abs_a4_q = (ABS_A4 * scale as f64).round() as u64;

        let p_var = api.constant(CircuitField::<C>::from_u256(U256::from(p_q)));
        let a1_var = api.constant(CircuitField::<C>::from_u256(U256::from(a1_q)));
        let a3_var = api.constant(CircuitField::<C>::from_u256(U256::from(a3_q)));
        let a5_var = api.constant(CircuitField::<C>::from_u256(U256::from(a5_q)));
        let abs_a2_var = api.constant(CircuitField::<C>::from_u256(U256::from(abs_a2_q)));
        let abs_a4_var = api.constant(CircuitField::<C>::from_u256(U256::from(abs_a4_q)));

        let cross_tolerance = 4u64 * scale;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
        let two_cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            2 * cross_tolerance,
        )));
        let cross_tol_bits = (2 * cross_tolerance).next_power_of_two().trailing_zeros() as usize;

        let t_tolerance = 2u64 * scale;
        let t_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(t_tolerance)));
        let two_t_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(2 * t_tolerance)));
        let t_tol_bits = (2 * t_tolerance).next_power_of_two().trailing_zeros() as usize;

        let lookup_bits = function_lookup_bits(self.scale_exponent);
        let mut decomposed = DecomposedExpLookup::build::<C, Builder>(
            api,
            lookup_bits,
            scale,
            compute_exp_quantized,
        );

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let hint_out = api.new_hint(ERF_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            let abs_hint = api.new_hint(ERF_ABS_HINT_KEY, &[x], 2);
            let abs_x = abs_hint[0];
            let is_nonneg = abs_hint[1];

            api.assert_is_bool(is_nonneg);
            let two_is_nonneg = api.add(is_nonneg, is_nonneg);
            let sign = api.sub(two_is_nonneg, one_var);
            let signed_abs = api.mul(sign, abs_x);
            api.assert_is_equal(signed_abs, x);
            logup_ctx.range_check::<C, Builder>(api, abs_x, self.n_bits)?;

            let p_abs = api.mul(p_var, abs_x);
            let p_scaled = erf_div_rem::<C, Builder>(api, logup_ctx, p_abs, scale_var, se)?;
            let denom = api.add(scale_var, p_scaled);

            let t = erf_div_rem::<C, Builder>(api, logup_ctx, scale_sq_var, denom, se + 1)?;
            logup_ctx.range_check::<C, Builder>(api, t, self.n_bits)?;
            let t_denom = api.mul(t, denom);
            let t_diff = api.sub(t_denom, scale_sq_var);
            let t_shifted = api.add(t_diff, t_tol_var);
            logup_ctx.range_check::<C, Builder>(api, t_shifted, t_tol_bits)?;
            let t_upper = api.sub(two_t_tol_var, t_shifted);
            logup_ctx.range_check::<C, Builder>(api, t_upper, t_tol_bits)?;

            let t_sq = api.mul(t, t);
            let t_sq_div = erf_div_rem::<C, Builder>(api, logup_ctx, t_sq, scale_var, se)?;

            let a5_tsq = api.mul(a5_var, t_sq_div);
            let a5_tsq_div = erf_div_rem::<C, Builder>(api, logup_ctx, a5_tsq, scale_var, se)?;
            let pos_inner2 = api.add(a3_var, a5_tsq_div);

            let pi2_tsq = api.mul(pos_inner2, t_sq_div);
            let pi2_tsq_div = erf_div_rem::<C, Builder>(api, logup_ctx, pi2_tsq, scale_var, se)?;
            let pos_inner1 = api.add(a1_var, pi2_tsq_div);

            let pos_t = api.mul(t, pos_inner1);
            let pos_val = erf_div_rem::<C, Builder>(api, logup_ctx, pos_t, scale_var, se)?;

            let a4_tsq = api.mul(abs_a4_var, t_sq_div);
            let a4_tsq_div = erf_div_rem::<C, Builder>(api, logup_ctx, a4_tsq, scale_var, se)?;
            let neg_inner = api.add(abs_a2_var, a4_tsq_div);

            let neg_tsq = api.mul(neg_inner, t_sq_div);
            let neg_val = erf_div_rem::<C, Builder>(api, logup_ctx, neg_tsq, scale_var, se)?;

            let abs_x_sq = api.mul(abs_x, abs_x);
            let x_sq = erf_div_rem::<C, Builder>(api, logup_ctx, abs_x_sq, scale_var, se)?;
            let neg_x_sq = api.sub(zero_var, x_sq);
            let exp_neg_x_sq = decomposed.verify_exp::<C, Builder>(api, logup_ctx, neg_x_sq)?;

            let pos_exp_prod = api.mul(pos_val, exp_neg_x_sq);
            let pos_exp = erf_div_rem::<C, Builder>(api, logup_ctx, pos_exp_prod, scale_var, se)?;

            let neg_exp_prod = api.mul(neg_val, exp_neg_x_sq);
            let neg_exp = erf_div_rem::<C, Builder>(api, logup_ctx, neg_exp_prod, scale_var, se)?;

            let erf_abs = api.sub(scale_var, pos_exp);
            let erf_abs = api.add(erf_abs, neg_exp);

            let erf_signed = api.mul(sign, erf_abs);

            let diff = api.sub(y, erf_signed);
            let shifted = api.add(diff, cross_tol_var);
            logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;
            let upper = api.sub(two_cross_tol_var, shifted);
            logup_ctx.range_check::<C, Builder>(api, upper, cross_tol_bits)?;

            let y_plus_scale = api.add(y, scale_var);
            logup_ctx.range_check::<C, Builder>(api, y_plus_scale, self.n_bits + 1)?;
            let scale_minus_y = api.sub(scale_var, y);
            logup_ctx.range_check::<C, Builder>(api, scale_minus_y, self.n_bits + 1)?;

            out_storage.push(y);
        }

        if !x_input.is_empty() {
            decomposed.finalize::<C, Builder>(api);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Erf,
                msg: format!("ErfLayer: cannot reshape result into shape {shape:?}"),
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
                layer: LayerKind::Erf,
                msg: format!("Erf expects exactly 1 input, got {}", layer.inputs.len()),
            }
            .into());
        }
        if layer.outputs.len() != 1 {
            return Err(LayerError::Other {
                layer: LayerKind::Erf,
                msg: format!("Erf expects exactly 1 output, got {}", layer.outputs.len()),
            }
            .into());
        }

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::Erf,
                msg: format!(
                    "scale_exponent {} >= 32: scale^2 overflows u64 in Erf polynomial verification",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Erf,
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
