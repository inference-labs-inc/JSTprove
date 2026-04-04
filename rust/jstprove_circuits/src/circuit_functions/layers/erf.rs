use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{
        FunctionLookupTable, ShiftRangeContext, constrained_clip, function_lookup::i64_to_field,
        range_check::LogupRangeCheckContext,
    },
    hints::erf::{ERF_HINT_KEY, compute_erf_quantized},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

const ERF_DOWNSAMPLE: i64 = 8;
const ERF_TABLE_BITS: usize = 18;

#[derive(Debug)]
pub struct ErfLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    #[allow(dead_code)]
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ErfLayer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale)));

        let downsample = ERF_DOWNSAMPLE;
        let downsample_var =
            api.constant(CircuitField::<C>::from_u256(U256::from(downsample as u64)));
        let half = 1i64 << (ERF_TABLE_BITS - 1);
        let half_shifted = half * downsample;
        let half_shifted_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            half_shifted as u64,
        )));
        let offset_var = api.constant(CircuitField::<C>::from_u256(U256::from(half as u64)));
        let downsample_rem_bits = {
            let mut b = 0usize;
            let mut v = downsample;
            while v > 1 {
                v >>= 1;
                b += 1;
            }
            b
        };

        let limit = (half - 1) * downsample;
        let limit_var = api.constant(i64_to_field::<C>(limit));
        let neg_limit_var = api.constant(i64_to_field::<C>(-half_shifted));

        let clamp_shift_exp = self.n_bits + 2;
        let clamp_ctx = ShiftRangeContext::new(api, LayerKind::Erf, clamp_shift_exp)?;

        let mut erf_table = FunctionLookupTable::build_signed::<C, Builder>(
            api,
            |k, s| compute_erf_quantized(k * downsample, s),
            ERF_TABLE_BITS,
            scale,
        );

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let x_clamped = constrained_clip(
                api,
                &clamp_ctx,
                logup_ctx,
                x,
                Some(neg_limit_var),
                Some(limit_var),
            )?;

            let x_shifted = api.add(x_clamped, half_shifted_var);

            let x_shifted_idx = api.unconstrained_int_div(x_shifted, downsample_var);
            let x_shifted_rem = api.unconstrained_mod(x_shifted, downsample_var);
            let recon = api.mul(x_shifted_idx, downsample_var);
            let recon = api.add(recon, x_shifted_rem);
            api.assert_is_equal(recon, x_shifted);
            logup_ctx.range_check::<C, Builder>(api, x_shifted_rem, downsample_rem_bits)?;

            let x_signed_idx = api.sub(x_shifted_idx, offset_var);

            let x_rounded = api.mul(x_signed_idx, downsample_var);
            let erf_hint_out = api.new_hint(ERF_HINT_KEY, &[x_rounded, scale_var], 1);
            let erf_val = erf_hint_out[0];

            erf_table.query(x_signed_idx, erf_val);

            let y_plus_scale = api.add(erf_val, scale_var);
            logup_ctx.range_check::<C, Builder>(api, y_plus_scale, self.n_bits)?;
            let scale_minus_y = api.sub(scale_var, erf_val);
            logup_ctx.range_check::<C, Builder>(api, scale_minus_y, self.n_bits)?;

            out_storage.push(erf_val);
        }

        if !x_input.is_empty() {
            erf_table.finalize::<C, Builder>(api);
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
