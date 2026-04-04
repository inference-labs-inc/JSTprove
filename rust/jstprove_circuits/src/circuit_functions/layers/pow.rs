use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
    },
};

#[derive(Debug)]
pub struct PowLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
    scale_exponent: u32,
    n_bits: usize,
    const_exponents: Vec<i32>,
}

fn constrained_pow_integer<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    logup_ctx: &mut LogupRangeCheckContext,
    x: Variable,
    exp: i32,
    scale_var: Variable,
    scale_exponent: u32,
) -> Result<Variable, CircuitError> {
    if exp == 0 {
        return Ok(scale_var);
    }
    if exp == 1 {
        return Ok(x);
    }

    let abs_exp = exp.unsigned_abs();
    let mut result = scale_var;
    let mut base = x;
    let mut e = abs_exp;

    while e > 0 {
        if e & 1 == 1 {
            let product = api.mul(result, base);
            result = api.unconstrained_int_div(product, scale_var);
            let rem = api.unconstrained_mod(product, scale_var);
            let recon = api.mul(result, scale_var);
            let recon = api.add(recon, rem);
            api.assert_is_equal(recon, product);
            logup_ctx.range_check::<C, Builder>(api, rem, scale_exponent as usize)?;
        }
        e >>= 1;
        if e > 0 {
            let sq = api.mul(base, base);
            base = api.unconstrained_int_div(sq, scale_var);
            let sq_rem = api.unconstrained_mod(sq, scale_var);
            let sq_recon = api.mul(base, scale_var);
            let sq_recon = api.add(sq_recon, sq_rem);
            api.assert_is_equal(sq_recon, sq);
            logup_ctx.range_check::<C, Builder>(api, sq_rem, scale_exponent as usize)?;
        }
    }

    if exp < 0 {
        let scale_sq = api.mul(scale_var, scale_var);
        let inv = api.unconstrained_int_div(scale_sq, result);
        let inv_rem = api.unconstrained_mod(scale_sq, result);
        let inv_recon = api.mul(inv, result);
        let inv_recon = api.add(inv_recon, inv_rem);
        api.assert_is_equal(inv_recon, scale_sq);
        result = inv;
    }

    Ok(result)
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for PowLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Pow, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Pow,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for (i, &x) in x_input.iter().enumerate() {
            let exp = if self.const_exponents.len() == 1 {
                self.const_exponents[0]
            } else {
                self.const_exponents[i]
            };

            let y = constrained_pow_integer::<C, Builder>(
                api,
                logup_ctx,
                x,
                exp,
                scale_var,
                self.scale_exponent,
            )?;

            logup_ctx.range_check::<C, Builder>(api, y, self.n_bits)?;

            out_storage.push(y);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Pow,
                msg: format!("PowLayer: cannot reshape result into shape {shape:?}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
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
                layer: LayerKind::Pow,
                name: "input X".to_string(),
            })?;

        let const_exponent = if let Some(exp_name) = layer.inputs.get(1) {
            get_w_or_b::<i64, _>(layer_context.w_and_b_map, exp_name).ok()
        } else {
            None
        };

        let const_exponent = const_exponent.ok_or_else(|| LayerError::Other {
            layer: LayerKind::Pow,
            msg: "Pow requires a compile-time constant integer exponent for sound \
                  constraint verification; variable or fractional exponents are not supported"
                .to_string(),
        })?;

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Pow,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        let scale_f64 = scaling as f64;
        let mut const_exponents = Vec::new();
        for &val in &const_exponent {
            let real_exp = val as f64 / scale_f64;
            let int_exp = real_exp.round();
            if (real_exp - int_exp).abs() > 1e-6 {
                return Err(LayerError::Other {
                    layer: LayerKind::Pow,
                    msg: format!(
                        "Pow exponent {real_exp} is not an integer; only integer exponents \
                         are supported for sound constraint verification"
                    ),
                }
                .into());
            }
            let int_exp = int_exp as i32;
            if int_exp.unsigned_abs() > 16 {
                return Err(LayerError::Other {
                    layer: LayerKind::Pow,
                    msg: format!(
                        "Pow exponent {int_exp} exceeds supported range [-16, 16]; \
                         large exponents risk field overflow"
                    ),
                }
                .into());
            }
            const_exponents.push(int_exp);
        }

        if const_exponents.is_empty() {
            return Err(LayerError::Other {
                layer: LayerKind::Pow,
                msg: "Pow exponent tensor is empty".to_string(),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            scaling,
            scale_exponent: circuit_params.scale_exponent,
            n_bits,
            const_exponents,
        }))
    }
}
