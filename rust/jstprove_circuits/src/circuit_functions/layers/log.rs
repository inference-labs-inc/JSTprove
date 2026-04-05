use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{DecomposedExpLookup, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::{exp::compute_exp_quantized, log::LOG_HINT_KEY},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct LogLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LogLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Log, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Log,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let cross_tolerance = 2u64 * self.scaling;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
        let double_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            2u64 * cross_tolerance,
        )));
        let cross_tol_bits = (2 * cross_tolerance).next_power_of_two().trailing_zeros() as usize;

        let lookup_bits = function_lookup_bits(self.scale_exponent);
        let mut decomposed = DecomposedExpLookup::build::<C, Builder>(
            api,
            lookup_bits,
            self.scaling,
            compute_exp_quantized,
        );

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let hint_out = api.new_hint(LOG_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            logup_ctx.range_check::<C, Builder>(api, x, self.n_bits)?;

            let one_var = api.constant(CircuitField::<C>::from_u256(U256::from(1u64)));
            let x_minus_one = api.sub(x, one_var);
            logup_ctx.range_check::<C, Builder>(api, x_minus_one, self.n_bits)?;

            let exp_y = decomposed.verify_exp::<C, Builder>(api, logup_ctx, y)?;

            let lhs = api.mul(exp_y, scale_var);
            let rhs = api.mul(x, scale_var);
            let diff = api.sub(lhs, rhs);
            let shifted = api.add(diff, cross_tol_var);
            logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;
            let complement = api.sub(double_tol_var, shifted);
            logup_ctx.range_check::<C, Builder>(api, complement, cross_tol_bits)?;

            out_storage.push(y);
        }

        if !x_input.is_empty() {
            decomposed.finalize::<C, Builder>(api);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Log,
                msg: format!("LogLayer: cannot reshape result into shape {shape:?}"),
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
                layer: LayerKind::Log,
                name: "input X".to_string(),
            })?;

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::Log,
                msg: format!(
                    "scale_exponent {} is too large (must be < 32)",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Log,
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

#[cfg(test)]
mod tests {
    use crate::circuit_functions::hints::log::log_hint;
    use ethnum::U256;
    use expander_compiler::field::{BN254Fr, FieldArith};

    type F = BN254Fr;

    fn field(n: i64) -> F {
        if n >= 0 {
            F::from_u256(U256::from(n as u64))
        } else {
            let mag = U256::from(n.unsigned_abs());
            F::from_u256(F::MODULUS - mag)
        }
    }

    fn field_to_i64_test(f: F) -> i64 {
        let p_half = F::MODULUS / 2;
        let xu = f.to_u256();
        if xu > p_half {
            -((F::MODULUS - xu).as_u64() as i64)
        } else {
            xu.as_u64() as i64
        }
    }

    #[test]
    fn log_layer_hint_one_to_zero() {
        let scale: u64 = 1 << 18;
        let inputs = [field(scale as i64), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        log_hint::<F>(&inputs, &mut outputs).unwrap();
        assert_eq!(field_to_i64_test(outputs[0]), 0);
    }

    #[test]
    fn log_layer_hint_negative_output_for_small_real_value() {
        let scale: u64 = 1 << 10;
        let x_q = scale as i64 / 2;
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        log_hint::<F>(&inputs, &mut outputs).unwrap();
        let result = field_to_i64_test(outputs[0]);
        assert!(result < 0, "log(0.5) should be negative, got {result}");
    }
}
