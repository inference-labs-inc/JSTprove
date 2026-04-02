use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{DecomposedExpLookup, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::{exp::compute_exp_quantized, sigmoid::SIGMOID_HINT_KEY},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct SigmoidLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SigmoidLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Sigmoid, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Sigmoid,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let scale_sq = api.constant(CircuitField::<C>::from_u256(U256::from(
            self.scaling * self.scaling,
        )));
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));

        let cross_tolerance = 2u64 * self.scaling;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
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
            let hint_out = api.new_hint(SIGMOID_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            logup_ctx.range_check::<C, Builder>(api, y, self.n_bits)?;
            let upper = api.sub(scale_var, y);
            logup_ctx.range_check::<C, Builder>(api, upper, self.n_bits)?;

            let neg_x = api.sub(zero_var, x);
            let exp_neg_x = decomposed.verify_exp::<C, Builder>(api, logup_ctx, neg_x)?;

            let denom = api.add(scale_var, exp_neg_x);
            let lhs = api.mul(y, denom);
            let diff = api.sub(lhs, scale_sq);
            let shifted = api.add(diff, cross_tol_var);
            logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;

            out_storage.push(y);
        }

        decomposed.finalize::<C, Builder>(api);

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Sigmoid,
                msg: format!("SigmoidLayer: cannot reshape result into shape {shape:?}"),
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
                layer: LayerKind::Sigmoid,
                name: "input X".to_string(),
            })?;

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Sigmoid,
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
