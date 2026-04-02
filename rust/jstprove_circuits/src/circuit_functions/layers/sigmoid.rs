// Elementwise ONNX `Sigmoid` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: `api.new_hint("jstprove.sigmoid_hint", &[x, scale], 1)` computes
//    `round(sigmoid(x_q / scale) * scale)` in native f64 and returns it as a
//    field element. No circuit constraint is added by this step.
// 2. **Range check**: `logup_ctx.range_check(api, y, n_bits)` constrains the
//    output to `[0, 2^n_bits)`, ensuring it stays within the quantised range.
//
// # Soundness caveat
// The output is bounded but NOT proven equal to `sigmoid(input)`. A malicious
// prover can substitute any non-negative value that passes the range check.
// This is the same level of soundness used for Exp and Softmax in this
// codebase; full lookup-table soundness is a future improvement.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    hints::sigmoid::SIGMOID_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

// -------- Struct --------

#[derive(Debug)]
pub struct SigmoidLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Number of bits used for the LogUp range check on the output.
    n_bits: usize,
    /// Scaling factor `2^scale_exponent`, baked into the hint call.
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SigmoidLayer {
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Resolve the single input tensor (no initializer — Sigmoid has no weights).
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Sigmoid, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Sigmoid,
            name: x_name.clone(),
        })?;

        let shape = x_input.shape().to_vec();

        // Build a constant variable for the scaling factor, shared across elements.
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let n_bits = self.n_bits;
        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            // 1. Compute sigmoid(x_q / scale) * scale via the native-f64 hint.
            let hint_out = api.new_hint(SIGMOID_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            // 2. Bound the output to [0, 2^n_bits) via LogUp range check.
            logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;

            out_storage.push(y);
        }

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
        // Sigmoid has exactly one data input and no weights.
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
        }))
    }
}
