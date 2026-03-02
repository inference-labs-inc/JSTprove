// ONNX `Softmax` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: `api.new_hint("jstprove.softmax_hint", &[x_0, ..., x_{n-1}, scale], n)`
//    computes the full numerically-stable softmax in native f64 and returns all
//    `n` output elements as field elements. No circuit constraint is added by
//    this step.
// 2. **Range check**: each output element `y_i` is constrained to
//    `[0, 2^n_bits)` via `logup_ctx.range_check`, ensuring values stay within
//    the quantised range `[0, scale]`.
//
// # Soundness caveat
// Each output is bounded but NOT proven to equal `softmax(input)[i]`. A
// malicious prover can substitute any non-negative in-range value. This is the
// same level of soundness used for Exp; full lookup-table soundness is a
// planned future extension.
//
// # Axis handling
// The current implementation treats the entire flattened input tensor as a
// single softmax reduction (equivalent to axis=-1 on a 1-D tensor). This
// covers the most common use case: softmax applied to a 1-D logit vector after
// Flatten/Reshape. Multi-axis batched softmax is a future extension.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    hints::softmax::SOFTMAX_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

// -------- Struct --------

#[derive(Debug)]
pub struct SoftmaxLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Number of bits used for the LogUp range check on each output element.
    n_bits: usize,
    /// Scaling factor `2^scale_exponent`, baked into the hint call.
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SoftmaxLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Resolve the single input tensor (Softmax has no initializer weights).
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Softmax, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Softmax,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let n = shape.iter().product::<usize>();

        // Build a constant variable for the scaling factor, shared across the call.
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        // Collect all input variables, then append the scale constant.
        // Hint layout: inputs = [x_0, x_1, ..., x_{n-1}, scale], outputs = n elements.
        let mut hint_inputs: Vec<Variable> = x_input.iter().copied().collect();
        hint_inputs.push(scale_var);

        // 1. Compute softmax(x) in native f64 via the hint; get n output variables.
        let hint_out = api.new_hint(SOFTMAX_HINT_KEY, &hint_inputs, n);

        // 2. Bound each output to [0, 2^n_bits) via a shared LogUp range check.
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let n_bits = self.n_bits;
        for &y in &hint_out {
            logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;
        }

        // Finalise the shared LogUp table (emits the consistency constraint).
        logup_ctx.finalize::<C, Builder>(api);

        let result = ArrayD::from_shape_vec(IxDyn(&shape), hint_out).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Softmax,
                msg: format!("SoftmaxLayer: cannot reshape result into shape {shape:?}"),
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
        // Softmax has exactly one data input and no weights.
        let _x_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Softmax,
                name: "input X".to_string(),
            })?;

        let n_bits = layer_context.n_bits_for(&layer.name);

        // scaling = 2^scale_exponent; fits comfortably in u64 for practical exponents.
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Softmax,
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
