// Elementwise ONNX `Gelu` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: `api.new_hint("jstprove.gelu_hint", &[x, scale], 1)` computes
//    `round(gelu(x_q / scale) * scale)` in native f64 and returns it as a
//    field element. No circuit constraint is added by this step.
// 2. **No range check**: GELU outputs can be negative (for negative inputs),
//    so we do not apply a LogUp range check (which only supports non-negative
//    values). The output is unconstrained beyond being a valid field element.
//
// # Soundness caveat
// The output is NOT constrained. A malicious prover can substitute any value.
// This is the same level of soundness used for LayerNorm in this codebase;
// full lookup-table soundness is a future improvement.
//
// # GELU formula
// gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::gelu::GELU_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
};

// -------- Struct --------

#[derive(Debug)]
pub struct GeluLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Scaling factor `2^scale_exponent`, baked into the hint call.
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GeluLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Resolve the single input tensor (no initializer — GELU has no weights).
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Gelu, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Gelu,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();

        // Build a constant variable for the scaling factor, shared across elements.
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input.iter() {
            // Compute gelu(x_q / scale) * scale via the native-f64 hint.
            // No range check because GELU outputs can be negative.
            let hint_out = api.new_hint(GELU_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            out_storage.push(y);
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
        // GELU has exactly one data input and no weights.
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Gelu,
                name: "input X".to_string(),
            })?;

        // Validate the `approximate` attribute. Only the tanh approximation is
        // implemented; the exact (erf-based) GELU requires a lookup table that is
        // not yet available. Reject at build time so callers are not silently
        // given the tanh result when they requested exact mode.
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
                    "Gelu approximate='{}' is not supported in the Expander backend: \
                     only approximate='tanh' is implemented. The exact (erf-based) Gelu \
                     requires a lookup-table constraint that is not yet available.",
                    approximate
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
        }))
    }
}
