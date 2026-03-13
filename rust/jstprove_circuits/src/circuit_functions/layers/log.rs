// Elementwise ONNX `Log` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: `api.new_hint("jstprove.log_hint", &[x, scale], 1)` computes
//    `round(log(x_q / scale) * scale)` in native f64 and returns it as a
//    field element. No circuit constraint is added by this step.
// 2. **No range check**: Log outputs can be negative (for real inputs < 1),
//    so we do not apply a LogUp range check (which only supports non-negative
//    values). The output is unconstrained beyond being a valid field element.
//
// # Soundness caveat
// The output is NOT constrained. A malicious prover can substitute any value.
// Full lookup-table soundness is a future improvement.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    hints::log::LOG_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

// -------- Struct --------

#[derive(Debug)]
pub struct LogLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Scaling factor `2^scale_exponent`, baked into the hint call.
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LogLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Log, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Log,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());
        for &x in x_input.iter() {
            // Compute log(x_q / scale) * scale via the native-f64 hint.
            // No range check because Log outputs can be negative.
            let hint_out = api.new_hint(LOG_HINT_KEY, &[x, scale_var], 1);
            out_storage.push(hint_out[0]);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Log,
                msg: format!("LogLayer: cannot reshape result into {shape:?}"),
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
                layer: LayerKind::Log,
                name: "input X".to_string(),
            })?;

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
            scaling,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_functions::hints::log::log_hint;
    use ethnum::U256;
    use expander_compiler::field::BN254Fr;

    type F = BN254Fr;

    fn field(n: i64) -> F {
        if n >= 0 {
            F::from_u256(U256::from(n as u64))
        } else {
            let mag = U256::from((-n) as u64);
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
        let x_q = scale as i64 / 2; // represents 0.5
        let inputs = [field(x_q), F::from_u256(U256::from(scale))];
        let mut outputs = [F::zero()];
        log_hint::<F>(&inputs, &mut outputs).unwrap();
        let result = field_to_i64_test(outputs[0]);
        assert!(result < 0, "log(0.5) should be negative, got {result}");
    }
}
