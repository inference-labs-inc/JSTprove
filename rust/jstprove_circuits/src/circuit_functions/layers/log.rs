// Elementwise ONNX `Log` layer for int64 fixed-point tensors.
//
// # ZK approach
// Log(x) = ln(x) is a transcendental function. A sound implementation requires
// a lookup-table constraint (log-lookup). Until that is available, `apply`
// refuses to execute and returns an error, following the same fail-closed
// policy as TopKLayer.
//
// # Current status
// **Not supported** in the Expander backend. `apply` always returns an error.

use std::collections::HashMap;

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
};

// -------- Struct --------

// Fields populated by `build` are kept for when a sound log-lookup constraint
// is eventually implemented. `apply` always returns an error in the meantime.
#[allow(dead_code)]
#[derive(Debug)]
pub struct LogLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LogLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Log is not soundly implementable via hint alone: the hint produces an
        // unconstrained output that a malicious prover can set to any value.
        // Full soundness requires a log-lookup table, which is not yet implemented.
        Err(LayerError::Other {
            layer: LayerKind::Log,
            msg: "Log is not yet supported in the Expander backend: soundly proving \
                  the log relation requires a lookup-table constraint that is not \
                  yet implemented."
                .to_string(),
        }
        .into())
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
    use crate::circuit_functions::hints::log::log_hint;
    use ethnum::U256;
    use expander_compiler::field::{BN254Fr, FieldArith};

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
