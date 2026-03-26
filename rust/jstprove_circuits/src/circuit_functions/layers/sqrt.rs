// Elementwise ONNX `Sqrt` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: computes sqrt(x_q / scale) * scale in native f64.
// 2. **Input non-negativity check**: constrains x to [0, 2^n_bits), which
//    enforces x >= 0 before invoking the hint.
// 3. **Output range check**: constrains sqrt result to [0, 2^n_bits).

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    hints::sqrt::SQRT_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct SqrtLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    #[allow(dead_code)]
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SqrtLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Sqrt, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Sqrt,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let n_bits = self.n_bits;
        let remainder_bits = n_bits + 1;
        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            // Prove x >= 0 before invoking the hint.  Without this constraint a
            // dishonest prover could supply a negative x; the hint clamps it to
            // 0 and the output range-check still passes, creating a false proof.
            logup_ctx.range_check::<C, Builder>(api, x, n_bits)?;
            let hint_out = api.new_hint(SQRT_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;

            let x_times_scale = api.mul(x, scale_var);
            let y_sq = api.mul(y, y);
            let remainder = api.sub(x_times_scale, y_sq);
            logup_ctx.range_check::<C, Builder>(api, remainder, remainder_bits)?;

            let two_y = api.add(y, y);
            let upper = api.sub(two_y, remainder);
            logup_ctx.range_check::<C, Builder>(api, upper, remainder_bits)?;

            out_storage.push(y);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Sqrt,
                msg: format!("SqrtLayer: cannot reshape result into shape {shape:?}"),
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
                layer: LayerKind::Sqrt,
                msg: format!("Sqrt expects exactly 1 input, got {}", layer.inputs.len()),
            }
            .into());
        }
        if layer.outputs.len() != 1 {
            return Err(LayerError::Other {
                layer: LayerKind::Sqrt,
                msg: format!("Sqrt expects exactly 1 output, got {}", layer.outputs.len()),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Sqrt,
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
