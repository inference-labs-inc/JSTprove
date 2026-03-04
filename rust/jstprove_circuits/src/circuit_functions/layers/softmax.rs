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
// Softmax is applied independently along the specified axis (default -1, i.e.
// the last axis, matching ONNX opset ≥ 13 semantics). The hint is called once
// per lane along that axis, so batched inputs are handled correctly.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, Axis, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::range_check::LogupRangeCheckContext,
    hints::softmax::SOFTMAX_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{AXIS, INPUT},
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
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
    /// Softmax axis (may be negative; -1 means the last axis).
    axis: i64,
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
        let rank = shape.len();

        // Normalise the axis: negative values count from the end.
        let axis = if self.axis < 0 {
            let a = rank as i64 + self.axis;
            if a < 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Softmax,
                    msg: format!("axis {} out of range for tensor rank {rank}", self.axis),
                }
                .into());
            }
            a as usize
        } else {
            let a = self.axis as usize;
            if a >= rank {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Softmax,
                    msg: format!("axis {} out of range for tensor rank {rank}", self.axis),
                }
                .into());
            }
            a
        };

        // Number of elements along the softmax axis.
        let n = shape[axis];

        // Build a constant variable for the scaling factor, shared across the call.
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        // Shared LogUp context for all range checks in this layer.
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);
        let n_bits = self.n_bits;

        // Pre-allocate the output array with a zero placeholder; each element
        // will be overwritten by the corresponding hint output below.
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let mut out_array = ArrayD::from_elem(IxDyn(&shape), zero_var);

        // Iterate over corresponding input and output lanes simultaneously.
        // `lanes(Axis(axis))` / `lanes_mut(Axis(axis))` both visit slices in the
        // same C-order over all other axes, so zip() pairs them correctly.
        // Writing directly into out_array lanes avoids any index-ordering issues.
        let input_lanes: Vec<_> = x_input.lanes(Axis(axis)).into_iter().collect();
        for (in_lane, mut out_lane) in input_lanes
            .into_iter()
            .zip(out_array.lanes_mut(Axis(axis)).into_iter())
        {
            // Hint layout: [x_0, ..., x_{n-1}, scale] → n outputs.
            let mut hint_inputs: Vec<Variable> = in_lane.iter().copied().collect();
            hint_inputs.push(scale_var);

            let hint_out = api.new_hint(SOFTMAX_HINT_KEY, &hint_inputs, n);

            for (out_elem, &y) in out_lane.iter_mut().zip(hint_out.iter()) {
                logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;
                *out_elem = y;
            }
        }

        // Finalise the shared LogUp table (emits the consistency constraint).
        logup_ctx.finalize::<C, Builder>(api);

        // Guard: Softmax must produce exactly one output tensor.
        if self.outputs.len() != 1 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Softmax,
                param: format!(
                    "output Y: expected exactly 1 output, got {}",
                    self.outputs.len()
                ),
            }
            .into());
        }

        Ok((self.outputs.clone(), out_array))
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
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Softmax,
                name: "input X".to_string(),
            })?;

        if layer.outputs.len() != 1 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Softmax,
                param: format!(
                    "output Y: expected exactly 1 output, got {}",
                    layer.outputs.len()
                ),
            }
            .into());
        }

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

        // Reject opset < 13: apply() uses per-axis lanes semantics (opset ≥ 13).
        // Opset < 13 requires 2D-coercion semantics (flatten to 2D then apply
        // softmax along axis 1) which is not yet implemented; accepting such
        // models would silently produce incorrect outputs.
        if layer.opset_version_number < 13 {
            return Err(LayerError::Other {
                layer: LayerKind::Softmax,
                msg: format!(
                    "opset {} is not supported: apply() uses per-axis lanes semantics \
                    (opset ≥ 13 only); 2D coercion required by opset < 13 is not yet \
                    implemented",
                    layer.opset_version_number
                ),
            }
            .into());
        }

        // Read the axis attribute.
        // ONNX opset ≥13 defaults to -1 (last axis).
        let default_axis: i64 = -1;
        let axis: i64 = match extract_params(layer).ok() {
            Some(params) => get_param_or_default(&layer.name, AXIS, &params, Some(&default_axis))
                .map_err(|e| LayerError::Other {
                layer: LayerKind::Softmax,
                msg: format!("failed to read 'axis' attribute: {e}"),
            })?,
            None => default_axis,
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits,
            scaling,
            axis,
        }))
    }
}
