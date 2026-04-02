// ONNX `LayerNormalization` layer for ZK circuits (Expander backend).
//
// # ZK approach
// LayerNormalization applies normalisation (mean/variance), a scale (gamma),
// and a shift (beta) — operations involving square roots and divisions that
// cannot be expressed as field polynomials.
//
// 1. **Hint**: `api.new_hint("jstprove.layer_norm_hint", inputs, n)` computes
//    the full layer norm in native f64 for one normalisation lane and injects
//    all `n` output elements as unconstrained witnesses.
//    Input layout: [x_0..x_{n-1}, γ_0..γ_{n-1}, β_0..β_{n-1}, scale].
// 2. **No range check**: the output can be negative, so the existing LogUp
//    range check (which only handles [0, 2^n_bits)) cannot be applied.
//    Downstream arithmetic layers (e.g. Gemm) constrain the outputs indirectly.
//
// # Axis handling
// Normalization is applied along the specified axis and all trailing axes
// (ONNX LayerNormalization semantics). Each "lane" — the slice starting from
// `axis` — is processed by one hint call.
//
// # Weight quantisation
// gamma (Scale) is quantised at α¹; beta (B) is quantised at α².  The hint
// function decodes both accordingly.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, i64_to_field},
    hints::layer_norm::LAYER_NORM_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default, get_w_or_b},
    },
};

// -------- Struct --------

#[derive(Debug)]
pub struct LayerNormLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    axis: usize,
    scaling: u64,
    #[allow(dead_code)]
    scale_exponent: u32,
    n_bits: usize,
    gamma: Vec<i64>,
    beta: Vec<i64>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LayerNormLayer {
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Guard: exactly one output tensor (fail fast before any computation).
        if self.outputs.len() != 1 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::LayerNormalization,
                param: format!(
                    "output Y: expected exactly 1 output, got {}",
                    self.outputs.len()
                ),
            }
            .into());
        }

        let x_name = get_input_name(&self.inputs, 0, LayerKind::LayerNormalization, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::LayerNormalization,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let rank = shape.len();
        let axis = self.axis;

        if axis >= rank {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: format!("axis {axis} out of range for tensor rank {rank}"),
            }
            .into());
        }

        // lane_size = product(shape[axis..])
        let lane_size: usize = shape[axis..].iter().product();

        if self.gamma.len() != lane_size {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: format!(
                    "gamma length {} != lane_size {}",
                    self.gamma.len(),
                    lane_size
                ),
            }
            .into());
        }
        if self.beta.len() != lane_size {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: format!("beta length {} != lane_size {}", self.beta.len(), lane_size),
            }
            .into());
        }

        // Encode gamma as constant Variables (signed i64 → field element).
        let gamma_vars: Vec<Variable> = self
            .gamma
            .iter()
            .map(|&g| {
                let field_val = if g >= 0 {
                    CircuitField::<C>::from_u256(U256::from(g as u64))
                } else {
                    let mag = U256::from(g.unsigned_abs());
                    CircuitField::<C>::from_u256(CircuitField::<C>::MODULUS - mag)
                };
                api.constant(field_val)
            })
            .collect();

        // Encode beta as constant Variables.
        let beta_vars: Vec<Variable> = self
            .beta
            .iter()
            .map(|&b| {
                let field_val = if b >= 0 {
                    CircuitField::<C>::from_u256(U256::from(b as u64))
                } else {
                    let mag = U256::from(b.unsigned_abs());
                    CircuitField::<C>::from_u256(CircuitField::<C>::MODULUS - mag)
                };
                api.constant(field_val)
            })
            .collect();

        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        // Pre-allocate output array.
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let mut out_array = ArrayD::from_elem(IxDyn(&shape), zero_var);

        // Iterate over lanes along the normalisation axis.
        // For ONNX LayerNormalization, normalization is over [axis, rank).
        // We treat the tensor as [outer, lane] where:
        //   outer = product(shape[:axis]),  lane = product(shape[axis:]).
        // ndarray's lanes(Axis(axis)) visits each axis-length slice; for
        // multi-dimensional trailing axes we flatten via the outer iteration.
        //
        // However, ndarray's Axis(axis) only gives slices of shape[axis] elements
        // (the single axis dimension), not the full trailing slice.  For a 3-D
        // tensor [B, T, C] with axis=1, we want lanes of size T*C, not T.
        //
        // Strategy: reshape the tensor view to [outer, lane] for the hint loop,
        // then write back into the original shape.
        let outer_size: usize = shape[..axis].iter().product();
        let flat_input: Vec<Variable> = x_input
            .as_slice()
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: "input tensor is not contiguous".to_string(),
            })?
            .to_vec();

        let n_bits = self.n_bits;
        let signed_offset: CircuitField<C> = i64_to_field::<C>(1i64 << (n_bits as u32 - 1));
        let offset_var = api.constant(signed_offset);

        let mut flat_output: Vec<Variable> = Vec::with_capacity(flat_input.len());

        for outer_i in 0..outer_size {
            let start = outer_i * lane_size;
            let end = start + lane_size;

            let mut hint_inputs: Vec<Variable> = flat_input[start..end].to_vec();
            hint_inputs.extend_from_slice(&gamma_vars);
            hint_inputs.extend_from_slice(&beta_vars);
            hint_inputs.push(scale_var);

            let hint_out = api.new_hint(LAYER_NORM_HINT_KEY, &hint_inputs, lane_size);

            for &y in &hint_out {
                let shifted = api.add(y, offset_var);
                logup_ctx.range_check::<C, Builder>(api, shifted, n_bits)?;
            }

            let mut input_sum = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
            for &x in &flat_input[start..end] {
                input_sum = api.add(input_sum, x);
            }
            let mut output_sum = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
            for &y in &hint_out {
                output_sum = api.add(output_sum, y);
            }
            let expected_output_sum = {
                let beta_sum_i64: i64 = self.beta.iter().sum();
                let beta_sum_scaled = (beta_sum_i64 as f64 / self.scaling as f64).round() as i64;
                api.constant(i64_to_field::<C>(beta_sum_scaled))
            };
            let sum_diff = api.sub(output_sum, expected_output_sum);
            let sum_tolerance = api.constant(CircuitField::<C>::from_u256(U256::from(
                lane_size as u64 + 1,
            )));
            let sum_diff_shifted = api.add(sum_diff, sum_tolerance);
            let sum_tol_bits =
                (2 * lane_size + 3).next_power_of_two().trailing_zeros() as usize + 1;
            logup_ctx.range_check::<C, Builder>(api, sum_diff_shifted, sum_tol_bits)?;

            flat_output.extend_from_slice(&hint_out);
        }

        // Reshape flat output back into the original tensor shape.
        let out_flat_ref = out_array
            .as_slice_mut()
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: "output array is not contiguous".to_string(),
            })?;
        out_flat_ref.copy_from_slice(&flat_output);

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // Guard: at least two inputs (data + gamma).
        if layer.inputs.len() < 2 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::LayerNormalization,
                param: format!(
                    "expected at least 2 inputs (data, gamma[, beta]), got {}",
                    layer.inputs.len()
                ),
            }
            .into());
        }

        // Guard: exactly one output.
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::LayerNormalization,
                param: "output tensor Y".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::LayerNormalization,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        // Read axis attribute (ONNX LayerNormalization default = -1).
        let default_axis: i64 = -1;
        let raw_axis: i64 = match extract_params(layer).ok() {
            Some(params) => get_param_or_default(&layer.name, "axis", &params, Some(&default_axis))
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::LayerNormalization,
                    msg: format!("failed to read 'axis' attribute: {e}"),
                })?,
            None => default_axis,
        };

        let rank = output_shape.len();
        let axis = if raw_axis < 0 {
            let a = rank as i64 + raw_axis;
            if a < 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::LayerNormalization,
                    msg: format!("axis {raw_axis} out of range for data rank {rank}"),
                }
                .into());
            }
            a as usize
        } else {
            let a = raw_axis as usize;
            if a >= rank {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::LayerNormalization,
                    msg: format!("axis {raw_axis} out of range for data rank {rank}"),
                }
                .into());
            }
            a
        };

        // Load gamma (Scale) weights — quantised at α¹ by the scale plan.
        let gamma_name = &layer.inputs[1];
        let gamma_array: ndarray::ArrayD<i64> = get_w_or_b(layer_context.w_and_b_map, gamma_name)
            .map_err(|e| LayerError::Other {
            layer: LayerKind::LayerNormalization,
            msg: format!("failed to read gamma tensor '{gamma_name}': {e}"),
        })?;
        let gamma: Vec<i64> = gamma_array
            .as_slice()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::LayerNormalization,
                msg: "gamma tensor is not contiguous".to_string(),
            })?
            .to_vec();

        // Load beta (B) weights — quantised at α² by the scale plan.  Optional.
        let lane_size: usize = output_shape[axis..].iter().product();
        let beta: Vec<i64> = if let Some(beta_name) = layer.inputs.get(2) {
            let beta_array: ndarray::ArrayD<i64> = get_w_or_b(layer_context.w_and_b_map, beta_name)
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::LayerNormalization,
                    msg: format!("failed to read beta tensor '{beta_name}': {e}"),
                })?;
            beta_array
                .as_slice()
                .ok_or_else(|| LayerError::Other {
                    layer: LayerKind::LayerNormalization,
                    msg: "beta tensor is not contiguous".to_string(),
                })?
                .to_vec()
        } else {
            vec![0i64; lane_size]
        };

        let n_bits = layer_context.n_bits_for(&layer.name);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axis,
            scaling,
            scale_exponent: circuit_params.scale_exponent,
            n_bits,
            gamma,
            beta,
        }))
    }
}
