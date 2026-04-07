use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{
        LogupRangeCheckContext, euclidean_division::div_pos_integer_pow2_constant_inner,
        i64_to_field,
    },
    hints::layer_norm_verified::LAYER_NORM_VERIFIED_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default, get_w_or_b},
    },
};

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn compute_max_norm_bits(lane_size: usize, scale_exponent: u32) -> usize {
    let n_f64 = lane_size as f64;
    let alpha = (1u64 << scale_exponent) as f64;
    let max_norm = n_f64.sqrt() * alpha;
    let raw = (max_norm.ceil() as u64)
        .next_power_of_two()
        .trailing_zeros() as usize;
    raw + 1
}

#[derive(Debug)]
pub struct LayerNormLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    axis: usize,
    scaling: u64,
    scale_exponent: u32,
    n_bits: usize,
    gamma: Vec<i64>,
    beta: Vec<i64>,
    max_abs_gamma: u64,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for LayerNormLayer {
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap,
        clippy::similar_names
    )]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
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

        let lane_size: usize = shape[axis..].iter().product();

        if self.gamma.len() != lane_size || self.beta.len() != lane_size {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::LayerNormalization,
                msg: format!(
                    "gamma/beta length mismatch: gamma={}, beta={}, lane_size={}",
                    self.gamma.len(),
                    self.beta.len(),
                    lane_size
                ),
            }
            .into());
        }

        let gamma_vars: Vec<Variable> = self
            .gamma
            .iter()
            .map(|&g| api.constant(i64_to_field::<C>(g)))
            .collect();

        let beta_vars: Vec<Variable> = self
            .beta
            .iter()
            .map(|&b| api.constant(i64_to_field::<C>(b)))
            .collect();

        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let n_const = api.constant(CircuitField::<C>::from_u256(U256::from(lane_size as u64)));
        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));

        let n_bits = self.n_bits;
        let signed_offset = {
            let shift = U256::from(1u64) << (n_bits as u32 - 1);
            CircuitField::<C>::from_u256(shift)
        };
        let offset_var = api.constant(signed_offset);

        let max_norm_bits = compute_max_norm_bits(lane_size, self.scale_exponent);
        let norm_offset = api.constant(CircuitField::<C>::from_u256(U256::from(
            1u64 << max_norm_bits,
        )));
        let n_alpha_sq = api.constant(CircuitField::<C>::from_u256(U256::from(
            lane_size as u64 * self.scaling * self.scaling,
        )));
        let norm_var_tol_native = lane_size as u64 * self.scaling * self.scaling;
        let norm_var_tolerance = api.constant(CircuitField::<C>::from_u256(U256::from(
            norm_var_tol_native,
        )));
        let norm_var_tol_bits = (2 * norm_var_tol_native)
            .next_power_of_two()
            .trailing_zeros() as usize;

        let mean_tolerance =
            api.constant(CircuitField::<C>::from_u256(U256::from(lane_size as u64)));
        let mean_tol_bits = (2 * lane_size + 1).next_power_of_two().trailing_zeros() as usize;

        // `per_elem_tolerance_val = scaling + 2 * max|γ_q| + 4`, computed with
        // checked arithmetic: on overflow or if the value would push
        // `next_power_of_two` out of range, fail the build rather than
        // silently saturating into a panic or an unsound bound.
        let per_elem_tolerance_val: u64 = self
            .max_abs_gamma
            .checked_mul(2)
            .and_then(|v| v.checked_add(self.scaling))
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::LayerNormalization,
                msg: format!(
                    "per-element tolerance overflow: scaling={}, max_abs_gamma={}",
                    self.scaling, self.max_abs_gamma
                ),
            })?;
        let per_elem_tol_upper: u64 = per_elem_tolerance_val
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::LayerNormalization,
                msg: format!(
                    "per-element tolerance bit-width overflow: tol={per_elem_tolerance_val}"
                ),
            })?;
        // `u64::next_power_of_two` panics when the input exceeds `2^63`.
        if per_elem_tol_upper > (1u64 << 63) {
            return Err(LayerError::Other {
                layer: LayerKind::LayerNormalization,
                msg: format!(
                    "per-element tolerance bit-width exceeds 63 bits: tol={per_elem_tolerance_val}"
                ),
            }
            .into());
        }
        let per_elem_tolerance = api.constant(CircuitField::<C>::from_u256(U256::from(
            per_elem_tolerance_val,
        )));
        // Range-check bit widths computed via `next_power_of_two` round up to a
        // power-of-two width `b`, so `range_check(shifted, b)` only enforces
        // `shifted < 2^b`, leaving the effective upper bound on the signed
        // residual as `2^b - 1 - tol` rather than `tol`. Mirror each such check
        // with `range_check(tol - diff, b)` to tighten both sides to |diff| ≤ tol.
        let per_elem_tol_bits: usize =
            per_elem_tol_upper.next_power_of_two().trailing_zeros() as usize;

        let norm_div_shift_exponent: usize = 2usize
            .saturating_mul(n_bits)
            .saturating_add(1)
            .max(self.scale_exponent as usize + 1);
        let norm_div_shift_over_scale_exponent: usize =
            norm_div_shift_exponent - self.scale_exponent as usize;
        let norm_div_shift_u256 = U256::ONE << (norm_div_shift_exponent as u32);
        let norm_div_shift_const = api.constant(CircuitField::<C>::from_u256(norm_div_shift_u256));
        let norm_div_shift_over_scale_u256 =
            U256::ONE << (norm_div_shift_over_scale_exponent as u32);
        let norm_div_shift_over_scale_const =
            api.constant(CircuitField::<C>::from_u256(norm_div_shift_over_scale_u256));

        let mean_signed_offset_u256 = U256::ONE << (n_bits as u32);
        let mean_signed_offset_const =
            api.constant(CircuitField::<C>::from_u256(mean_signed_offset_u256));
        let mean_abs_bits: usize = n_bits + 1;

        let outer_size: usize = shape[..axis].iter().product();
        let flat_input: Vec<Variable> = x_input.iter().copied().collect();

        let mut out_array = ArrayD::from_elem(IxDyn(&shape), zero_var);
        let mut flat_output: Vec<Variable> = Vec::with_capacity(flat_input.len());

        for outer_i in 0..outer_size {
            let start = outer_i * lane_size;
            let end = start + lane_size;

            let mut hint_inputs: Vec<Variable> = flat_input[start..end].to_vec();
            hint_inputs.extend_from_slice(&gamma_vars);
            hint_inputs.extend_from_slice(&beta_vars);
            hint_inputs.push(scale_var);

            let hint_out = api.new_hint(LAYER_NORM_VERIFIED_HINT_KEY, &hint_inputs, lane_size + 2);
            let y_vars = &hint_out[..lane_size];
            let mean_q = hint_out[lane_size];
            let inv_std_q = hint_out[lane_size + 1];

            logup_ctx.range_check::<C, Builder>(api, inv_std_q, n_bits)?;

            let mut input_sum = zero_var;
            for &x in &flat_input[start..end] {
                input_sum = api.add(input_sum, x);
            }
            let n_times_mean = api.mul(n_const, mean_q);
            let mean_diff = api.sub(input_sum, n_times_mean);
            let mean_shifted = api.add(mean_diff, mean_tolerance);
            logup_ctx.range_check::<C, Builder>(api, mean_shifted, mean_tol_bits)?;

            let mean_abs_shifted = api.add(mean_q, mean_signed_offset_const);
            logup_ctx.range_check::<C, Builder>(api, mean_abs_shifted, mean_abs_bits)?;

            let mut norm_sq_sum = zero_var;

            for (i, &y) in y_vars.iter().enumerate() {
                let shifted = api.add(y, offset_var);
                logup_ctx.range_check::<C, Builder>(api, shifted, n_bits)?;

                let dev = api.sub(flat_input[start + i], mean_q);
                let norm_unscaled = api.mul(dev, inv_std_q);

                // `norm_unscaled` is a signed integer that can be negative; the
                // unconstrained U256 IntDiv used inside the gadget requires a
                // non-negative dividend in signed interpretation. Delegate to
                // the shared Euclidean-division gadget, which adds the
                // positive shift before dividing and returns the signed
                // quotient.
                let norm_q = div_pos_integer_pow2_constant_inner::<C, Builder>(
                    api,
                    logup_ctx,
                    norm_unscaled,
                    scale_var,
                    norm_div_shift_const,
                    self.scale_exponent as usize,
                    norm_div_shift_over_scale_exponent,
                    norm_div_shift_over_scale_const,
                    true,
                )?;

                let norm_shifted = api.add(norm_q, norm_offset);
                logup_ctx.range_check::<C, Builder>(api, norm_shifted, max_norm_bits + 1)?;

                let norm_sq = api.mul(norm_q, norm_q);
                norm_sq_sum = api.add(norm_sq_sum, norm_sq);

                let lhs = api.mul(y, scale_var);
                let term = api.mul(norm_q, gamma_vars[i]);
                let rhs = api.add(term, beta_vars[i]);
                let elem_diff = api.sub(lhs, rhs);
                let elem_shifted = api.add(elem_diff, per_elem_tolerance);
                logup_ctx.range_check::<C, Builder>(api, elem_shifted, per_elem_tol_bits)?;
                let elem_mirrored = api.sub(per_elem_tolerance, elem_diff);
                logup_ctx.range_check::<C, Builder>(api, elem_mirrored, per_elem_tol_bits)?;
            }

            let var_diff = api.sub(norm_sq_sum, n_alpha_sq);
            let var_shifted = api.add(var_diff, norm_var_tolerance);
            logup_ctx.range_check::<C, Builder>(api, var_shifted, norm_var_tol_bits)?;

            flat_output.extend_from_slice(y_vars);
        }

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

        let max_abs_gamma: u64 = gamma.iter().map(|g| g.unsigned_abs()).max().unwrap_or(0);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axis,
            scaling,
            scale_exponent: circuit_params.scale_exponent,
            n_bits,
            gamma,
            beta,
            max_abs_gamma,
        }))
    }
}
