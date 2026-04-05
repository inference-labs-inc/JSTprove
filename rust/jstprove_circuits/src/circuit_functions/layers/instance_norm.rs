use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, i64_to_field},
    hints::instance_norm::INSTANCE_NORM_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
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
pub struct InstanceNormLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
    scale_exponent: u32,
    n_bits: usize,
    gamma: Vec<i64>,
    beta: Vec<i64>,
    n_channels: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for InstanceNormLayer {
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
                layer: LayerKind::InstanceNormalization,
                param: format!(
                    "output Y: expected exactly 1 output, got {}",
                    self.outputs.len()
                ),
            }
            .into());
        }

        let x_name = get_input_name(&self.inputs, 0, LayerKind::InstanceNormalization, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::InstanceNormalization,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let rank = shape.len();
        if rank < 3 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::InstanceNormalization,
                msg: format!(
                    "InstanceNorm expects at least 3-D input [N,C,spatial...], got rank {rank}"
                ),
            }
            .into());
        }

        let big_n = shape[0];
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        if c != self.n_channels {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::InstanceNormalization,
                msg: format!(
                    "InstanceNorm channel mismatch: expected {}, got {c}",
                    self.n_channels
                ),
            }
            .into());
        }

        let lane_size = spatial_size;

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
        let norm_var_tolerance = api.constant(CircuitField::<C>::from_u256(U256::from(
            lane_size as u64 * self.scaling,
        )));
        let norm_var_tol_bits = (2 * lane_size as u64 * self.scaling)
            .next_power_of_two()
            .trailing_zeros() as usize;

        let mean_tolerance =
            api.constant(CircuitField::<C>::from_u256(U256::from(lane_size as u64)));
        let mean_tol_bits = (2 * lane_size + 1).next_power_of_two().trailing_zeros() as usize;

        let per_elem_tolerance = api.constant(CircuitField::<C>::from_u256(U256::from(3u64)));
        let per_elem_tol_bits: usize = 3;

        let flat_input: Vec<Variable> = x_input
            .as_slice()
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::InstanceNormalization,
                msg: "input tensor is not contiguous".to_string(),
            })?
            .to_vec();

        let mut flat_output: Vec<Variable> = Vec::with_capacity(flat_input.len());

        for n_idx in 0..big_n {
            for c_idx in 0..c {
                let gamma_var = api.constant(i64_to_field::<C>(self.gamma[c_idx]));
                let beta_var = api.constant(i64_to_field::<C>(self.beta[c_idx]));

                let start = (n_idx * c + c_idx) * spatial_size;
                let end = start + spatial_size;

                let mut hint_inputs: Vec<Variable> = flat_input[start..end].to_vec();

                let gamma_vars_for_hint: Vec<Variable> =
                    (0..lane_size).map(|_| gamma_var).collect();
                let beta_vars_for_hint: Vec<Variable> = (0..lane_size).map(|_| beta_var).collect();

                hint_inputs.extend_from_slice(&gamma_vars_for_hint);
                hint_inputs.extend_from_slice(&beta_vars_for_hint);
                hint_inputs.push(scale_var);

                let hint_out = api.new_hint(INSTANCE_NORM_HINT_KEY, &hint_inputs, lane_size + 2);
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

                let mut norm_sq_sum = zero_var;

                for (i, &y) in y_vars.iter().enumerate() {
                    let shifted = api.add(y, offset_var);
                    logup_ctx.range_check::<C, Builder>(api, shifted, n_bits)?;

                    let dev = api.sub(flat_input[start + i], mean_q);
                    let norm_unscaled = api.mul(dev, inv_std_q);

                    let norm_q = api.unconstrained_int_div(norm_unscaled, scale_var);
                    let norm_rem = api.unconstrained_mod(norm_unscaled, scale_var);
                    let recon = api.mul(norm_q, scale_var);
                    let recon = api.add(recon, norm_rem);
                    api.assert_is_equal(recon, norm_unscaled);
                    logup_ctx.range_check::<C, Builder>(
                        api,
                        norm_rem,
                        self.scale_exponent as usize,
                    )?;

                    let norm_shifted = api.add(norm_q, norm_offset);
                    logup_ctx.range_check::<C, Builder>(api, norm_shifted, max_norm_bits + 1)?;

                    let norm_sq = api.mul(norm_q, norm_q);
                    norm_sq_sum = api.add(norm_sq_sum, norm_sq);

                    let lhs = api.mul(y, scale_var);
                    let term = api.mul(norm_q, gamma_var);
                    let rhs = api.add(term, beta_var);
                    let elem_diff = api.sub(lhs, rhs);
                    let elem_shifted = api.add(elem_diff, per_elem_tolerance);
                    logup_ctx.range_check::<C, Builder>(api, elem_shifted, per_elem_tol_bits)?;
                }

                let var_diff = api.sub(norm_sq_sum, n_alpha_sq);
                let var_shifted = api.add(var_diff, norm_var_tolerance);
                logup_ctx.range_check::<C, Builder>(api, var_shifted, norm_var_tol_bits)?;

                flat_output.extend_from_slice(y_vars);
            }
        }

        let out_array = ArrayD::from_shape_vec(IxDyn(&shape), flat_output).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::InstanceNormalization,
                msg: format!("InstanceNorm output reshape failed: {e}"),
            }
        })?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn build(
        _layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        Err(LayerError::Other {
            layer: LayerKind::InstanceNormalization,
            msg: "InstanceNormalization is not yet supported: the algebraic verification \
                  requires signed euclidean division infrastructure that is not yet \
                  implemented (same limitation as LayerNormalization for standalone models)"
                .to_string(),
        }
        .into())
    }
}
