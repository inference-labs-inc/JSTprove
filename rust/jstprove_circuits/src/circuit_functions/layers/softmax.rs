use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, Axis, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{FunctionLookupTable, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::{exp::compute_exp_quantized, softmax_verified::SOFTMAX_VERIFIED_HINT_KEY},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{AXIS, INPUT},
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
};

#[derive(Debug)]
pub struct SoftmaxLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    axis: i64,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SoftmaxLayer {
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_lines
    )]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Softmax, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Softmax,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let rank = shape.len();

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

        let n = shape[axis];
        if n == 0 {
            let zero = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
            let empty_out = ArrayD::from_elem(IxDyn(&shape), zero);
            return Ok((self.outputs.clone(), empty_out));
        }
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let n_bits = self.n_bits;
        let scale_exp = self.scaling.trailing_zeros();
        let lookup_bits = function_lookup_bits(scale_exp);
        let max_diff_bits = lookup_bits - 1;

        if self.scaling > i64::MAX as u64 {
            return Err(LayerError::Other {
                layer: LayerKind::Softmax,
                msg: format!(
                    "scaling {} exceeds i64::MAX; compute_exp_quantized cannot represent exp(0)",
                    self.scaling
                ),
            }
            .into());
        }

        let mut exp_lookup = FunctionLookupTable::build_signed::<C, Builder>(
            api,
            compute_exp_quantized,
            lookup_bits,
            self.scaling,
        );

        let cross_tolerance = n as u64 * self.scaling;
        let cross_tol_var = api.constant(CircuitField::<C>::from_u256(U256::from(cross_tolerance)));
        let cross_tol_bits = (2 * cross_tolerance).next_power_of_two().trailing_zeros() as usize;

        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let mut out_array = ArrayD::from_elem(IxDyn(&shape), zero_var);

        let input_lanes: Vec<_> = x_input.lanes(Axis(axis)).into_iter().collect();
        for (in_lane, mut out_lane) in input_lanes
            .into_iter()
            .zip(out_array.lanes_mut(Axis(axis)).into_iter())
        {
            let x_vars: Vec<Variable> = in_lane.iter().copied().collect();
            let mut hint_inputs = x_vars.clone();
            hint_inputs.push(scale_var);

            let hint_out = api.new_hint(SOFTMAX_VERIFIED_HINT_KEY, &hint_inputs, 2 * n + 1);
            let y_vars = &hint_out[..n];
            let max_q = hint_out[n];
            let e_vars = &hint_out[n + 1..];

            let mut max_product = api.sub(max_q, x_vars[0]);
            logup_ctx.range_check::<C, Builder>(api, max_product, max_diff_bits)?;
            for &x_i in &x_vars[1..] {
                let diff = api.sub(max_q, x_i);
                logup_ctx.range_check::<C, Builder>(api, diff, max_diff_bits)?;
                max_product = api.mul(max_product, diff);
            }
            api.assert_is_zero(max_product);

            let mut sum_exp = zero_var;
            for (i, &x_i) in x_vars.iter().enumerate() {
                let d_i = api.sub(x_i, max_q);
                exp_lookup.query(d_i, e_vars[i]);
                sum_exp = api.add(sum_exp, e_vars[i]);
            }

            let mut lane_sum = zero_var;
            for (i, out_elem) in out_lane.iter_mut().enumerate() {
                let y = y_vars[i];
                logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;
                let upper_diff = api.sub(scale_var, y);
                logup_ctx.range_check::<C, Builder>(api, upper_diff, n_bits)?;

                let lhs = api.mul(y, sum_exp);
                let rhs = api.mul(e_vars[i], scale_var);
                let cross_diff = api.sub(lhs, rhs);
                let shifted = api.add(cross_diff, cross_tol_var);
                logup_ctx.range_check::<C, Builder>(api, shifted, cross_tol_bits)?;

                lane_sum = api.add(lane_sum, y);
                *out_elem = y;
            }

            let sum_tol = api.constant(CircuitField::<C>::from_u256(U256::from(n as u64)));
            let sum_diff = api.sub(lane_sum, scale_var);
            let sum_shifted = api.add(sum_diff, sum_tol);
            let sum_tol_bits = (2 * n + 1).next_power_of_two().trailing_zeros() as usize;
            logup_ctx.range_check::<C, Builder>(api, sum_shifted, sum_tol_bits)?;
        }

        exp_lookup.finalize::<C, Builder>(api);

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
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Softmax,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        if layer.opset_version_number < 13 {
            return Err(LayerError::Other {
                layer: LayerKind::Softmax,
                msg: format!(
                    "opset {} not supported: per-axis lanes semantics requires opset ≥ 13",
                    layer.opset_version_number
                ),
            }
            .into());
        }

        let default_axis: i64 = -1;
        let axis: i64 = match extract_params(layer) {
            Ok(params) => get_param_or_default(&layer.name, AXIS, &params, Some(&default_axis))
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::Softmax,
                    msg: format!("failed to read 'axis' attribute: {e}"),
                })?,
            Err(LayerError::MissingParameter { .. }) => default_axis,
            Err(e) => return Err(e.into()),
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
