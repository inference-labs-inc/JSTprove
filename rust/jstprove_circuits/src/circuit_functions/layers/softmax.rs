use std::collections::HashMap;

use circuit_std_rs::logup::LogUpSingleKeyTable;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::gadgets::{LogupRangeCheckContext, ShiftRangeContext, constrained_clip};
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

const EXP_TABLE_BITS: usize = 8;
const EXP_TABLE_SIZE: usize = 1 << EXP_TABLE_BITS;

#[derive(Debug)]
pub struct SoftmaxLayer {
    name: String,
    axis: isize,
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scale_exponent: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SoftmaxLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Softmax, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Softmax,
                name: input_name.clone(),
            })?
            .clone();

        let axis = if self.axis < 0 {
            (layer_input.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let shift_exponent = self.n_bits.saturating_sub(1);
        let shift_ctx = ShiftRangeContext::new::<C, Builder>(api, shift_exponent)?;
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let scale = 1u64 << self.scale_exponent;

        let mut exp_table = LogUpSingleKeyTable::new(EXP_TABLE_BITS);
        let (table_keys, table_values) = build_exp_table::<C, Builder>(api, scale);
        exp_table.new_table(table_keys, table_values);

        let out = softmax_along_axis::<C, Builder>(
            api,
            &shift_ctx,
            &mut logup_ctx,
            &mut exp_table,
            &layer_input,
            axis,
            scale,
            self.scale_exponent,
        )?;

        logup_ctx.finalize::<C, Builder>(api);
        exp_table.final_check::<C, Builder>(api);

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Softmax,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let axis: isize = params
            .get("axis")
            .and_then(|v| v.as_i64())
            .map(|v| v as isize)
            .unwrap_or(-1);

        Ok(Box::new(Self {
            name: layer.name.clone(),
            axis,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits: layer_context.n_bits,
            scale_exponent: circuit_params.scale_exponent as usize,
        }))
    }
}

fn build_exp_table<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scale: u64,
) -> (Vec<Variable>, Vec<Vec<Variable>>) {
    let mut keys = Vec::with_capacity(EXP_TABLE_SIZE);
    let mut values = Vec::with_capacity(EXP_TABLE_SIZE);

    for i in 0..EXP_TABLE_SIZE {
        let x_unscaled = (i as f64 / 16.0) - 8.0;
        let exp_val = x_unscaled.exp();
        let exp_clamped = exp_val.min(2981.0);
        let exp_scaled = (exp_clamped * scale as f64).round() as u64;

        keys.push(api.constant(i as u32));
        values.push(vec![api.constant(exp_scaled as u32)]);
    }

    (keys, values)
}

fn softmax_along_axis<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    range_ctx: &mut LogupRangeCheckContext,
    exp_table: &mut LogUpSingleKeyTable,
    input: &ArrayD<Variable>,
    axis: usize,
    scale: u64,
    scale_exponent: usize,
) -> Result<ArrayD<Variable>, CircuitError> {
    let shape = input.shape().to_vec();
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Ok(input.clone());
    }

    let mut result_data: Vec<Variable> = Vec::with_capacity(input.len());

    let outer_shape: Vec<usize> = shape.iter().take(axis).cloned().collect();
    let inner_shape: Vec<usize> = shape.iter().skip(axis + 1).cloned().collect();

    let outer_size: usize = outer_shape.iter().product::<usize>().max(1);
    let inner_size: usize = inner_shape.iter().product::<usize>().max(1);

    for outer_idx in 0..outer_size {
        for inner_idx in 0..inner_size {
            let mut slice_elements: Vec<Variable> = Vec::with_capacity(axis_len);
            for axis_idx in 0..axis_len {
                let flat_idx = outer_idx * (axis_len * inner_size) + axis_idx * inner_size + inner_idx;
                slice_elements.push(input.as_slice().unwrap()[flat_idx]);
            }

            let exp_values: Vec<Variable> = slice_elements
                .iter()
                .map(|&x| {
                    constrained_exp(api, shift_ctx, range_ctx, exp_table, x, scale, scale_exponent)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut sum = exp_values[0];
            for &exp_val in exp_values.iter().skip(1) {
                sum = api.add(sum, exp_val);
            }

            let softmax_values: Vec<Variable> = exp_values
                .iter()
                .map(|&exp_val| {
                    constrained_softmax_div(api, range_ctx, exp_val, sum, scale, scale_exponent)
                })
                .collect::<Result<Vec<_>, _>>()?;

            for val in softmax_values {
                result_data.push(val);
            }
        }
    }

    let reordered = reorder_from_slices(&result_data, &shape, axis);

    ArrayD::from_shape_vec(shape, reordered).map_err(|e| {
        CircuitError::from(LayerError::Other {
            layer: LayerKind::Softmax,
            msg: format!("Failed to create output array: {e}"),
        })
    })
}

fn reorder_from_slices<T: Clone>(
    slice_order_data: &[T],
    shape: &[usize],
    axis: usize,
) -> Vec<T> {
    let axis_len = shape[axis];
    let outer_size: usize = shape.iter().take(axis).product::<usize>().max(1);
    let inner_size: usize = shape.iter().skip(axis + 1).product::<usize>().max(1);

    let mut result = Vec::with_capacity(slice_order_data.len());
    result.resize(slice_order_data.len(), slice_order_data[0].clone());

    let mut src_idx = 0;
    for outer_idx in 0..outer_size {
        for inner_idx in 0..inner_size {
            for axis_idx in 0..axis_len {
                let dst_idx = outer_idx * (axis_len * inner_size) + axis_idx * inner_size + inner_idx;
                result[dst_idx] = slice_order_data[src_idx].clone();
                src_idx += 1;
            }
        }
    }

    result
}

fn constrained_exp<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    range_ctx: &mut LogupRangeCheckContext,
    table: &mut LogUpSingleKeyTable,
    x: Variable,
    scale: u64,
    scale_exponent: usize,
) -> Result<Variable, CircuitError> {
    let lower_abs = api.constant((8 * scale) as u32);
    let zero = api.constant(0);
    let lower = api.sub(zero, lower_abs);
    let upper = api.constant((8 * scale - 1) as u32);

    let x_clamped = constrained_clip(api, shift_ctx, range_ctx, x, Some(lower), Some(upper))?;

    let offset = api.constant((8 * scale) as u32);
    let x_shifted = api.add(x_clamped, offset);

    let hint_inputs = vec![x_shifted, api.constant(scale_exponent as u32)];
    let hint_out = api.new_hint("myhint.exp_bucket", &hint_inputs, 2);
    let idx = hint_out[0];
    let out = hint_out[1];

    let bucket_width = scale >> 4;
    let bucket_width_var = api.constant(bucket_width as u32);
    let l_bound = api.mul(idx, bucket_width_var);
    let r_bound = api.add(l_bound, bucket_width_var);

    let delta_low = api.sub(x_shifted, l_bound);
    let one = api.constant(1);
    let r_minus_x = api.sub(r_bound, x_shifted);
    let delta_high = api.sub(r_minus_x, one);

    let bucket_bits = scale_exponent.saturating_sub(4).max(1);
    range_ctx
        .range_check::<C, Builder>(api, delta_low, bucket_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Softmax,
            msg: format!("range_check delta_low failed: {e}"),
        })?;
    range_ctx
        .range_check::<C, Builder>(api, delta_high, bucket_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Softmax,
            msg: format!("range_check delta_high failed: {e}"),
        })?;

    table.query(idx, vec![out]);

    Ok(out)
}

fn constrained_softmax_div<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    range_ctx: &mut LogupRangeCheckContext,
    exp_val: Variable,
    sum: Variable,
    scale: u64,
    scale_exponent: usize,
) -> Result<Variable, CircuitError> {
    let scale_var = api.constant(scale as u32);
    let hint_inputs = vec![exp_val, sum, scale_var];
    let hint_out = api.new_hint("myhint.softmax_div", &hint_inputs, 2);
    let quotient = hint_out[0];
    let remainder = hint_out[1];

    let scaled_exp = api.mul(exp_val, scale_var);
    let prod = api.mul(quotient, sum);
    let reconstructed = api.add(prod, remainder);
    api.assert_is_equal(scaled_exp, reconstructed);

    let sum_bits = scale_exponent + 12;
    range_ctx
        .range_check::<C, Builder>(api, remainder, sum_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Softmax,
            msg: format!("range_check remainder failed: {e}"),
        })?;

    let one = api.constant(1);
    let sum_minus_one = api.sub(sum, one);
    let diff = api.sub(sum_minus_one, remainder);
    range_ctx
        .range_check::<C, Builder>(api, diff, sum_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Softmax,
            msg: format!("range_check remainder < sum failed: {e}"),
        })?;

    Ok(quotient)
}

#[cfg(test)]
mod soundness_tests {
    use crate::circuit_functions::gadgets::LogupRangeCheckContext;
    use crate::circuit_functions::hints::build_logup_hint_registry;
    use expander_compiler::frontend::*;

    const TEST_SCALE_EXPONENT: usize = 8;

    declare_circuit!(SoftmaxDivTestCircuit {
        exp_val: Variable,
        sum: Variable,
        expected_quotient: PublicVariable
    });

    impl Define<BN254Config> for SoftmaxDivTestCircuit<Variable> {
        fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
            let scale = 1u64 << TEST_SCALE_EXPONENT;

            let mut range_ctx = LogupRangeCheckContext::new_default();
            range_ctx.init::<BN254Config, Builder>(api);

            let scale_var = api.constant(scale as u32);
            let hint_inputs = vec![self.exp_val, self.sum, scale_var];
            let hint_out = api.new_hint("myhint.softmax_div", &hint_inputs, 2);
            let quotient = hint_out[0];
            let remainder = hint_out[1];

            let scaled_exp = api.mul(self.exp_val, scale_var);
            let prod = api.mul(quotient, self.sum);
            let reconstructed = api.add(prod, remainder);
            api.assert_is_equal(scaled_exp, reconstructed);

            let sum_bits = TEST_SCALE_EXPONENT + 12;
            range_ctx
                .range_check::<BN254Config, Builder>(api, remainder, sum_bits)
                .expect("range check remainder failed");

            let one = api.constant(1);
            let sum_minus_one = api.sub(self.sum, one);
            let diff = api.sub(sum_minus_one, remainder);
            range_ctx
                .range_check::<BN254Config, Builder>(api, diff, sum_bits)
                .expect("range check remainder < sum failed");

            range_ctx.finalize::<BN254Config, Builder>(api);

            api.assert_is_equal(quotient, self.expected_quotient);
        }
    }

    type F = CircuitField<BN254Config>;

    #[test]
    fn test_softmax_div_valid() {
        let mut hint_registry = build_logup_hint_registry::<F>();
        let scale: u64 = 1 << TEST_SCALE_EXPONENT;

        let compile_result: CompileResult<BN254Config> =
            compile(&SoftmaxDivTestCircuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let exp_val: u64 = 100;
        let sum: u64 = 300;
        let expected = (exp_val * scale) / sum;

        let assignment = SoftmaxDivTestCircuit::<F> {
            exp_val: F::from(exp_val as u32),
            sum: F::from(sum as u32),
            expected_quotient: F::from(expected as u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(output.iter().all(|&x| x), "Valid softmax division should pass");
    }

    #[test]
    fn test_softmax_div_uniform() {
        let mut hint_registry = build_logup_hint_registry::<F>();
        let scale: u64 = 1 << TEST_SCALE_EXPONENT;

        let compile_result: CompileResult<BN254Config> =
            compile(&SoftmaxDivTestCircuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let exp_val: u64 = scale;
        let sum: u64 = 3 * scale;
        let expected = (exp_val * scale) / sum;

        let assignment = SoftmaxDivTestCircuit::<F> {
            exp_val: F::from(exp_val as u32),
            sum: F::from(sum as u32),
            expected_quotient: F::from(expected as u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(output.iter().all(|&x| x), "Uniform softmax (1/3) should pass");
    }

    #[test]
    fn test_softmax_div_wrong_quotient_rejected() {
        let mut hint_registry = build_logup_hint_registry::<F>();
        let scale: u64 = 1 << TEST_SCALE_EXPONENT;

        let compile_result: CompileResult<BN254Config> =
            compile(&SoftmaxDivTestCircuit::default(), CompileOptions::default())
                .expect("Compilation failed");

        let exp_val: u64 = 100;
        let sum: u64 = 300;
        let wrong_quotient = (exp_val * scale) / sum + 10;

        let assignment = SoftmaxDivTestCircuit::<F> {
            exp_val: F::from(exp_val as u32),
            sum: F::from(sum as u32),
            expected_quotient: F::from(wrong_quotient as u32),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registry)
            .expect("Witness solving failed");

        let output = compile_result.layered_circuit.run(&witness);
        assert!(
            !output.iter().all(|&x| x),
            "Wrong quotient should be rejected"
        );
    }
}
