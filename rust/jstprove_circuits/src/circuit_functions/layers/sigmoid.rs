use std::collections::HashMap;

use circuit_std_rs::logup::LogUpSingleKeyTable;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, ShiftRangeContext, constrained_clip},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

const SIGMOID_TABLE_BITS: usize = 8;
const SIGMOID_TABLE_SIZE: usize = 1 << SIGMOID_TABLE_BITS;

#[derive(Debug)]
pub struct SigmoidLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scale_exponent: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SigmoidLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Sigmoid, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Sigmoid,
                name: input_name.clone(),
            })?
            .clone();

        let scale = 1u64 << self.scale_exponent;

        let shift_exponent = self.n_bits.saturating_sub(1);
        let shift_ctx = ShiftRangeContext::new::<C, Builder>(api, shift_exponent)?;
        let mut range_ctx = LogupRangeCheckContext::new_default();
        range_ctx.init::<C, Builder>(api);

        let mut logup_table = LogUpSingleKeyTable::new(SIGMOID_TABLE_BITS);
        let (table_keys, table_values) = build_sigmoid_table::<C, Builder>(api, scale);
        logup_table.new_table(table_keys, table_values);

        let shape = layer_input.shape().to_vec();
        let mut results: Vec<Variable> = Vec::with_capacity(layer_input.len());

        for &x in layer_input.iter() {
            let out = constrained_sigmoid::<C, Builder>(
                api,
                &shift_ctx,
                &mut range_ctx,
                &mut logup_table,
                x,
                scale,
                self.scale_exponent,
            )?;
            results.push(out);
        }

        range_ctx.finalize::<C, Builder>(api);
        logup_table.final_check::<C, Builder>(api);

        let out = ArrayD::from_shape_vec(shape, results).map_err(|e| LayerError::Other {
            layer: LayerKind::Sigmoid,
            msg: format!("Failed to reshape result: {e}"),
        })?;

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
        let (_, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Sigmoid,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits: layer_context.n_bits,
            scale_exponent: circuit_params.scale_exponent as usize,
        }))
    }
}

fn build_sigmoid_table<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scale: u64,
) -> (Vec<Variable>, Vec<Vec<Variable>>) {
    let mut keys = Vec::with_capacity(SIGMOID_TABLE_SIZE);
    let mut values = Vec::with_capacity(SIGMOID_TABLE_SIZE);

    for i in 0..SIGMOID_TABLE_SIZE {
        let x_unscaled = (i as f64 / 16.0) - 8.0;
        let sigmoid_val = 1.0 / (1.0 + (-x_unscaled).exp());
        let sigmoid_scaled = (sigmoid_val * scale as f64).round() as u32;

        keys.push(api.constant(i as u32));
        values.push(vec![api.constant(sigmoid_scaled)]);
    }

    (keys, values)
}

fn constrained_sigmoid<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    range_ctx: &mut LogupRangeCheckContext,
    table: &mut LogUpSingleKeyTable,
    x: Variable,
    scale: u64,
    scale_exponent: usize,
) -> Result<Variable, CircuitError> {
    let lower = api.constant((-(8i64 * scale as i64)) as u32);
    let upper = api.constant((8 * scale - 1) as u32);

    let x_clamped = constrained_clip(api, shift_ctx, range_ctx, x, Some(lower), Some(upper))?;

    let offset = api.constant((8 * scale) as u32);
    let x_shifted = api.add(x_clamped, offset);

    let hint_inputs = vec![x_shifted, api.constant(scale_exponent as u32)];
    let hint_out = api.new_hint("myhint.sigmoid_bucket", &hint_inputs, 2);
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
            layer: LayerKind::Sigmoid,
            msg: format!("range_check delta_low failed: {e}"),
        })?;
    range_ctx
        .range_check::<C, Builder>(api, delta_high, bucket_bits)
        .map_err(|e| LayerError::Other {
            layer: LayerKind::Sigmoid,
            msg: format!("range_check delta_high failed: {e}"),
        })?;

    table.query(idx, vec![out]);

    Ok(out)
}
