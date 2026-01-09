use std::collections::HashMap;

use circuit_std_rs::logup::LogUpSingleKeyTable;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
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

        let mut logup_table = LogUpSingleKeyTable::new(SIGMOID_TABLE_BITS);
        let (table_keys, table_values) =
            build_sigmoid_table::<C, Builder>(api, scale, self.scale_exponent);
        logup_table.new_table(table_keys, table_values);

        let shape = layer_input.shape().to_vec();
        let mut results: Vec<Variable> = Vec::with_capacity(layer_input.len());

        for &x in layer_input.iter() {
            let (idx, out) =
                sigmoid_lookup_hint::<C, Builder>(api, x, scale, self.scale_exponent)?;
            logup_table.query(idx, vec![out]);
            results.push(out);
        }

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
    _scale_exponent: usize,
) -> (Vec<Variable>, Vec<Vec<Variable>>) {
    let mut keys = Vec::with_capacity(SIGMOID_TABLE_SIZE);
    let mut values = Vec::with_capacity(SIGMOID_TABLE_SIZE);

    for i in 0..SIGMOID_TABLE_SIZE {
        let x_unscaled = (i as f64 / (SIGMOID_TABLE_SIZE as f64 / 16.0)) - 8.0;
        let sigmoid_val = 1.0 / (1.0 + (-x_unscaled).exp());
        let sigmoid_scaled = (sigmoid_val * scale as f64).round() as u32;

        keys.push(api.constant(i as u32));
        values.push(vec![api.constant(sigmoid_scaled)]);
    }

    (keys, values)
}

fn sigmoid_lookup_hint<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    scale: u64,
    scale_exponent: usize,
) -> Result<(Variable, Variable), CircuitError> {
    let hint_inputs = vec![
        x,
        api.constant(scale as u32),
        api.constant(scale_exponent as u32),
    ];
    let hint_outputs = api.new_hint("myhint.sigmoidhint", &hint_inputs, 2);
    Ok((hint_outputs[0], hint_outputs[1]))
}
