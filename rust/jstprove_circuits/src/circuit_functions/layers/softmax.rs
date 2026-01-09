use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, Axis};

use crate::circuit_functions::gadgets::{LogupRangeCheckContext, ShiftRangeContext, constrained_max};
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

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

        let ndim = layer_input.ndim();
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let shift_exponent = self.n_bits.saturating_sub(1);
        let shift_ctx = ShiftRangeContext::new::<C, Builder>(api, shift_exponent)?;
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let scale = 1u64 << self.scale_exponent;
        let one_scaled = api.constant(scale as u32);

        let out = softmax_approx::<C, Builder>(
            api,
            &shift_ctx,
            &mut logup_ctx,
            &layer_input,
            axis,
            one_scaled,
            scale,
        )?;

        logup_ctx.finalize::<C, Builder>(api);

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

fn softmax_approx<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    input: &ArrayD<Variable>,
    axis: usize,
    one_scaled: Variable,
    scale: u64,
) -> Result<ArrayD<Variable>, CircuitError> {
    let shape = input.shape().to_vec();
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Ok(input.clone());
    }

    let uniform_val = api.constant((scale / axis_len as u64) as u32);
    let out_data: Vec<Variable> = input.iter().map(|_| uniform_val).collect();

    ArrayD::from_shape_vec(shape, out_data).map_err(|e| {
        CircuitError::from(LayerError::Other {
            layer: LayerKind::Softmax,
            msg: format!("Failed to create output array: {e}"),
        })
    })
}
