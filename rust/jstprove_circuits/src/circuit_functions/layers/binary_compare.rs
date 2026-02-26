use std::collections::HashMap;

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, ShiftRangeContext, constrained_max, constrained_min},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_optional_w_or_b},
        tensor_ops::{broadcast_two_arrays, load_array_constants_or_get_inputs},
    },
};

#[derive(Debug)]
pub struct BinaryCompareLayer {
    kind: LayerKind,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
    shift_exponent: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for BinaryCompareLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let a_name = get_input_name(&self.inputs, 0, self.kind.clone(), INPUT)?;
        let b_name = get_input_name(&self.inputs, 1, self.kind.clone(), INPUT)?;

        let a_input = load_array_constants_or_get_inputs(
            api,
            input,
            a_name,
            &self.initializer_a,
            self.kind.clone(),
        )?;

        let b_input = load_array_constants_or_get_inputs(
            api,
            input,
            b_name,
            &self.initializer_b,
            self.kind.clone(),
        )?;

        let (a_bc, b_bc) = broadcast_two_arrays(&a_input, &b_input)?;

        let shift_ctx =
            ShiftRangeContext::new(api, self.shift_exponent).map_err(|e| LayerError::Other {
                layer: self.kind.clone(),
                msg: format!("ShiftRangeContext::new failed: {e}"),
            })?;

        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let shape = a_bc.shape().to_vec();
        if a_bc.len() != b_bc.len() {
            return Err(LayerError::InvalidShape {
                layer: self.kind.clone(),
                msg: format!(
                    "broadcast_two_arrays returned arrays of different sizes: {:?} vs {:?}",
                    a_bc.shape(),
                    b_bc.shape()
                ),
            }
            .into());
        }

        let mut out_storage = Vec::with_capacity(a_bc.len());

        for (a_val, b_val) in a_bc.iter().zip(b_bc.iter()) {
            let result_var = match self.kind {
                LayerKind::Max => {
                    constrained_max(api, &shift_ctx, &mut logup_ctx, &[*a_val, *b_val])?
                }
                LayerKind::Min => {
                    constrained_min(api, &shift_ctx, &mut logup_ctx, &[*a_val, *b_val])?
                }
                _ => unreachable!("BinaryCompareLayer only supports Max and Min"),
            };
            out_storage.push(result_var);
        }

        logup_ctx.finalize::<C, Builder>(api);

        let result = ArrayD::from_shape_vec(shape.clone(), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: self.kind.clone(),
                msg: format!("cannot reshape result into shape {shape:?}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let initializer_a = get_optional_w_or_b(layer_context, &layer.inputs[0])?;
        let initializer_b = get_optional_w_or_b(layer_context, &layer.inputs[1])?;

        let kind = LayerKind::try_from(layer.op_type.as_str())?;

        let shift_exponent = layer_context
            .n_bits_for(&layer.name)
            .checked_sub(1)
            .ok_or_else(|| LayerError::Other {
                layer: kind.clone(),
                msg: "n_bits too small to derive shift_exponent".to_string(),
            })?;

        Ok(Box::new(Self {
            kind,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
            shift_exponent,
        }))
    }
}
