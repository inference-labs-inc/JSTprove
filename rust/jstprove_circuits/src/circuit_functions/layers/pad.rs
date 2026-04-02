//! ONNX `Pad` layer implementation for JSTprove.
//!
//! Pads an input tensor with a constant value. Supports `constant` mode only
//! (the most common in ML models). The padding specification follows ONNX
//! convention: `[x1_begin, x2_begin, ..., x1_end, x2_end, ...]`.
//!
//! No rescaling is needed since padding is a pure data-movement operation.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::{extract_params, get_input_name, get_param_or_default, get_w_or_b},
};

#[derive(Debug)]
pub struct PadLayer {
    pads: Vec<i64>,
    constant_value: i64,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for PadLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Pad, "data")?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Pad,
                name: input_name.clone(),
            })?
            .clone();

        let shape = layer_input.shape();
        let rank = shape.len();
        let pads = &self.pads;

        if pads.len() != 2 * rank {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Pad,
                msg: format!(
                    "pads length {} does not match 2 * rank {}",
                    pads.len(),
                    2 * rank
                ),
            }
            .into());
        }

        let mut new_shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let begin = pads[i];
            let end = pads[i + rank];
            if begin < 0 || end < 0 {
                return Err(LayerError::UnsupportedConfig {
                    layer: LayerKind::Pad,
                    msg: "Negative padding (cropping) is not supported".to_string(),
                }
                .into());
            }
            new_shape.push(
                (begin as usize)
                    .checked_add(shape[i])
                    .and_then(|s| s.checked_add(end as usize))
                    .ok_or_else(|| LayerError::InvalidShape {
                        layer: LayerKind::Pad,
                        msg: format!("output shape overflow at dim {i}"),
                    })?,
            );
        }

        let pad_val = api.constant(u32::try_from(self.constant_value).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Pad,
                msg: format!("constant_value {} does not fit in u32", self.constant_value),
            }
        })?);
        let mut output = ArrayD::from_elem(IxDyn(&new_shape), pad_val);

        let mut indices = vec![0usize; rank];
        let total = layer_input.len();
        for flat_idx in 0..total {
            let mut remainder = flat_idx;
            for d in (0..rank).rev() {
                indices[d] = remainder % shape[d];
                remainder /= shape[d];
            }

            let mut out_indices: Vec<usize> = Vec::with_capacity(rank);
            for d in 0..rank {
                out_indices.push(indices[d] + pads[d] as usize);
            }

            output[IxDyn(&out_indices)] = layer_input[IxDyn(&indices)];
        }

        Ok((self.outputs.clone(), output))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // Reject non-constant padding modes early — only "constant" is implemented.
        match extract_params(layer) {
            Ok(params) => {
                let default_mode = "constant".to_string();
                let mode = get_param_or_default::<String>(
                    &layer.name,
                    "mode",
                    &params,
                    Some(&default_mode),
                )
                .unwrap_or(default_mode);
                if mode != "constant" {
                    return Err(LayerError::UnsupportedConfig {
                        layer: LayerKind::Pad,
                        msg: format!(
                            "Pad mode '{mode}' is not supported; only 'constant' mode is implemented"
                        ),
                    }
                    .into());
                }
            }
            Err(e) => {
                return Err(LayerError::Other {
                    layer: LayerKind::Pad,
                    msg: format!("failed to extract params for mode check: {e}"),
                }
                .into());
            }
        }

        // ONNX Pad: inputs are [data, pads, constant_value(optional)]
        // pads is an initializer tensor of shape [2*rank]
        let pads_name = get_input_name(&layer.inputs, 1, LayerKind::Pad, "pads")?;
        let pads_array = get_w_or_b(layer_context.w_and_b_map, pads_name)?;
        let pads: Vec<i64> = pads_array
            .as_slice()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Pad,
                msg: format!("pads initializer '{pads_name}' is not a contiguous i64 array"),
            })?
            .to_vec();

        let constant_value: i64 = if layer.inputs.len() > 2 {
            let cv_name = &layer.inputs[2];
            if !cv_name.is_empty() {
                let arr = get_w_or_b(layer_context.w_and_b_map, cv_name)?;
                arr.as_slice()
                    .and_then(|s| s.first().copied())
                    .ok_or_else(|| LayerError::Other {
                        layer: LayerKind::Pad,
                        msg: format!("cannot extract constant_value from initializer '{cv_name}'"),
                    })?
            } else {
                0
            }
        } else {
            0
        };

        Ok(Box::new(Self {
            pads,
            constant_value,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
