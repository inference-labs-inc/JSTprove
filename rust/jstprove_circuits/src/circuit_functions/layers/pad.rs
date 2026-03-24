// ONNX `Pad` layer for ZK circuits.
//
// # ZK approach
// Pad inserts constant values (always 0 in ZK context) around a tensor.
// This is purely structural: the output is formed by selecting from the input
// or from a constant zero. No hint, no range check.
//
// A compile-time `index_map` is built during `build()`. Each output position
// maps to either a valid input position (Some(idx)) or a pad position (None → zero).
//
// # Supported modes
// Only mode="constant" is supported. "reflect" and "edge" are rejected.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
    },
};

// -------- Struct --------

pub struct PadLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    /// index_map[out_flat] = Some(in_flat) or None (pad with zero).
    index_map: Vec<Option<usize>>,
}

// -------- Index helpers --------

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            coords[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
    coords
}

fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += coords[i] * stride;
        stride *= shape[i];
    }
    flat
}

fn build_pad_index_map(
    input_shape: &[usize],
    output_shape: &[usize],
    pads_begin: &[usize],
) -> Vec<Option<usize>> {
    let total_out: usize = output_shape.iter().product();
    let rank = input_shape.len();

    (0..total_out)
        .map(|out_flat| {
            let out_coords = unravel(out_flat, output_shape);
            // Check if this position is in the input (not padded).
            let mut in_coords = Vec::with_capacity(rank);
            for ax in 0..rank {
                let in_pos = out_coords[ax].checked_sub(pads_begin[ax]);
                match in_pos {
                    Some(pos) if pos < input_shape[ax] => in_coords.push(pos),
                    _ => return None, // padded position
                }
            }
            Some(ravel(&in_coords, input_shape))
        })
        .collect()
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for PadLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Pad, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Pad,
            name: x_name.to_string(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Pad,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        // Constant zero field element for pad positions.
        let zero = api.constant(0);

        let out_flat: Vec<Variable> = self
            .index_map
            .iter()
            .map(|idx| match idx {
                Some(i) => data_flat[*i],
                None => zero,
            })
            .collect();

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Pad,
                    msg: format!("pad output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Pad,
                param: "data input".to_string(),
            })?;
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Pad,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Pad,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Pad,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let rank = input_shape.len();

        // Check mode (only "constant" is supported).
        let mode = get_mode_param(layer);
        if mode != "constant" {
            return Err(LayerError::Other {
                layer: LayerKind::Pad,
                msg: format!(
                    "Pad mode '{}' is not supported in the Expander backend; only mode='constant' is implemented.",
                    mode
                ),
            }
            .into());
        }

        // Read pads from input[1] or from params.
        let pads: Vec<i64> = if let Some(pads_name) = layer.inputs.get(1).filter(|n| !n.is_empty())
        {
            if let Ok(arr) = get_w_or_b(layer_context.w_and_b_map, pads_name) {
                arr.as_slice()
                    .ok_or_else(|| LayerError::Other {
                        layer: LayerKind::Pad,
                        msg: format!("pads tensor '{}' is not contiguous", pads_name),
                    })?
                    .to_vec()
            } else if let Some(rmpv::Value::Map(entries)) = layer.params.as_ref() {
                entries
                    .iter()
                    .find_map(|(k, v)| {
                        if let rmpv::Value::String(s) = k {
                            if s.as_str() == Some(pads_name.as_str()) {
                                return parse_i64_vec(v);
                            }
                        }
                        None
                    })
                    .ok_or_else(|| LayerError::Other {
                        layer: LayerKind::Pad,
                        msg: format!(
                            "pads input '{}' not found in initializers or params",
                            pads_name
                        ),
                    })?
            } else {
                return Err(LayerError::Other {
                    layer: LayerKind::Pad,
                    msg: format!("pads input '{}' not found", pads_name),
                }
                .into());
            }
        } else {
            // Try attribute.
            layer
                .params
                .as_ref()
                .and_then(|p| {
                    if let rmpv::Value::Map(m) = p {
                        m.iter().find_map(|(k, v)| {
                            if k == &rmpv::Value::String("pads".into()) {
                                parse_i64_vec(v)
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                })
                .ok_or_else(|| LayerError::Other {
                    layer: LayerKind::Pad,
                    msg: "Pad has no pads attribute or input".to_string(),
                })?
        };

        if pads.len() != 2 * rank {
            return Err(LayerError::Other {
                layer: LayerKind::Pad,
                msg: format!("Pad pads length {} != 2 * rank {}", pads.len(), rank),
            }
            .into());
        }

        // Validate constant_value (input[2]): only zero padding is supported.
        if let Some(cv_name) = layer.inputs.get(2).filter(|n| !n.is_empty()) {
            if let Ok(cv) = get_w_or_b::<f64, _>(layer_context.w_and_b_map, cv_name) {
                let has_nonzero = cv.iter().any(|&v| v != 0.0);
                if has_nonzero {
                    return Err(LayerError::Other {
                        layer: LayerKind::Pad,
                        msg: format!(
                            "Pad constant_value input '{}' is non-zero; only zero padding is supported",
                            cv_name
                        ),
                    }
                    .into());
                }
            }
        }

        if let Some(&neg) = pads.iter().find(|&&v| v < 0) {
            return Err(LayerError::Other {
                layer: LayerKind::Pad,
                msg: format!(
                    "Pad contains negative pad value {neg}; negative (cropping) pads are not supported"
                ),
            }
            .into());
        }

        let pads_begin: Vec<usize> = pads[..rank].iter().map(|&v| v as usize).collect();

        let index_map = build_pad_index_map(&input_shape, &output_shape, &pads_begin);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            index_map,
        }))
    }
}

fn parse_i64_vec(v: &rmpv::Value) -> Option<Vec<i64>> {
    match v {
        rmpv::Value::Array(xs) => xs
            .iter()
            .map(|x| {
                if let rmpv::Value::Integer(i) = x {
                    i.as_i64()
                } else {
                    None
                }
            })
            .collect(),
        rmpv::Value::Integer(i) => i.as_i64().map(|x| vec![x]),
        _ => None,
    }
}

fn get_mode_param(layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer) -> String {
    layer
        .params
        .as_ref()
        .and_then(|p| {
            if let rmpv::Value::Map(m) = p {
                m.iter().find_map(|(k, v)| {
                    if k == &rmpv::Value::String("mode".into()) {
                        if let rmpv::Value::String(s) = v {
                            s.as_str().map(|x| x.to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        })
        .unwrap_or_else(|| "constant".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_index_map_1d() {
        // input [3], pads_begin=[1], pads_end=[2] → output [6]
        // output: [pad, in[0], in[1], in[2], pad, pad]
        let input_shape = [3usize];
        let output_shape = [6usize];
        let pads_begin = [1usize];
        let map = build_pad_index_map(&input_shape, &output_shape, &pads_begin);
        assert_eq!(map, vec![None, Some(0), Some(1), Some(2), None, None]);
    }

    #[test]
    fn pad_index_map_2d() {
        // input [2,2], pads_begin=[1,0] → output [3,2]
        let input_shape = [2usize, 2];
        let output_shape = [3usize, 2];
        let pads_begin = [1usize, 0];
        let map = build_pad_index_map(&input_shape, &output_shape, &pads_begin);
        // row 0: pad, pad
        // row 1: in[0,0], in[0,1]
        // row 2: in[1,0], in[1,1]
        assert_eq!(map, vec![None, None, Some(0), Some(1), Some(2), Some(3)]);
    }
}
