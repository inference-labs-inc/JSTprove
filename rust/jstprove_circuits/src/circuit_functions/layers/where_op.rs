use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::{get_optional_w_or_b, get_w_or_b_or_constant},
};

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

fn broadcast_index(out_coords: &[usize], shape: &[usize], out_rank: usize) -> usize {
    let rank = shape.len();
    let pad = out_rank - rank;
    let coords: Vec<usize> = out_coords[pad..]
        .iter()
        .zip(shape.iter())
        .map(|(&oc, &s)| if s == 1 { 0 } else { oc })
        .collect();
    ravel(&coords, shape)
}

enum WhereMode {
    Precomputed {
        mask: Vec<bool>,
        x_indices: Vec<usize>,
        y_indices: Vec<usize>,
    },
    Runtime {
        cond_initializer: Option<ArrayD<i64>>,
        cond_shape: Vec<usize>,
        x_shape: Vec<usize>,
        y_shape: Vec<usize>,
    },
}

pub struct WhereLayer {
    #[allow(dead_code)]
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    cond_name: String,
    x_name: String,
    y_name: String,
    mode: WhereMode,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for WhereLayer {
    #[allow(clippy::too_many_lines)]
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_input = input
            .get(&self.x_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Where,
                name: self.x_name.clone(),
            })?;
        let y_input = input
            .get(&self.y_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Where,
                name: self.y_name.clone(),
            })?;

        let x_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Where,
            msg: "X tensor is not contiguous".to_string(),
        })?;
        let y_flat = y_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Where,
            msg: "Y tensor is not contiguous".to_string(),
        })?;

        let out_flat = match &self.mode {
            WhereMode::Precomputed {
                mask,
                x_indices,
                y_indices,
            } => mask
                .iter()
                .enumerate()
                .map(|(i, &cond)| {
                    if cond {
                        x_flat[x_indices[i]]
                    } else {
                        y_flat[y_indices[i]]
                    }
                })
                .collect::<Vec<Variable>>(),
            WhereMode::Runtime {
                cond_initializer,
                cond_shape,
                x_shape,
                y_shape,
            } => {
                let total_out: usize = self.output_shape.iter().product();
                let out_rank = self.output_shape.len();

                let cond_flat: Vec<Variable> = if let Some(init) = cond_initializer {
                    let zero = api.constant(0u32);
                    let one = api.constant(1u32);
                    init.iter()
                        .map(|&v| if v != 0 { one } else { zero })
                        .collect()
                } else {
                    let cond_input =
                        input
                            .get(&self.cond_name)
                            .ok_or_else(|| LayerError::MissingInput {
                                layer: LayerKind::Where,
                                name: self.cond_name.clone(),
                            })?;
                    let raw = cond_input
                        .as_slice()
                        .ok_or_else(|| LayerError::InvalidShape {
                            layer: LayerKind::Where,
                            msg: "condition tensor is not contiguous".to_string(),
                        })?;
                    let one = api.constant(1u32);
                    raw.iter()
                        .map(|&v| {
                            let iz = api.is_zero(v);
                            api.sub(one, iz)
                        })
                        .collect()
                };

                let mut out_vars = Vec::with_capacity(total_out);
                for out_flat_idx in 0..total_out {
                    let out_coords = unravel(out_flat_idx, &self.output_shape);
                    let ci = broadcast_index(&out_coords, cond_shape, out_rank);
                    let xi = broadcast_index(&out_coords, x_shape, out_rank);
                    let yi = broadcast_index(&out_coords, y_shape, out_rank);

                    let c = cond_flat[ci];
                    let x_val = x_flat[xi];
                    let y_val = y_flat[yi];
                    let diff = api.sub(x_val, y_val);
                    let selected = api.mul(c, diff);
                    let result = api.add(selected, y_val);
                    out_vars.push(result);
                }
                out_vars
            }
        };

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Where,
                    msg: format!("Where output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(clippy::too_many_lines, clippy::uninlined_format_args)]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let cond_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Where,
                param: "condition input".to_string(),
            })?
            .clone();
        let x_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Where,
                param: "X input".to_string(),
            })?
            .clone();
        let y_name = layer
            .inputs
            .get(2)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Where,
                param: "Y input".to_string(),
            })?
            .clone();

        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Where,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Where,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let cond_shape = layer_context
            .shapes_map
            .get(cond_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Where,
                msg: format!("missing condition shape for '{cond_name}'"),
            })?
            .clone();

        let x_shape = layer_context
            .shapes_map
            .get(x_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Where,
                msg: format!("missing X shape for '{x_name}'"),
            })?
            .clone();

        let y_shape = layer_context
            .shapes_map
            .get(y_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Where,
                msg: format!("missing Y shape for '{y_name}'"),
            })?
            .clone();

        let cond_const = get_w_or_b_or_constant(layer_context, &cond_name).ok();

        let mode = if let Some(cond_data) = cond_const {
            let cond_flat = cond_data
                .as_slice()
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Where,
                    msg: "condition tensor is not contiguous".to_string(),
                })?;

            let total_out: usize = output_shape.iter().product();
            let out_rank = output_shape.len();

            let mut mask = Vec::with_capacity(total_out);
            let mut x_indices = Vec::with_capacity(total_out);
            let mut y_indices = Vec::with_capacity(total_out);

            for out_flat_idx in 0..total_out {
                let out_coords = unravel(out_flat_idx, &output_shape);
                let cond_idx = broadcast_index(&out_coords, &cond_shape, out_rank);
                let xi = broadcast_index(&out_coords, &x_shape, out_rank);
                let yi = broadcast_index(&out_coords, &y_shape, out_rank);

                mask.push(cond_flat[cond_idx] != 0);
                x_indices.push(xi);
                y_indices.push(yi);
            }

            WhereMode::Precomputed {
                mask,
                x_indices,
                y_indices,
            }
        } else {
            let cond_opt =
                get_optional_w_or_b(layer_context, &cond_name).map_err(|e| LayerError::Other {
                    layer: LayerKind::Where,
                    msg: format!("Where condition '{}': {}", cond_name, e),
                })?;

            WhereMode::Runtime {
                cond_initializer: cond_opt,
                cond_shape: cond_shape.clone(),
                x_shape: x_shape.clone(),
                y_shape: y_shape.clone(),
            }
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            cond_name,
            x_name,
            y_name,
            mode,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unravel_ravel_roundtrip() {
        let shape = [3usize, 4, 5];
        for flat in 0..60 {
            let coords = unravel(flat, &shape);
            assert_eq!(ravel(&coords, &shape), flat);
        }
    }

    #[test]
    fn broadcast_index_no_broadcast() {
        let out_coords = [1usize, 2, 3];
        let shape = [4usize, 5, 6];
        let idx = broadcast_index(&out_coords, &shape, 3);
        assert_eq!(idx, ravel(&out_coords, &shape));
    }

    #[test]
    fn broadcast_index_scalar_broadcast() {
        let shape = [1usize];
        for oc in 0..5 {
            let out_coords = [oc];
            assert_eq!(broadcast_index(&out_coords, &shape, 1), 0);
        }
    }

    #[test]
    fn broadcast_index_leading_dims_padded() {
        let shape = [3usize];
        let out_rank = 2;
        assert_eq!(broadcast_index(&[0, 0], &shape, out_rank), 0);
        assert_eq!(broadcast_index(&[2, 1], &shape, out_rank), 1);
        assert_eq!(broadcast_index(&[3, 2], &shape, out_rank), 2);
    }

    #[test]
    fn broadcast_index_size_one_dim_collapses() {
        let shape = [1usize, 3];
        let out_rank = 2;
        assert_eq!(broadcast_index(&[0, 0], &shape, out_rank), 0);
        assert_eq!(broadcast_index(&[5, 2], &shape, out_rank), 2);
    }
}
