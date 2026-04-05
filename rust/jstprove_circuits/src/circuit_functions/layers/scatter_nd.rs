use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

#[derive(Debug)]
pub struct ScatterNDLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    update_map: HashMap<usize, usize>,
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

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ScatterNDLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let data_name = self
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ScatterND,
                param: "data input".to_string(),
            })?;
        let updates_name = self
            .inputs
            .get(2)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ScatterND,
                param: "updates input".to_string(),
            })?;

        let data = input
            .get(data_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ScatterND,
                name: data_name.clone(),
            })?;
        let updates = input
            .get(updates_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ScatterND,
                name: updates_name.clone(),
            })?;

        let data_flat = data.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::ScatterND,
            msg: "data tensor is not contiguous".to_string(),
        })?;
        let updates_flat = updates.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::ScatterND,
            msg: "updates tensor is not contiguous".to_string(),
        })?;

        let mut out_flat: Vec<Variable> = data_flat.to_vec();

        for (&data_idx, &update_idx) in &self.update_map {
            if data_idx < out_flat.len() && update_idx < updates_flat.len() {
                out_flat[data_idx] = updates_flat[update_idx];
            }
        }

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::ScatterND,
                    msg: format!("ScatterND output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        if layer.inputs.len() != 3 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::ScatterND,
                param: format!(
                    "ScatterND expects 3 inputs (data, indices, updates), got {}",
                    layer.inputs.len()
                ),
            }
            .into());
        }

        let data_name = &layer.inputs[0];
        let indices_name = &layer.inputs[1];

        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ScatterND,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ScatterND,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let data_shape = layer_context
            .shapes_map
            .get(data_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ScatterND,
                msg: format!("missing data shape for '{data_name}'"),
            })?
            .clone();

        let indices_array: ndarray::ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, indices_name).map_err(|e| LayerError::Other {
                layer: LayerKind::ScatterND,
                msg: format!(
                    "ScatterND indices '{indices_name}' must be compile-time constant: {e}"
                ),
            })?;

        let indices_shape = indices_array.shape().to_vec();
        let indices_rank = indices_shape.len();
        if indices_rank < 1 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ScatterND,
                msg: "ScatterND indices must have at least 1 dimension".to_string(),
            }
            .into());
        }

        let last_dim = indices_shape[indices_rank - 1];
        if last_dim > data_shape.len() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ScatterND,
                msg: format!(
                    "ScatterND: indices last dim {last_dim} > data rank {}",
                    data_shape.len()
                ),
            }
            .into());
        }

        let num_updates: usize = indices_shape[..indices_rank - 1].iter().product();
        let slice_size: usize = if last_dim < data_shape.len() {
            data_shape[last_dim..].iter().product()
        } else {
            1
        };

        let indices_flat = indices_array
            .as_slice()
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::ScatterND,
                msg: "indices tensor is not contiguous".to_string(),
            })?;

        let mut update_map = HashMap::new();

        for update_idx in 0..num_updates {
            let coord_start = update_idx * last_dim;
            let coords: Vec<usize> = (0..last_dim)
                .map(|j| {
                    let v = indices_flat[coord_start + j];
                    if v < 0 {
                        (v + data_shape[j] as i64) as usize
                    } else {
                        v as usize
                    }
                })
                .collect();

            let mut base_flat = ravel(&coords, &data_shape[..last_dim]);
            let remaining_stride: usize = if last_dim < data_shape.len() {
                data_shape[last_dim..].iter().product()
            } else {
                1
            };
            base_flat *= remaining_stride;

            for s in 0..slice_size {
                let data_flat_idx = base_flat + s;
                let updates_flat_idx = update_idx * slice_size + s;
                update_map.insert(data_flat_idx, updates_flat_idx);
            }
        }

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            update_map,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ravel_basic() {
        let shape = [3usize, 4];
        assert_eq!(ravel(&[0, 0], &shape), 0);
        assert_eq!(ravel(&[1, 2], &shape), 6);
        assert_eq!(ravel(&[2, 3], &shape), 11);
    }

    #[test]
    fn scatter_nd_update_map_simple() {
        let _data_shape = [8usize];
        let indices: Vec<i64> = vec![4, 3, 1, 7];
        let _last_dim = 1;
        let num_updates = 4;
        let _slice_size = 1;

        let mut update_map = HashMap::new();
        for (update_idx, &idx) in indices.iter().enumerate().take(num_updates) {
            let v = idx as usize;
            let data_flat_idx = v;
            update_map.insert(data_flat_idx, update_idx);
        }

        assert_eq!(*update_map.get(&4).unwrap(), 0);
        assert_eq!(*update_map.get(&3).unwrap(), 1);
        assert_eq!(*update_map.get(&1).unwrap(), 2);
        assert_eq!(*update_map.get(&7).unwrap(), 3);
    }
}
