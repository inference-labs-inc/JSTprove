use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::{extract_params, get_param_or_default, get_w_or_b},
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct GatherElementsLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    data_shape: Vec<usize>,
    axis: usize,
    flat_index_map: Vec<usize>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GatherElementsLayer {
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
                layer: LayerKind::GatherElements,
                param: "data input".to_string(),
            })?;

        let data = input
            .get(data_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::GatherElements,
                name: data_name.clone(),
            })?;

        let data_flat = data.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::GatherElements,
            msg: "data tensor is not contiguous".to_string(),
        })?;

        let out_flat: Vec<Variable> = self
            .flat_index_map
            .iter()
            .map(|&idx| data_flat[idx])
            .collect();

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::GatherElements,
                    msg: format!("GatherElements output reshape failed: {e}"),
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
        if layer.inputs.len() < 2 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::GatherElements,
                param: format!(
                    "GatherElements expects 2 inputs (data, indices), got {}",
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
                layer: LayerKind::GatherElements,
                param: "output tensor".to_string(),
            })?;

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GatherElements,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let data_shape = layer_context
            .shapes_map
            .get(data_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GatherElements,
                msg: format!("missing data shape for '{data_name}'"),
            })?
            .clone();

        let raw_axis: i64 = match extract_params(layer).ok() {
            Some(params) => get_param_or_default(&layer.name, "axis", &params, Some(&0i64))
                .map_err(|e| LayerError::Other {
                    layer: LayerKind::GatherElements,
                    msg: format!("failed to read 'axis' attribute: {e}"),
                })?,
            None => 0,
        };

        let rank = data_shape.len();
        let axis = if raw_axis < 0 {
            let a = rank as i64 + raw_axis;
            if a < 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::GatherElements,
                    msg: format!("axis {raw_axis} out of range for data rank {rank}"),
                }
                .into());
            }
            a as usize
        } else {
            let a = raw_axis as usize;
            if a >= rank {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::GatherElements,
                    msg: format!("axis {raw_axis} out of range for data rank {rank}"),
                }
                .into());
            }
            a
        };

        let indices_array: ndarray::ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, indices_name).map_err(|e| LayerError::Other {
                layer: LayerKind::GatherElements,
                msg: format!(
                    "GatherElements indices '{indices_name}' must be compile-time constant: {e}"
                ),
            })?;

        let indices_shape = indices_array.shape().to_vec();
        let indices_flat = indices_array
            .as_slice()
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GatherElements,
                msg: "indices tensor is not contiguous".to_string(),
            })?;

        let total_out: usize = indices_shape.iter().product();
        let mut flat_index_map = Vec::with_capacity(total_out);

        let mut data_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        let mut indices_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
        }

        for (flat_idx, idx_entry) in indices_flat.iter().enumerate().take(total_out) {
            let mut remaining = flat_idx;
            let mut data_flat = 0usize;
            for dim in 0..rank {
                let coord = remaining / indices_strides[dim];
                remaining %= indices_strides[dim];

                if dim == axis {
                    let idx_val = *idx_entry;
                    let resolved = if idx_val < 0 {
                        (idx_val + data_shape[axis] as i64) as usize
                    } else {
                        idx_val as usize
                    };
                    data_flat += resolved * data_strides[dim];
                } else {
                    data_flat += coord * data_strides[dim];
                }
            }
            flat_index_map.push(data_flat);
        }

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            data_shape,
            axis,
            flat_index_map,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gather_elements_axis0_index_map() {
        let data_shape = [3usize, 3];
        let indices: [i64; 4] = [1, 2, 0, 2];
        let indices_shape = [2usize, 2];
        let axis = 0;

        let mut data_strides = [1usize; 2];
        data_strides[0] = data_shape[1];

        let mut indices_strides = [1usize; 2];
        indices_strides[0] = indices_shape[1];

        let total = 4;
        let mut map = Vec::with_capacity(total);
        for (flat_idx, &index_val) in indices.iter().enumerate().take(total) {
            let mut remaining = flat_idx;
            let mut data_flat = 0usize;
            for dim in 0..2 {
                let coord = remaining / indices_strides[dim];
                remaining %= indices_strides[dim];
                if dim == axis {
                    let resolved = index_val as usize;
                    data_flat += resolved * data_strides[dim];
                } else {
                    data_flat += coord * data_strides[dim];
                }
            }
            map.push(data_flat);
        }

        assert_eq!(map, vec![3, 7, 0, 7]);
    }
}
