use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::gather_elements::GATHER_ELEMENTS_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::{extract_params, get_param_or_default, get_w_or_b_or_constant},
};

enum GatherElementsMode {
    Precomputed {
        flat_index_map: Vec<usize>,
    },
    Dynamic {
        indices_name: String,
        indices_shape: Vec<usize>,
        axis: usize,
        n_bits: usize,
    },
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct GatherElementsLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    data_shape: Vec<usize>,
    axis: usize,
    mode: GatherElementsMode,
}

impl std::fmt::Debug for GatherElementsMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Precomputed { flat_index_map } => f
                .debug_struct("Precomputed")
                .field("flat_index_map_len", &flat_index_map.len())
                .finish(),
            Self::Dynamic { indices_name, .. } => f
                .debug_struct("Dynamic")
                .field("indices_name", indices_name)
                .finish(),
        }
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GatherElementsLayer {
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
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

        let out_flat: Vec<Variable> = match &self.mode {
            GatherElementsMode::Precomputed { flat_index_map } => {
                let data_len = data_flat.len();
                flat_index_map
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| {
                        if idx >= data_len {
                            return Err(LayerError::InvalidShape {
                                layer: LayerKind::GatherElements,
                                msg: format!(
                                    "flat index {idx} (output position {i}) out of bounds \
                                     for data of length {data_len}"
                                ),
                            }
                            .into());
                        }
                        Ok(data_flat[idx])
                    })
                    .collect::<Result<Vec<_>, CircuitError>>()?
            }
            GatherElementsMode::Dynamic {
                indices_name,
                indices_shape,
                axis,
                n_bits,
            } => {
                let indices_input =
                    input
                        .get(indices_name)
                        .ok_or_else(|| LayerError::MissingInput {
                            layer: LayerKind::GatherElements,
                            name: indices_name.clone(),
                        })?;

                let indices_flat =
                    indices_input
                        .as_slice()
                        .ok_or_else(|| LayerError::InvalidShape {
                            layer: LayerKind::GatherElements,
                            msg: "indices tensor is not contiguous".to_string(),
                        })?;

                let rank = self.data_shape.len();
                let mut data_strides = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    data_strides[i] = data_strides[i + 1] * self.data_shape[i + 1];
                }

                let mut indices_strides = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
                }

                let axis_size = self.data_shape[*axis];
                let axis_size_var = api.constant(axis_size as u32);

                let total_out: usize = indices_shape.iter().product();
                let axis_data_stride = data_strides[*axis];
                let mut out_vars = Vec::with_capacity(total_out);
                for (flat_idx, &idx_var) in indices_flat.iter().enumerate().take(total_out) {
                    let mut remaining = flat_idx;
                    let mut base_flat = 0usize;
                    for dim in 0..rank {
                        let coord = remaining / indices_strides[dim];
                        remaining %= indices_strides[dim];
                        if dim != *axis {
                            base_flat += coord * data_strides[dim];
                        }
                    }

                    let axis_slice: Vec<Variable> = (0..axis_size)
                        .map(|a| data_flat[base_flat + a * axis_data_stride])
                        .collect();
                    let mut hint_inputs = Vec::with_capacity(2 + axis_size);
                    hint_inputs.push(axis_size_var);
                    hint_inputs.push(idx_var);
                    hint_inputs.extend_from_slice(&axis_slice);

                    let hint_out = api.new_hint(GATHER_ELEMENTS_HINT_KEY, &hint_inputs, 1);
                    let y = hint_out[0];

                    let idx_wrapped = api.add(idx_var, axis_size_var);
                    let mut mux_sum = api.constant(0u32);
                    for (a, &elem) in axis_slice.iter().enumerate() {
                        let a_var = api.constant(a as u32);
                        let diff_pos = api.sub(idx_var, a_var);
                        let match_pos = api.is_zero(diff_pos);
                        let diff_neg = api.sub(idx_wrapped, a_var);
                        let match_neg = api.is_zero(diff_neg);
                        let matched = api.add(match_pos, match_neg);
                        let selected = api.mul(matched, elem);
                        mux_sum = api.add(mux_sum, selected);
                    }
                    api.assert_is_equal(y, mux_sum);

                    logup_ctx.range_check::<C, Builder>(api, y, *n_bits)?;
                    out_vars.push(y);
                }

                out_vars
            }
        };

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

        let mode = if let Ok(indices_array) = get_w_or_b_or_constant(layer_context, indices_name) {
            let indices_shape = indices_array.shape().to_vec();
            let indices_flat =
                indices_array
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

            let axis_len = data_shape[axis] as i64;
            for (flat_idx, idx_entry) in indices_flat.iter().enumerate().take(total_out) {
                let mut remaining = flat_idx;
                let mut data_flat_idx = 0usize;
                for dim in 0..rank {
                    let coord = remaining / indices_strides[dim];
                    remaining %= indices_strides[dim];

                    if dim == axis {
                        let idx_val = *idx_entry;
                        if idx_val < -axis_len || idx_val >= axis_len {
                            return Err(LayerError::InvalidShape {
                                layer: LayerKind::GatherElements,
                                msg: format!(
                                    "index {idx_val} at flat position {flat_idx} out of range \
                                     [{}, {}] for axis {axis} with size {}",
                                    -axis_len,
                                    axis_len - 1,
                                    data_shape[axis]
                                ),
                            }
                            .into());
                        }
                        let resolved = if idx_val < 0 {
                            (idx_val + axis_len) as usize
                        } else {
                            idx_val as usize
                        };
                        data_flat_idx += resolved * data_strides[dim];
                    } else {
                        data_flat_idx += coord * data_strides[dim];
                    }
                }
                flat_index_map.push(data_flat_idx);
            }

            GatherElementsMode::Precomputed { flat_index_map }
        } else {
            let indices_shape = layer_context
                .shapes_map
                .get(indices_name.as_str())
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::GatherElements,
                    msg: format!("missing indices shape for '{indices_name}'"),
                })?
                .clone();

            let n_bits = layer_context.n_bits_for(&layer.name);

            GatherElementsMode::Dynamic {
                indices_name: indices_name.clone(),
                indices_shape,
                axis,
                n_bits,
            }
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            data_shape,
            axis,
            mode,
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
