use std::collections::{HashMap, HashSet};

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{AXES, INPUT},
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param},
        value_array::map_get,
    },
};

#[derive(Debug)]
pub struct SqueezeLayer {
    name: String,
    axes: Option<Vec<i64>>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl SqueezeLayer {
    fn normalize_axes(&self, axes: &[i64], rank: usize) -> Result<Vec<usize>, CircuitError> {
        let rank_i64 = i64::try_from(rank).map_err(|_| LayerError::InvalidParameterValue {
            layer: LayerKind::Squeeze,
            layer_name: self.name.clone(),
            param_name: AXES.into(),
            value: format!("rank {rank} cannot be represented as i64"),
        })?;

        let mut out: Vec<usize> = Vec::with_capacity(axes.len());
        let mut seen: HashSet<usize> = HashSet::new();

        for &a in axes {
            let ax_i64 = if a < 0 { a + rank_i64 } else { a };

            if ax_i64 < 0 || ax_i64 >= rank_i64 {
                return Err(LayerError::InvalidParameterValue {
                    layer: LayerKind::Squeeze,
                    layer_name: self.name.clone(),
                    param_name: AXES.into(),
                    value: format!("axis {a} out of range for rank {rank}"),
                }
                .into());
            }

            let ax = usize::try_from(ax_i64).map_err(|_| LayerError::InvalidParameterValue {
                layer: LayerKind::Squeeze,
                layer_name: self.name.clone(),
                param_name: AXES.into(),
                value: format!("axis {a} is not a valid usize index after normalization"),
            })?;

            if !seen.insert(ax) {
                return Err(LayerError::InvalidParameterValue {
                    layer: LayerKind::Squeeze,
                    layer_name: self.name.clone(),
                    param_name: AXES.into(),
                    value: format!("duplicate axis {ax} in axes={axes:?}"),
                }
                .into());
            }

            out.push(ax);
        }

        out.sort_unstable();
        Ok(out)
    }

    fn squeezed_shape(
        &self,
        input_shape: &[usize],
        axes: Option<&Vec<i64>>,
    ) -> Result<Vec<usize>, CircuitError> {
        let rank = input_shape.len();

        // axes omitted => remove all dims of size 1
        let Some(axes_ref) = axes else {
            let shape: Vec<usize> = input_shape.iter().copied().filter(|&d| d != 1).collect();
            return Ok(shape);
        };

        let axes_u = self.normalize_axes(axes_ref.as_slice(), rank)?;

        let axes_set: HashSet<usize> = axes_u.iter().copied().collect();

        // Validate specified axes are actually squeezable
        for &ax in &axes_u {
            let dim = input_shape[ax];
            if dim != 1 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Squeeze,
                    msg: format!(
                        "cannot squeeze axis {ax}: expected dim==1, got {dim} (shape={input_shape:?})"
                    ),
                }
                .into());
            }
        }

        let out_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if axes_set.contains(&i) { None } else { Some(d) })
            .collect();

        Ok(out_shape)
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SqueezeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Squeeze, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Squeeze,
                name: input_name.clone(),
            })?
            .clone();

        let in_shape: Vec<usize> = layer_input.shape().to_vec();
        let out_shape = self.squeezed_shape(&in_shape, self.axes.as_ref())?;

        // Reshape without changing element order.
        let flat: Vec<Variable> = layer_input.iter().copied().collect();

        let out = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::Squeeze,
                msg: format!("failed to reshape in Squeeze: {e} (out_shape={out_shape:?})"),
            }
        })?;

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Squeeze,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        // axes may be missing (axes omitted semantics)
        // When present, parse_attributes on Python side should serialize it as a list.
        // `get_param` will error if missing, so we only call it if the key exists.
        let axes: Option<Vec<i64>> = if let rmpv::Value::Map(ref entries) = params {
            if map_get(entries, AXES).is_some() {
                Some(get_param(&layer.name, AXES, &params)?)
            } else {
                None
            }
        } else if params.is_nil() {
            None
        } else {
            return Err(LayerError::Other {
                layer: LayerKind::Squeeze,
                msg: format!(
                    "invalid params container for layer {}: expected Map or Nil",
                    layer.name
                ),
            }
            .into());
        };

        let squeeze = Self {
            name: layer.name.clone(),
            axes,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };

        Ok(Box::new(squeeze))
    }
}
