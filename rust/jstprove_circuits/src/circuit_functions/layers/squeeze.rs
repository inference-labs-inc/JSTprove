use std::collections::{HashMap, HashSet};

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{AXES, INPUT},
        onnx_model::{extract_params, get_input_name, get_param},
        shaping::normalize_axes,
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

        let axes_u = normalize_axes(axes_ref.as_slice(), rank, &LayerKind::Squeeze, &self.name)?;

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
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let params = extract_params(layer).map_err(|e| LayerError::Other {
            layer: LayerKind::Squeeze,
            msg: format!("extract_params failed: {e}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(axes: Option<Vec<i64>>) -> SqueezeLayer {
        SqueezeLayer {
            name: "test".to_string(),
            axes,
            inputs: vec![],
            outputs: vec![],
        }
    }

    #[test]
    fn squeeze_no_axes_removes_all_unit_dims() {
        let l = layer(None);
        assert_eq!(l.squeezed_shape(&[1, 3, 1, 4], None).unwrap(), vec![3, 4]);
    }

    #[test]
    fn squeeze_no_axes_no_unit_dims_unchanged() {
        let l = layer(None);
        assert_eq!(l.squeezed_shape(&[2, 3, 4], None).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn squeeze_explicit_axes_removes_specified_dims() {
        let axes = vec![0i64, 2];
        let l = layer(Some(axes.clone()));
        assert_eq!(
            l.squeezed_shape(&[1, 3, 1, 4], Some(&axes)).unwrap(),
            vec![3, 4]
        );
    }

    #[test]
    fn squeeze_explicit_single_axis() {
        let axes = vec![0i64];
        let l = layer(Some(axes.clone()));
        assert_eq!(
            l.squeezed_shape(&[1, 3, 2, 4], Some(&axes)).unwrap(),
            vec![3, 2, 4]
        );
    }

    #[test]
    fn squeeze_axis_on_nonunit_dim_errors() {
        let axes = vec![1i64];
        let l = layer(Some(axes.clone()));
        assert!(l.squeezed_shape(&[1, 3, 1, 4], Some(&axes)).is_err());
    }

    #[test]
    fn squeeze_negative_axis_resolves_correctly() {
        let axes = vec![-1i64];
        let l = layer(Some(axes.clone()));
        assert_eq!(
            l.squeezed_shape(&[3, 2, 1], Some(&axes)).unwrap(),
            vec![3, 2]
        );
    }

    #[test]
    fn squeeze_empty_shape_no_axes_returns_empty() {
        let l = layer(None);
        let result = l.squeezed_shape(&[], None).unwrap();
        assert!(result.is_empty());
    }
}
