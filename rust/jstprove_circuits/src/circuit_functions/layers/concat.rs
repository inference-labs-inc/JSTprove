use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, Axis};

use crate::circuit_functions::{
    CircuitError,
    layers::{
        LayerError, LayerKind,
        layer_ops::{LayerOp, LayerResult},
    },
    utils::{onnx_model::extract_params_and_expected_shape, typecasting::AsIsize},
};

#[derive(Debug)]
pub struct ConcatLayer {
    _name: String,
    axis: isize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConcatLayer {
    fn apply(&self, _api: &mut Builder, input: HashMap<String, ArrayD<Variable>>) -> LayerResult {
        let mut arrays: Vec<ArrayD<Variable>> = Vec::new();

        for input_name in &self.inputs {
            if let Some(arr) = input.get(input_name) {
                arrays.push(arr.clone());
            }
        }

        if arrays.is_empty() {
            return Err(CircuitError::from(LayerError::MissingInput {
                layer: LayerKind::Concat,
                name: self.inputs.first().cloned().unwrap_or_default(),
            }));
        }

        let first = &arrays[0];
        let ndim = first.ndim();
        let ndim_isize = ndim.as_isize().map_err(|e| LayerError::Other {
            layer: LayerKind::Concat,
            msg: e.to_string(),
        })?;

        let resolved_axis = if self.axis < 0 {
            ndim_isize
                .checked_add(self.axis)
                .ok_or_else(|| LayerError::Other {
                    layer: LayerKind::Concat,
                    msg: "axis underflow when resolving negative axis".to_string(),
                })?
        } else {
            self.axis
        };

        let axis = usize::try_from(resolved_axis).map_err(|_| LayerError::Other {
            layer: LayerKind::Concat,
            msg: format!("invalid concat axis: {resolved_axis}"),
        })?;
        // let axis = if self.axis < 0 {
        //     (ndim as isize + self.axis) as usize
        // } else {
        //     self.axis as usize
        // };
        let views: Vec<_> = arrays.iter().map(ndarray::ArrayBase::view).collect();
        let out = ndarray::concatenate(Axis(axis), &views).map_err(|e| LayerError::Other {
            layer: LayerKind::Concat,
            msg: format!("Concatenation failed: {e}"),
        })?;

        Ok(vec![(self.outputs.clone(), out)])
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
                layer: LayerKind::Concat,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let axis: isize = params
            .get("axis")
            .and_then(serde_json::Value::as_i64)
            .and_then(|v| isize::try_from(v).ok())
            .unwrap_or(0);

        Ok(Box::new(Self {
            _name: layer.name.clone(),
            axis,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
