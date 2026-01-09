use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param_or_default},
    },
};

#[derive(Debug)]
pub struct SliceLayer {
    name: String,
    starts: Vec<isize>,
    ends: Vec<isize>,
    axes: Vec<usize>,
    steps: Vec<isize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SliceLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Slice, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Slice,
                name: input_name.clone(),
            })?
            .clone();

        let ndim = layer_input.ndim();
        let shape = layer_input.shape().to_vec();

        let out = layer_input.slice_each_axis(|ax| {
            let axis_idx = ax.axis.0;
            let axis_len = shape[axis_idx] as isize;

            if let Some(pos) = self.axes.iter().position(|&a| a == axis_idx) {
                let mut start = self.starts.get(pos).copied().unwrap_or(0);
                let mut end = self.ends.get(pos).copied().unwrap_or(axis_len);
                let step = self.steps.get(pos).copied().unwrap_or(1);

                if start < 0 {
                    start += axis_len;
                }
                if end < 0 {
                    end += axis_len;
                }
                if end > axis_len {
                    end = axis_len;
                }

                ndarray::Slice::new(start, Some(end), step)
            } else {
                ndarray::Slice::from(..)
            }
        });

        Ok((self.outputs.clone(), out.to_owned()))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Slice,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;

        let starts: Vec<isize> =
            get_param_or_default(&layer.name, "starts", &params, Some(&vec![]))?;
        let ends: Vec<isize> = get_param_or_default(&layer.name, "ends", &params, Some(&vec![]))?;
        let axes: Vec<usize> = get_param_or_default(&layer.name, "axes", &params, Some(&vec![]))?;
        let steps: Vec<isize> =
            get_param_or_default(&layer.name, "steps", &params, Some(&vec![]))?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            starts,
            ends,
            axes,
            steps,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
