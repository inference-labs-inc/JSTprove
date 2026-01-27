use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{
        LayerError, LayerKind,
        layer_ops::{LayerOp, LayerResult},
    },
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
        typecasting::{AsIsize, AsUsize, IsizeAsUsize},
    },
};

#[derive(Debug)]
pub struct SplitLayer {
    _name: String,
    axis: isize,
    split: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for SplitLayer {
    fn apply(&self, _api: &mut Builder, input: HashMap<String, ArrayD<Variable>>) -> LayerResult {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Split, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Split,
                name: input_name.clone(),
            })?
            .clone();

        let ndim = layer_input.ndim();
        let axis: usize = if self.axis < 0 {
            // ndim is usize, convert to isize safely
            let ndim_isize = ndim.as_isize()?; // uses your AsIsize
            (ndim_isize + self.axis).as_usize()? // safe conversion
        } else {
            self.axis.as_usize()? // safe conversion
        };

        let axis_len = layer_input.shape()[axis];
        let num_outputs = self.outputs.len();

        let splits = if self.split.is_empty() {
            let base = axis_len.div_ceil(num_outputs);
            let mut remaining = axis_len;

            (0..num_outputs)
                .map(|_| {
                    let sz = remaining.min(base);
                    remaining -= sz;
                    sz
                })
                .collect()
        } else {
            self.split.clone()
        };

        let mut results: Vec<ArrayD<Variable>> = Vec::new();
        let mut start = 0;

        for &len in &splits {
            let end = start + len;

            // Convert to isize safely outside the closure
            let start_isize = start.as_isize()?;
            let end_isize = end.as_isize()?;

            let sliced = layer_input
                .slice_each_axis(|ax| {
                    if ax.axis.0 == axis {
                        ndarray::Slice::from(start_isize..end_isize)
                    } else {
                        ndarray::Slice::from(..)
                    }
                })
                .to_owned();

            results.push(sliced);
            start = end;
        }
        let outputs = results
            .into_iter()
            .zip(self.outputs.iter())
            .map(|(tensor, name)| {
                // each split output gets exactly one name
                (vec![name.clone()], tensor)
            })
            .collect();

        Ok(outputs)
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
                layer: LayerKind::Split,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let axis: isize = params
            .get("axis")
            .and_then(serde_json::Value::as_i64)
            .map(|v| v.as_isize())
            .transpose()?
            .unwrap_or(0);

        let split: Vec<usize> = params
            .get("split")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().and_then(|i| i.as_usize().ok()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Box::new(Self {
            _name: layer.name.clone(),
            axis,
            split,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}
