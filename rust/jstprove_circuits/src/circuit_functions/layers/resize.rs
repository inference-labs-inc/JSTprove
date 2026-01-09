use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param_or_default},
    },
};

#[derive(Debug)]
pub struct ResizeLayer {
    name: String,
    mode: String,
    scales: Vec<f64>,
    sizes: Vec<i64>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ResizeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Resize, INPUT)?;
        let layer_input = input
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Resize,
                name: input_name.clone(),
            })?
            .clone();

        let in_shape = layer_input.shape();
        let out_shape = if !self.output_shape.is_empty() {
            self.output_shape.clone()
        } else if !self.sizes.is_empty() {
            self.sizes.iter().map(|&s| s as usize).collect()
        } else if !self.scales.is_empty() {
            in_shape
                .iter()
                .zip(self.scales.iter())
                .map(|(&d, &s)| (d as f64 * s) as usize)
                .collect()
        } else {
            in_shape.to_vec()
        };

        let out = resize_nearest(&layer_input, &out_shape)?;

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
                layer: LayerKind::Resize,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let mode: String = get_param_or_default(&layer.name, "mode", &params, Some(&"nearest".to_string()))?;
        let scales: Vec<f64> = get_param_or_default(&layer.name, "scales", &params, Some(&vec![]))?;
        let sizes: Vec<i64> = get_param_or_default(&layer.name, "sizes", &params, Some(&vec![]))?;

        let output_shape = layer_context
            .shapes_map
            .get(&layer.outputs[0])
            .cloned()
            .unwrap_or_default();

        Ok(Box::new(Self {
            name: layer.name.clone(),
            mode,
            scales,
            sizes,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
        }))
    }
}

fn resize_nearest<T: Clone>(input: &ArrayD<T>, out_shape: &[usize]) -> Result<ArrayD<T>, CircuitError> {
    let in_shape = input.shape();
    if in_shape.len() != out_shape.len() {
        return Err(CircuitError::from(LayerError::Other {
            layer: LayerKind::Resize,
            msg: format!("Shape mismatch: input {:?} vs output {:?}", in_shape, out_shape),
        }));
    }

    let total_out = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(total_out);

    let ndim = in_shape.len();
    let mut out_idx = vec![0usize; ndim];

    for _ in 0..total_out {
        let in_idx: Vec<usize> = (0..ndim)
            .map(|d| {
                let scale = in_shape[d] as f64 / out_shape[d] as f64;
                ((out_idx[d] as f64 + 0.5) * scale) as usize
            })
            .collect();

        let val = input.get(IxDyn(&in_idx)).ok_or_else(|| LayerError::Other {
            layer: LayerKind::Resize,
            msg: format!("Index out of bounds: {:?}", in_idx),
        })?;
        out_data.push(val.clone());

        for d in (0..ndim).rev() {
            out_idx[d] += 1;
            if out_idx[d] < out_shape[d] {
                break;
            }
            out_idx[d] = 0;
        }
    }

    ArrayD::from_shape_vec(IxDyn(out_shape), out_data).map_err(|e| {
        CircuitError::from(LayerError::Other {
            layer: LayerKind::Resize,
            msg: format!("Failed to create output array: {e}"),
        })
    })
}
