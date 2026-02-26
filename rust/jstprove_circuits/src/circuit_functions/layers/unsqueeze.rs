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
    },
};

#[derive(Debug)]
pub struct UnsqueezeLayer {
    name: String,
    axes: Vec<i64>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl UnsqueezeLayer {
    fn unsqueezed_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, CircuitError> {
        let rank_in = input_shape.len();
        let rank_out = rank_in + self.axes.len();

        let axes_u = normalize_axes(&self.axes, rank_out, &LayerKind::Unsqueeze, &self.name)?;
        let axes_set: HashSet<usize> = axes_u.iter().copied().collect();

        let mut out_shape: Vec<usize> = Vec::with_capacity(rank_out);
        let mut j: usize = 0;

        for i in 0..rank_out {
            if axes_set.contains(&i) {
                out_shape.push(1);
            } else {
                let dim = *input_shape.get(j).ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Unsqueeze,
                    msg: format!(
                        "ran out of input dims when building output shape (input_shape={input_shape:?}, axes={:?})",
                        self.axes
                    ),
                })?;
                out_shape.push(dim);
                j += 1;
            }
        }

        if j != rank_in {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Unsqueeze,
                msg: format!(
                    "did not consume all input dims (consumed={j}, rank_in={rank_in}); input_shape={input_shape:?}, axes={:?}",
                    self.axes
                ),
            }
            .into());
        }

        Ok(out_shape)
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for UnsqueezeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Unsqueeze, INPUT)?;
        let layer_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Unsqueeze,
                name: input_name.clone(),
            })?
            .clone();

        let in_shape: Vec<usize> = layer_input.shape().to_vec();
        let out_shape = self.unsqueezed_shape(&in_shape)?;

        // Reshape without changing element order.
        let flat: Vec<Variable> = layer_input.iter().copied().collect();

        let out = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::Unsqueeze,
                msg: format!("failed to reshape in Unsqueeze: {e} (out_shape={out_shape:?})"),
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
            layer: LayerKind::Unsqueeze,
            msg: format!("extract_params failed: {e}"),
        })?;

        // Unsqueeze requires axes.
        let axes: Vec<i64> = get_param(&layer.name, AXES, &params)?;

        let unsqueeze = Self {
            name: layer.name.clone(),
            axes,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };

        Ok(Box::new(unsqueeze))
    }
}
