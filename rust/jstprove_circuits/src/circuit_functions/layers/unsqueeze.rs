use std::collections::{HashMap, HashSet};

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{AXES, INPUT},
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param},
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
    fn normalize_axes(&self, rank_out: usize) -> Result<Vec<usize>, CircuitError> {
        let mut out: Vec<usize> = Vec::with_capacity(self.axes.len());
        let mut seen: HashSet<usize> = HashSet::new();

        for &a in &self.axes {
            let rank_i64 =
                i64::try_from(rank_out).map_err(|_| LayerError::InvalidParameterValue {
                    layer: LayerKind::Unsqueeze,
                    layer_name: self.name.clone(),
                    param_name: AXES.into(),
                    value: format!("rank {rank_out} cannot be represented as i64"),
                })?;

            let ax_i64 = if a < 0 { a + rank_i64 } else { a };

            if ax_i64 < 0 || ax_i64 >= rank_i64 {
                return Err(LayerError::InvalidParameterValue {
                    layer: LayerKind::Unsqueeze,
                    layer_name: self.name.clone(),
                    param_name: AXES.into(),
                    value: format!("axis {a} out of range for rank {rank_out}"),
                }
                .into());
            }

            let ax = usize::try_from(ax_i64).map_err(|_| LayerError::InvalidParameterValue {
                layer: LayerKind::Unsqueeze,
                layer_name: self.name.clone(),
                param_name: AXES.into(),
                value: format!("axis {a} is not a valid usize index after normalization"),
            })?;

            if !seen.insert(ax) {
                return Err(LayerError::InvalidParameterValue {
                    layer: LayerKind::Unsqueeze,
                    layer_name: self.name.clone(),
                    param_name: AXES.into(),
                    value: format!("duplicate axis {ax} in axes={:?}", self.axes),
                }
                .into());
            }

            out.push(ax);
        }

        out.sort_unstable();
        Ok(out)
    }

    fn unsqueezed_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, CircuitError> {
        let rank_in = input_shape.len();
        let rank_out = rank_in + self.axes.len();

        let axes_u = self.normalize_axes(rank_out)?;
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
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Unsqueeze,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
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
