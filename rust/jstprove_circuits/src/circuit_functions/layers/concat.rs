// ONNX `Concat` layer for ZK circuits.
//
// # ZK approach
// Concat joins multiple input tensors along a given axis.
// In the fixed-point quantised integer domain, this is purely structural:
// the output is formed by selecting elements from the inputs in order,
// analogous to Tile / Reshape — no arithmetic, no hint, no range check.
//
// At circuit evaluation time `apply()` uses ndarray::concatenate to assemble
// the output from the input arrays along the precomputed `axis`.

use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, Axis};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_param_or_default,
};

// -------- Struct --------

pub struct ConcatLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    axis: usize,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConcatLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Collect all input arrays (duplicates allowed — same name used twice).
        let input_arrays: Vec<ArrayD<Variable>> = self
            .inputs
            .iter()
            .map(|name| {
                input
                    .get(name)
                    .cloned()
                    .ok_or_else(|| LayerError::MissingInput {
                        layer: LayerKind::Concat,
                        name: name.clone(),
                    })
            })
            .collect::<Result<Vec<_>, LayerError>>()?;

        let views: Vec<_> = input_arrays.iter().map(|a| a.view()).collect();
        let result = ndarray::concatenate(Axis(self.axis), &views).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::Concat,
                msg: format!("concatenate failed at axis {}: {e}", self.axis),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        if layer.inputs.is_empty() {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Concat,
                param: "data inputs".to_string(),
            }
            .into());
        }

        // Determine rank from the first input's shape.
        let first_input_name = layer.inputs.first().unwrap();
        let rank = layer_context
            .shapes_map
            .get(first_input_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Concat,
                msg: format!("missing input shape for '{first_input_name}'"),
            })?
            .len();

        // Read the `axis` attribute; default to 0 if absent.
        let params = layer.params.clone();
        let axis_raw: i64 = params
            .as_ref()
            .and_then(|p| get_param_or_default::<i64>(&layer.name, "axis", p, Some(&0i64)).ok())
            .unwrap_or(0i64);

        let axis = if axis_raw < 0 {
            let a = rank as i64 + axis_raw;
            if a < 0 {
                return Err(LayerError::Other {
                    layer: LayerKind::Concat,
                    msg: format!("axis {axis_raw} out of range for rank {rank}"),
                }
                .into());
            }
            a as usize
        } else {
            axis_raw as usize
        };

        if axis >= rank {
            return Err(LayerError::Other {
                layer: LayerKind::Concat,
                msg: format!("axis {axis} >= rank {rank}"),
            }
            .into());
        }

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            axis,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn axis_normalization_positive() {
        // axis=1 for rank=3 stays 1
        let rank = 3usize;
        let axis_raw: i64 = 1;
        let axis = if axis_raw < 0 {
            (rank as i64 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        assert_eq!(axis, 1);
    }

    #[test]
    fn axis_normalization_negative() {
        // axis=-1 for rank=3 → axis=2
        let rank = 3usize;
        let axis_raw: i64 = -1;
        let axis = if axis_raw < 0 {
            (rank as i64 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        assert_eq!(axis, 2);
    }

    #[test]
    fn axis_normalization_negative_first() {
        // axis=-3 for rank=3 → axis=0
        let rank = 3usize;
        let axis_raw: i64 = -3;
        let axis = if axis_raw < 0 {
            (rank as i64 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        assert_eq!(axis, 0);
    }
}
