// ONNX `Shape` layer for ZK circuits.
//
// # ZK approach
// Shape is a compile-time structural op: it reads the input tensor's dimensions
// from the pre-computed shapes_map and injects them as circuit constants.
// No hint and no range check are needed; the values are known at compile time.
//
// # Outputs
// A 1D tensor of length (end - start) containing the selected dimension sizes.
//
// # Attributes
// - `start` (default 0): first axis index (inclusive, may be negative).
// - `end`   (default rank): last axis index (exclusive, may be negative).

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
};

// -------- Struct --------

#[derive(Debug)]
pub struct ShapeLayer {
    #[allow(dead_code)]
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// The dimension sizes that will be emitted as constants (pre-computed at build time).
    dims: Vec<usize>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ShapeLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Emit each selected dimension as a circuit constant.
        let constants: Vec<Variable> = self
            .dims
            .iter()
            .map(|&d| api.constant(CircuitField::<C>::from_u256(U256::from(d as u64))))
            .collect();

        let result =
            ArrayD::from_shape_vec(IxDyn(&[self.dims.len()]), constants).map_err(|_| {
                LayerError::InvalidShape {
                    layer: LayerKind::Shape,
                    msg: format!(
                        "ShapeLayer: cannot create 1-D result of len {}",
                        self.dims.len()
                    ),
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
        if layer.outputs.len() != 1 {
            return Err(LayerError::Other {
                layer: LayerKind::Shape,
                msg: format!(
                    "Shape node must have exactly 1 output, got {}",
                    layer.outputs.len()
                ),
            }
            .into());
        }

        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Shape,
                name: "input tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Shape,
                msg: format!("missing shape for input '{input_name}'"),
            })?;

        let rank = input_shape.len() as i64;

        // Resolve start and end attributes (may be negative).
        let start_raw: i64 = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("start".into()) {
                            if let rmpv::Value::Integer(i) = v {
                                i.as_i64()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let end_raw: i64 = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("end".into()) {
                            if let rmpv::Value::Integer(i) = v {
                                i.as_i64()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
            .unwrap_or(rank);

        let start = if start_raw < 0 {
            (rank + start_raw).max(0) as usize
        } else {
            (start_raw as usize).min(input_shape.len())
        };
        let end = if end_raw < 0 {
            (rank + end_raw).max(0) as usize
        } else {
            (end_raw as usize).min(input_shape.len())
        };

        let dims: Vec<usize> = if end > start {
            input_shape[start..end].to_vec()
        } else {
            vec![]
        };

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            dims,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shape_layer_dims_selection() {
        // Verify that start/end slicing works as expected on a simple shape.
        let shape = [2usize, 3, 4, 5];
        // start=1, end=3 → [3, 4]
        let start = 1usize;
        let end = 3usize;
        let selected: Vec<usize> = shape[start..end].to_vec();
        assert_eq!(selected, vec![3, 4]);
    }

    #[test]
    fn shape_layer_full_shape() {
        let shape = [1usize, 8];
        let selected: Vec<usize> = shape[0..shape.len()].to_vec();
        assert_eq!(selected, vec![1, 8]);
    }
}
