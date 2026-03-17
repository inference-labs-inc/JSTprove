// ONNX `Expand` layer for ZK circuits.
//
// # ZK approach
// Expand broadcasts the input tensor to a larger shape by repeating singleton
// dimensions. Like Tile, this is a purely structural operation: the output
// elements are a subset of the input elements (each input element may appear
// multiple times in the output). No hint or range check is needed.
//
// The broadcast follows standard numpy rules: axes are aligned from the right,
// and any input dimension of size 1 is repeated to match the target size.
//
// # Constraints
// - The target shape (`output_shape`) is resolved at build time from the
//   `shapes_map`. The shape input (input[1]) MUST be a compile-time constant.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

// -------- Struct --------

#[derive(Debug)]
pub struct ExpandLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// The target (broadcast) output shape, resolved at build time.
    output_shape: Vec<usize>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ExpandLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Expand, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Expand,
            name: x_name.to_string(),
        })?;

        // Use ndarray broadcast to expand singleton dimensions.
        let target = IxDyn(&self.output_shape);
        let result = x_input
            .broadcast(target)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Expand,
                msg: format!(
                    "cannot broadcast shape {:?} to {:?}",
                    x_input.shape(),
                    self.output_shape
                ),
            })?
            .into_owned();

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
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Expand,
                param: "input tensor".to_string(),
            })?;

        // Expand requires a second input (the target shape tensor).
        if layer.inputs.len() < 2 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Expand,
                param: "shape tensor (inputs[1])".to_string(),
            }
            .into());
        }
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Expand,
                param: "output tensor".to_string(),
            })?;

        // Verify input shape exists.
        layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Expand,
                msg: format!("missing input shape for '{input_name}'"),
            })?;

        // Verify the shape tensor (inputs[1]) is known at compile time.
        let shape_tensor_name = &layer.inputs[1];
        if !layer_context
            .shapes_map
            .contains_key(shape_tensor_name.as_str())
        {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Expand,
                msg: format!(
                    "shape tensor '{shape_tensor_name}' not found in shapes_map; \
                     Expand requires a compile-time constant shape input"
                ),
            }
            .into());
        }

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Expand,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn expand_broadcast_1x1_to_2x3() {
        use ndarray::{IxDyn, array};
        let x = array![[42i64]].into_dyn();
        let target = IxDyn(&[2, 3]);
        let result = x.broadcast(target).unwrap().into_owned();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&v| v == 42));
    }

    #[test]
    fn expand_broadcast_row_to_matrix() {
        use ndarray::{Array, IxDyn};
        let x = Array::from_vec(vec![1i64, 2, 3])
            .into_shape_with_order(IxDyn(&[1, 3]))
            .unwrap();
        let target = IxDyn(&[4, 3]);
        let result = x.broadcast(target).unwrap().into_owned();
        assert_eq!(result.shape(), &[4, 3]);
        for row in result.rows() {
            let row_vec: Vec<i64> = row.iter().copied().collect();
            assert_eq!(row_vec, vec![1, 2, 3]);
        }
    }
}
