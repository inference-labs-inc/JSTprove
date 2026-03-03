use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{ArrayD, Axis};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::constants::INPUT,
    utils::onnx_model::get_input_name,
};

#[derive(Debug)]
pub struct TileLayer {
    /// Per-axis repeat counts derived from output_shape / input_shape at build time.
    repeats: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TileLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Tile, INPUT)?;
        let layer_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Tile,
                name: input_name.clone(),
            })?
            .clone();

        // Tile along each axis by cloning the current result and concatenating.
        let mut result = layer_input;
        for (ax, &rep) in self.repeats.iter().enumerate() {
            if rep == 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!("repeat for axis {ax} is zero"),
                }
                .into());
            }
            if rep == 1 {
                continue;
            }
            // Clone so that `views` borrows `template`, not `result`, avoiding
            // a borrow-conflict when reassigning `result`.
            let template = result.clone();
            let views: Vec<_> = (0..rep).map(|_| template.view()).collect();
            result =
                ndarray::concatenate(Axis(ax), &views).map_err(|e| LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!("concatenate failed at axis {ax}: {e}"),
                })?;
        }

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
                layer: LayerKind::Tile,
                param: "input tensor".to_string(),
            })?;
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Tile,
                param: "output tensor".to_string(),
            })?;

        let input_shape =
            layer_context
                .shapes_map
                .get(input_name)
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!("missing input shape for '{input_name}'"),
                })?;
        let output_shape =
            layer_context
                .shapes_map
                .get(output_name)
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!("missing output shape for '{output_name}'"),
                })?;

        if input_shape.len() != output_shape.len() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Tile,
                msg: format!(
                    "input rank {} != output rank {}",
                    input_shape.len(),
                    output_shape.len()
                ),
            }
            .into());
        }

        let mut repeats = Vec::with_capacity(input_shape.len());
        for (ax, (&out, &inp)) in output_shape.iter().zip(input_shape.iter()).enumerate() {
            if out == 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!("output dim at axis {ax} is zero"),
                }
                .into());
            }
            if inp == 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!("input dim at axis {ax} is zero"),
                }
                .into());
            }
            if out % inp != 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Tile,
                    msg: format!(
                        "output dim {out} at axis {ax} is not divisible by input dim {inp}"
                    ),
                }
                .into());
            }
            repeats.push(out / inp);
        }

        Ok(Box::new(Self {
            repeats,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}

#[allow(dead_code)]
fn tile_shape(input_shape: &[usize], repeats: &[usize]) -> Vec<usize> {
    input_shape
        .iter()
        .zip(repeats.iter())
        .map(|(&d, &r)| d * r)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_shape_basic() {
        assert_eq!(tile_shape(&[2, 3], &[1, 4]), vec![2, 12]);
    }

    #[test]
    fn tile_shape_identity() {
        assert_eq!(tile_shape(&[4, 5], &[1, 1]), vec![4, 5]);
    }

    #[test]
    fn tile_shape_3d() {
        assert_eq!(tile_shape(&[1, 2, 3], &[2, 3, 4]), vec![2, 6, 12]);
    }
}
