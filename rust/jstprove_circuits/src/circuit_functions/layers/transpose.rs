// ONNX `Transpose` layer for ZK circuits.
//
// # ZK approach
// Transpose reorders tensor elements according to a permutation of axes.
// Because the circuit works in fixed-point quantised integers and field
// elements carry no type or layout information, Transpose is a purely
// structural operation: the output element at each flat position is the
// input element at a precomputed flat position.
//
// A compile-time index map `index_map[out_flat] = in_flat` is built during
// `build()` from the `perm` attribute (default: reversed axes).  At
// circuit evaluation time `apply()` simply selects from the input by index —
// no hint, no range check.

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
};

// -------- Struct --------

pub struct TransposeLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Output shape after permutation.
    output_shape: Vec<usize>,
    /// `index_map[out_flat] = in_flat` — precomputed at build time.
    index_map: Vec<usize>,
}

// -------- Index helpers --------

/// Convert a flat index to per-dimension coordinates (C-order / row-major).
fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            coords[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
    coords
}

/// Convert per-dimension coordinates to a flat index (C-order / row-major).
fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += coords[i] * stride;
        stride *= shape[i];
    }
    flat
}

/// Build the compile-time index map for a Transpose with the given `perm`.
///
/// `index_map[out_flat] = in_flat` maps every output position back to the
/// corresponding input position.
fn build_index_map(input_shape: &[usize], perm: &[usize]) -> Vec<usize> {
    let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p]).collect();
    let total_out: usize = output_shape.iter().product();
    let mut index_map = Vec::with_capacity(total_out);

    // Inverse permutation: inv_perm[i] = j  means output axis i came from input axis j.
    // For the input coordinate, we need to invert the perm:
    //   out_coords[i] = in_coords[perm[i]]
    // => in_coords[perm[i]] = out_coords[i]
    let mut inv_perm = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    for out_flat in 0..total_out {
        let out_coords = unravel(out_flat, &output_shape);
        // in_coords[p] = out_coords[inv_perm[p]]  for each input axis p
        let in_coords: Vec<usize> = (0..input_shape.len())
            .map(|p| out_coords[inv_perm[p]])
            .collect();
        index_map.push(ravel(&in_coords, input_shape));
    }

    index_map
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TransposeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Transpose, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Transpose,
            name: x_name.clone(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::Transpose,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        // Structural passthrough: select elements by the precomputed index map.
        let out_flat: Vec<Variable> = self.index_map.iter().map(|&idx| data_flat[idx]).collect();

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::Transpose,
                    msg: format!("transpose output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
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
                layer: LayerKind::Transpose,
                param: "data input".to_string(),
            })?;

        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::Transpose,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Transpose,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let output_shape = layer_context
            .shapes_map
            .get(output_name)
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::Transpose,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let rank = input_shape.len();

        // Read the `perm` attribute. Default is the reversed axis order.
        let params = extract_params(layer).ok();
        let default_perm: Vec<i64> = (0..rank).rev().map(|i| i as i64).collect();
        let perm_raw: Vec<i64> = params
            .as_ref()
            .and_then(|p| get_param_or_default(&layer.name, "perm", p, Some(&default_perm)).ok())
            .unwrap_or(default_perm);

        let perm: Vec<usize> = perm_raw
            .iter()
            .map(|&a| {
                let n = if a < 0 { rank as i64 + a } else { a } as usize;
                if n >= rank {
                    return Err(LayerError::Other {
                        layer: LayerKind::Transpose,
                        msg: format!("perm value {a} out of range for rank {rank}"),
                    }
                    .into());
                }
                Ok(n)
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;

        if perm.len() != rank {
            return Err(LayerError::Other {
                layer: LayerKind::Transpose,
                msg: format!("perm length {} != input rank {rank}", perm.len()),
            }
            .into());
        }

        // Validate that perm is a true permutation (no duplicate indices).
        let unique: HashSet<usize> = perm.iter().copied().collect();
        if unique.len() != perm.len() {
            return Err(LayerError::Other {
                layer: LayerKind::Transpose,
                msg: format!(
                    "layer '{}': perm {:?} contains duplicate indices and is not a valid permutation",
                    layer.name, perm
                ),
            }
            .into());
        }

        // Validate that applying perm to input_shape produces the expected output_shape.
        let expected_output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p]).collect();
        if expected_output_shape != output_shape {
            return Err(LayerError::Other {
                layer: LayerKind::Transpose,
                msg: format!(
                    "layer '{}': applying perm {:?} to input shape {:?} gives {:?} \
                     but output shape is {:?}",
                    layer.name, perm, input_shape, expected_output_shape, output_shape
                ),
            }
            .into());
        }

        let index_map = build_index_map(&input_shape, &perm);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            index_map,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_map_2d_transpose() {
        // Shape [2, 3], perm [1, 0] → shape [3, 2]
        // Input flat:  [0,1,2,3,4,5] indexed as [[0,1,2],[3,4,5]]
        // Transposed:  [[0,3],[1,4],[2,5]] → flat [0,3,1,4,2,5]
        let input_shape = vec![2, 3];
        let perm = vec![1, 0];
        let map = build_index_map(&input_shape, &perm);
        assert_eq!(map, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn index_map_identity_perm() {
        // perm = [0, 1, 2] → identity
        let input_shape = vec![2, 3, 4];
        let perm: Vec<usize> = vec![0, 1, 2];
        let map = build_index_map(&input_shape, &perm);
        let expected: Vec<usize> = (0..24).collect();
        assert_eq!(map, expected);
    }

    #[test]
    fn index_map_3d_perm_102() {
        // Shape [2, 3, 4], perm [1, 0, 2] → shape [3, 2, 4]
        // out[i,j,k] = in[j,i,k]
        let input_shape = vec![2, 3, 4];
        let perm = vec![1, 0, 2];
        let map = build_index_map(&input_shape, &perm);
        // Verify a few entries manually:
        // out[0,0,0] = in[0,0,0] = 0
        assert_eq!(map[0], 0);
        // out[0,1,0] = in[1,0,0] = 12  (stride for axis0 in [2,3,4] is 12)
        assert_eq!(map[4], 12);
        // out[1,0,2] = in[0,1,2] = 6
        assert_eq!(map[10], 6);
        assert_eq!(map.len(), 2 * 3 * 4);
    }

    #[test]
    fn index_map_reverse_default() {
        // Default perm for rank 3 is [2, 1, 0]
        let input_shape = vec![2, 3, 4];
        let perm = vec![2, 1, 0];
        let map = build_index_map(&input_shape, &perm);
        // out[0,0,0] = in[0,0,0] = 0
        assert_eq!(map[0], 0);
        // out[0,0,1] = in[1,0,0] = 12
        assert_eq!(map[1], 12);
        assert_eq!(map.len(), 24);
    }
}
