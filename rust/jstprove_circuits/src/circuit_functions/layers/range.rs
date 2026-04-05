use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

#[derive(Debug)]
pub struct RangeLayer {
    #[allow(dead_code)]
    inputs: Vec<String>,
    outputs: Vec<String>,
    sequence: Vec<i64>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for RangeLayer {
    #[allow(clippy::cast_sign_loss)]
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let constants: Vec<Variable> = self
            .sequence
            .iter()
            .map(|&v| {
                if v >= 0 {
                    api.constant(CircuitField::<C>::from_u256(U256::from(v as u64)))
                } else {
                    let mag = U256::from(v.unsigned_abs());
                    api.constant(CircuitField::<C>::from_u256(
                        CircuitField::<C>::MODULUS - mag,
                    ))
                }
            })
            .collect();

        let result =
            ArrayD::from_shape_vec(IxDyn(&[self.sequence.len()]), constants).map_err(|_| {
                LayerError::InvalidShape {
                    layer: LayerKind::Range,
                    msg: format!(
                        "RangeLayer: cannot create 1-D result of len {}",
                        self.sequence.len()
                    ),
                }
            })?;

        Ok((self.outputs.clone(), result))
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
        if layer.inputs.len() != 3 {
            return Err(LayerError::MissingParameter {
                layer: LayerKind::Range,
                param: format!(
                    "Range expects exactly 3 inputs (start, limit, delta), got {}",
                    layer.inputs.len()
                ),
            }
            .into());
        }

        let load_scalar = |name: &str| -> Result<i64, CircuitError> {
            let owned = name.to_string();
            let arr: ndarray::ArrayD<i64> =
                get_w_or_b(layer_context.w_and_b_map, &owned).map_err(|e| LayerError::Other {
                    layer: LayerKind::Range,
                    msg: format!("Range input '{name}' must be a compile-time constant: {e}"),
                })?;
            if arr.len() != 1 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Range,
                    msg: format!(
                        "Range input '{name}' must be scalar, got shape {:?}",
                        arr.shape()
                    ),
                }
                .into());
            }
            Ok(arr.as_slice().unwrap()[0])
        };

        let start = load_scalar(&layer.inputs[0])?;
        let limit = load_scalar(&layer.inputs[1])?;
        let delta = load_scalar(&layer.inputs[2])?;

        if delta == 0 {
            return Err(LayerError::Other {
                layer: LayerKind::Range,
                msg: "Range delta must not be zero".to_string(),
            }
            .into());
        }

        let mut sequence = Vec::new();
        let mut current = start;
        if delta > 0 {
            while current < limit {
                sequence.push(current);
                current = current.saturating_add(delta);
            }
        } else {
            while current > limit {
                sequence.push(current);
                current = current.saturating_add(delta);
            }
        }

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            sequence,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn range_ascending() {
        let start = 0i64;
        let limit = 5i64;
        let delta = 1i64;
        let mut seq = Vec::new();
        let mut c = start;
        while c < limit {
            seq.push(c);
            c += delta;
        }
        assert_eq!(seq, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn range_descending() {
        let start = 5i64;
        let limit = 0i64;
        let delta = -2i64;
        let mut seq = Vec::new();
        let mut c = start;
        while c > limit {
            seq.push(c);
            c += delta;
        }
        assert_eq!(seq, vec![5, 3, 1]);
    }

    #[test]
    fn range_empty_when_start_equals_limit() {
        let start = 3i64;
        let limit = 3i64;
        let delta = 1i64;
        let mut seq = Vec::new();
        let mut c = start;
        while c < limit {
            seq.push(c);
            c += delta;
        }
        assert!(seq.is_empty());
    }
}
