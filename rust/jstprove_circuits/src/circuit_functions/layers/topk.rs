// ONNX `TopK` layer for ZK circuits.
//
// # ZK approach
// TopK selects the K largest (or smallest) values along a specified axis.
// Because comparison and sorting cannot be expressed as low-degree polynomials
// over a finite field, a sound implementation requires a full sorting/permutation
// circuit. A hint-only approach (hint + range check) only proves outputs are in
// range, NOT that they are the actual top-K values — this is unsound.
//
// # Current status
// TopK is **not supported** in the Expander backend. `apply` returns an error
// immediately. Use the Remainder backend or remove TopK from the model.
//
// # Outputs
// TopK has two ONNX outputs: Values (output[0]) and Indices (output[1]).
// The Indices output is intentionally omitted; if a downstream layer requires
// it, a clear "missing tensor" error is produced at build time.
//
// # Supported attributes
// - `axis`    (default −1): axis along which top-K is selected.
// - `largest` (default 1) : 1 = K largest values (min not yet supported).
// - `sorted`  (default 1) : output is always in descending order.
//
// # K input
// In ONNX opset ≥ 10 K is supplied as input[1], an int64 scalar tensor.
// It MUST be a compile-time constant (model initializer or Constant node).

use std::collections::HashMap;

use ndarray::ArrayD;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

// -------- Struct --------

// Fields are populated during `build` but never read by `apply` (which always
// returns an error). Kept so `build` can be reused when a sorting circuit is
// eventually implemented.
#[allow(dead_code)]
pub struct TopKLayer {
    inputs: Vec<String>,
    values_output_name: String,
    axis: usize,
    k: usize,
    output_shape: Vec<usize>,
    n_bits: usize,
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TopKLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // TopK is not soundly implementable via hint alone: a hint + range check
        // only proves that outputs are in range, NOT that they are the actual
        // top-K values of the input. Full soundness requires a sorting circuit
        // with permutation constraints, which is not yet implemented.
        Err(LayerError::Other {
            layer: LayerKind::TopK,
            msg: "TopK is not yet supported in any backend: soundly proving \
                  top-K selection requires a sorting/permutation circuit that is not \
                  yet implemented. Remove TopK from the model or await a backend that \
                  implements sorting/permutation circuits."
                .to_string(),
        }
        .into())
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // input[0] = data tensor.
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::TopK,
                name: "data input".to_string(),
            })?;

        // output[0] = Values (required).
        let values_output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::TopK,
                param: "Values output".to_string(),
            })?
            .clone();

        // output[1] = Indices — not registered in the Expander backend.
        // Models that only use the values output work fine; if a downstream
        // layer actually consumes the indices tensor it will fail with a
        // clear missing-tensor error at that point.

        // Read K from input[1]. It may come from initializers or from
        // layer.params constant-node injection.
        let k_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::TopK,
                param: "K input".to_string(),
            })?;

        let parse_i64_vec = |v: &rmpv::Value| -> Option<Vec<i64>> {
            match v {
                rmpv::Value::Array(xs) => xs
                    .iter()
                    .map(|x| {
                        if let rmpv::Value::Integer(i) = x {
                            i.as_i64()
                        } else {
                            None
                        }
                    })
                    .collect(),
                rmpv::Value::Integer(i) => i.as_i64().map(|x| vec![x]),
                _ => None,
            }
        };

        let k_arr: ndarray::ArrayD<i64> = match get_w_or_b(layer_context.w_and_b_map, k_name) {
            Ok(arr) => arr,
            Err(init_err) => {
                let k_from_params = layer.params.as_ref().and_then(|p| {
                    if let rmpv::Value::Map(entries) = p {
                        entries.iter().find_map(|(key, value)| {
                            if let rmpv::Value::String(s) = key {
                                if s.as_str() == Some(k_name.as_str()) {
                                    return parse_i64_vec(value).and_then(|vals| {
                                        ndarray::ArrayD::from_shape_vec(
                                            ndarray::IxDyn(&[vals.len()]),
                                            vals,
                                        )
                                        .ok()
                                    });
                                }
                            }
                            None
                        })
                    } else {
                        None
                    }
                });

                k_from_params.ok_or_else(|| LayerError::Other {
                    layer: LayerKind::TopK,
                    msg: format!(
                        "failed to read K tensor '{k_name}' from initializers or layer params: {init_err}"
                    ),
                })?
            }
        };
        let k = {
            let slice = k_arr.as_slice().ok_or_else(|| LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("K tensor '{k_name}' is not contiguous"),
            })?;
            if slice.len() != 1 {
                return Err(LayerError::Other {
                    layer: LayerKind::TopK,
                    msg: format!(
                        "K tensor '{k_name}' must be a scalar; found {} elements",
                        slice.len()
                    ),
                }
                .into());
            }
            slice[0]
        };

        if k <= 0 {
            return Err(LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("K tensor '{k_name}' must be positive, got {k}"),
            }
            .into());
        }
        let k = k as usize;

        // Resolve axis (default −1).
        let input_name = layer.inputs.first().unwrap();
        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!("missing input shape for '{input_name}'"),
            })?;
        let rank = input_shape.len();

        let axis_raw: i64 = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("axis".into()) {
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
            .unwrap_or(-1);

        let axis = if axis_raw < 0 {
            let a = rank as i64 + axis_raw;
            if a < 0 {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::TopK,
                    msg: format!("axis {axis_raw} out of range for rank {rank}"),
                }
                .into());
            }
            a as usize
        } else {
            let a = axis_raw as usize;
            if a >= rank {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::TopK,
                    msg: format!("axis {axis_raw} out of range for rank {rank}"),
                }
                .into());
            }
            a
        };

        if k > input_shape[axis] {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!(
                    "K={k} exceeds axis dimension {} on axis {axis}",
                    input_shape[axis]
                ),
            }
            .into());
        }

        let largest_raw: i64 = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("largest".into()) {
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
            .unwrap_or(1);
        if largest_raw != 1 {
            return Err(LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("only largest=1 is supported, got largest={largest_raw}"),
            }
            .into());
        }

        // Compute expected output shape: same as input but with axis dimension = K.
        let expected_output_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if i == axis { k } else { d })
            .collect();
        let output_shape = layer_context
            .shapes_map
            .get(values_output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!("missing output shape for '{values_output_name}'"),
            })?
            .clone();
        if output_shape != expected_output_shape {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!(
                    "output shape {:?} from shapes_map does not match expected {:?} \
                     (input {:?}, axis {}, k {})",
                    output_shape, expected_output_shape, input_shape, axis, k
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            values_output_name,
            axis,
            k,
            output_shape,
            n_bits,
            scaling,
        }))
    }
}
