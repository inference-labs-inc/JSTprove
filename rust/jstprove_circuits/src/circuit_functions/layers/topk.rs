// ONNX `TopK` layer for ZK circuits.
//
// # ZK approach
// TopK selects the K largest (or smallest) values along a specified axis.
// Because comparison and sorting cannot be expressed as low-degree polynomials
// over a finite field, a hint is used:
//
// 1. **Hint**: for each lane of n values along the axis, call
//    `api.new_hint("jstprove.topk_hint", &[x_0, ..., x_{n-1}, scale], K)`
//    which sorts and returns the K largest values in descending order.
//    No circuit constraint is added by this step.
//
// 2. **Output wiring**: hint outputs are forwarded directly as TopK values.
//    Values are kept in signed field encoding (including `p - |x|` for
//    negatives), matching the representation used by the rest of the circuit.
//
// # Outputs
// TopK has two ONNX outputs: Values (output[0]) and Indices (output[1]).
// This implementation registers only the **Values** output in the circuit's
// tensor map. The Indices output is intentionally omitted; if a downstream
// layer requires it, a clear "missing tensor" error is produced at build time.
// Full indices support would require a permutation argument or a sorting
// network and is planned as a future extension.
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

use ethnum::U256;
use ndarray::{ArrayD, Axis, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, ShiftRangeContext},
    hints::topk::TOPK_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
    },
};

// -------- Struct --------

pub struct TopKLayer {
    inputs: Vec<String>,
    /// Only the Values output name is registered; Indices is stored but not
    /// inserted into the circuit tensor map.
    values_output_name: String,
    /// Resolved (non-negative) axis.
    axis: usize,
    /// Number of values to select per lane.
    k: usize,
    /// Output shape of the Values tensor.
    output_shape: Vec<usize>,
    /// n_bits metadata kept for compatibility with layer configuration.
    n_bits: usize,
    /// Scaling factor `2^scale_exponent`, passed to the hint.
    scaling: u64,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TopKLayer {
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::TopK, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::TopK,
            name: x_name.to_string(),
        })?;

        let in_shape = x_input.shape().to_vec();
        let rank = in_shape.len();
        if self.axis >= rank {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!("axis {} out of range for rank {rank}", self.axis),
            }
            .into());
        }

        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let n_bits = self.n_bits;
        let shift_exponent = n_bits.saturating_sub(1);
        let shift_ctx = ShiftRangeContext::new::<C, Builder>(api, shift_exponent).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("ShiftRangeContext::new failed for TopK: {e}"),
            }
        })?;
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        let zero_var = api.constant(CircuitField::<C>::from_u256(U256::from(0u64)));
        let mut out_array = ArrayD::from_elem(IxDyn(&self.output_shape), zero_var);

        // Iterate over lanes along the axis.  Both input and output have the
        // same rank; the output axis dimension is K while the input's is n.
        let input_lanes: Vec<_> = x_input.lanes(Axis(self.axis)).into_iter().collect();
        for (in_lane, mut out_lane) in input_lanes
            .into_iter()
            .zip(out_array.lanes_mut(Axis(self.axis)).into_iter())
        {
            // Hint layout: [x_0, ..., x_{n-1}, scale] → K outputs.
            let mut hint_inputs: Vec<Variable> = in_lane.iter().copied().collect();
            hint_inputs.push(scale_var);

            let hint_out = api.new_hint(TOPK_HINT_KEY, &hint_inputs, self.k);

            for (out_elem, &y) in out_lane.iter_mut().zip(hint_out.iter()) {
                let y_shifted = api.add(y, shift_ctx.offset);
                logup_ctx
                    .range_check::<C, Builder>(api, y_shifted, n_bits)
                    .map_err(|e| LayerError::Other {
                        layer: LayerKind::TopK,
                        msg: format!("signed TopK range check failed: {e}"),
                    })?;
                *out_elem = y;
            }
        }

        logup_ctx.finalize::<C, Builder>(api);

        // Only register the Values output. Indices output is not available in
        // the Expander backend.
        Ok((vec![self.values_output_name.clone()], out_array))
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
