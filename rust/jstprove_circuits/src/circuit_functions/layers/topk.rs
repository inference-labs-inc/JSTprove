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
// 2. **Range check**: each of the K output values is passed through a LogUp
//    range check that bounds it to `[0, 2^n_bits)`, ensuring the result stays
//    within the quantised range. Values below zero are not produced because
//    TopK selects from the input whose values are all non-negative after
//    quantisation (they come from activation functions with non-negative
//    outputs).
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
    gadgets::range_check::LogupRangeCheckContext,
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
    /// n_bits for the LogUp range check on each output value.
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
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);
        let n_bits = self.n_bits;

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
                logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;
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

        // Read K from input[1] (must be a compile-time constant initializer).
        let k_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::TopK,
                param: "K input".to_string(),
            })?;
        let k_arr: ndarray::ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, k_name).map_err(|e| LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("failed to read K tensor '{k_name}': {e}"),
            })?;
        let k = k_arr
            .as_slice()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("K tensor '{k_name}' is not contiguous"),
            })?
            .first()
            .copied()
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("K tensor '{k_name}' is empty"),
            })?;

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

        // Output shape: same as input but with axis dimension = K.
        let values_output_name_for_shape = values_output_name.clone();
        let output_shape = layer_context
            .shapes_map
            .get(values_output_name_for_shape.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!("missing output shape for '{values_output_name}'"),
            })?
            .clone();

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
