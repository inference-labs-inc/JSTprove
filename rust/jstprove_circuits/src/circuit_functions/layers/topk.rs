// ONNX `TopK` layer for ZK circuits.
//
// # ZK approach
// TopK selects the K largest values along a specified axis.  Because
// comparison and sorting cannot be expressed as low-degree polynomials over a
// finite field, the actual selection is performed outside the circuit by a
// hint function that returns K values and their original indices.
//
// The circuit then constrains:
// (a) **Membership** — each output value equals an input element at the
//     hint-provided index, verified via a multiplexer over the axis lane.
// (b) **Sorted order** — consecutive output values satisfy v[i] >= v[i+1],
//     verified by shifting to unsigned and applying a LogUp range check on
//     the difference.
// (c) **Top-K bound** — every non-selected input element is <= the K-th
//     (smallest selected) value, verified via conditional range checks.
// (d) **Index uniqueness** — the K hint-provided indices are pairwise
//     distinct.
//
// Together these prove that the K output values are a correct sorted
// selection of the K largest input values along the axis.
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
// In ONNX opset >= 10 K is supplied as input[1], an int64 scalar tensor.
// It MUST be a compile-time constant (model initializer or Constant node).

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::topk::TOPK_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

pub struct TopKLayer {
    inputs: Vec<String>,
    values_output_name: String,
    input_shape: Vec<usize>,
    axis: usize,
    k: usize,
    output_shape: Vec<usize>,
    n_bits: usize,
    scaling: u64,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TopKLayer {
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let data_name = &self.inputs[0];
        let data = input
            .get(data_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::TopK,
                name: data_name.clone(),
            })?;

        let data_flat = data.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::TopK,
            msg: "data tensor is not contiguous".to_string(),
        })?;

        let axis = self.axis;
        let k = self.k;
        let axis_dim = self.input_shape[axis];

        let inner: usize = self.input_shape[axis + 1..].iter().product();
        let outer: usize = self.input_shape[..axis].iter().product();

        let shift_n_bits = self.n_bits + 1;
        let offset_val = 1u64
            .checked_shl(self.n_bits as u32)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::TopK,
                msg: format!("n_bits {} too large for shift offset", self.n_bits),
            })?;
        let offset = api.constant(CircuitField::<C>::from_u256(ethnum::U256::from(offset_val)));
        let scale_var = api.constant(CircuitField::<C>::from_u256(ethnum::U256::from(
            self.scaling,
        )));

        let out_total: usize = self.output_shape.iter().product();
        let mut out_flat = vec![api.constant(0u32); out_total];

        for o in 0..outer {
            for inr in 0..inner {
                let lane: Vec<Variable> = (0..axis_dim)
                    .map(|a| data_flat[(o * axis_dim + a) * inner + inr])
                    .collect();

                let mut hint_inputs = Vec::with_capacity(axis_dim + 1);
                hint_inputs.extend_from_slice(&lane);
                hint_inputs.push(scale_var);

                let hint_out = api.new_hint(TOPK_HINT_KEY, &hint_inputs, 2 * k);

                let values = &hint_out[..k];
                let indices = &hint_out[k..];

                let mut mux_sums: Vec<Variable> = (0..k).map(|_| api.constant(0u32)).collect();
                let mut matched_counts: Vec<Variable> =
                    (0..k).map(|_| api.constant(0u32)).collect();
                let min_val = values[k - 1];

                for (a, &elem) in lane.iter().enumerate() {
                    let a_var = api.constant(a as u32);
                    let mut is_sel = api.constant(0u32);

                    for j in 0..k {
                        let diff = api.sub(indices[j], a_var);
                        let eq = api.is_zero(diff);
                        let selected = api.mul(eq, elem);
                        mux_sums[j] = api.add(mux_sums[j], selected);
                        matched_counts[j] = api.add(matched_counts[j], eq);
                        is_sel = api.add(is_sel, eq);
                    }

                    let delta = api.sub(min_val, elem);
                    let delta_shifted = api.add(delta, offset);
                    let one_c = api.constant(1u32);
                    let not_sel = api.sub(one_c, is_sel);
                    let check_val = api.mul(not_sel, delta_shifted);
                    let filler = api.mul(is_sel, offset);
                    let final_check = api.add(check_val, filler);

                    logup_ctx
                        .range_check::<C, Builder>(api, final_check, shift_n_bits)
                        .map_err(|e| LayerError::Other {
                            layer: LayerKind::TopK,
                            msg: format!("top-K bound range check failed: {e}"),
                        })?;
                }

                let one = api.constant(1u32);
                for j in 0..k {
                    api.assert_is_equal(matched_counts[j], one);
                    api.assert_is_equal(values[j], mux_sums[j]);

                    logup_ctx
                        .range_check::<C, Builder>(api, values[j], self.n_bits)
                        .map_err(|e| LayerError::Other {
                            layer: LayerKind::TopK,
                            msg: format!("range check on output value failed: {e}"),
                        })?;

                    let out_idx = (o * k + j) * inner + inr;
                    out_flat[out_idx] = values[j];
                }

                for j in 0..k.saturating_sub(1) {
                    let diff = api.sub(values[j], values[j + 1]);
                    let shifted = api.add(diff, offset);
                    logup_ctx
                        .range_check::<C, Builder>(api, shifted, shift_n_bits)
                        .map_err(|e| LayerError::Other {
                            layer: LayerKind::TopK,
                            msg: format!("sorted order range check failed: {e}"),
                        })?;
                }

                for i in 0..k {
                    for j in (i + 1)..k {
                        let diff = api.sub(indices[i], indices[j]);
                        let is_eq = api.is_zero(diff);
                        let zero = api.constant(0u32);
                        api.assert_is_equal(is_eq, zero);
                    }
                }
            }
        }

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::TopK,
                    msg: format!("output reshape failed: {e}"),
                }
            })?;

        Ok((vec![self.values_output_name.clone()], out_array))
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::TopK,
                name: "data input".to_string(),
            })?;

        let values_output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::TopK,
                param: "Values output".to_string(),
            })?
            .clone();

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

        let input_name = layer.inputs.first().unwrap();
        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::TopK,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();
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
                    "output shape {output_shape:?} from shapes_map does not match expected {expected_output_shape:?} \
                     (input {input_shape:?}, axis {axis}, k {k})"
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
            input_shape,
            axis,
            k,
            output_shape,
            n_bits,
            scaling,
        }))
    }
}
