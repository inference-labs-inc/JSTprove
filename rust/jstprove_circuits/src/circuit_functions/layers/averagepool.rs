// ONNX `AveragePool` layer for ZK circuits.
//
// # ZK approach
// AveragePool computes the spatial average over a sliding kernel window.
// Division is not expressible as a low-degree polynomial, so a hint is used:
//
// 1. **Hint**: for each output position, call
//    `api.new_hint("jstprove.averagepool_hint", &[x_0, ..., x_{n-1}, scale], 1)`
//    which returns `round(sum(x_i) / n_valid)` in the field.
//
// 2. **Sum constraint** (soundness): for each output position,
//    compute `sum` directly via `api.add`, then constrain:
//      n_valid * y + r  =  sum
//    where `r = sum − n_valid * y` is the rounding residual.
//
// 3. **Residual range check**: `r ∈ [−⌊n/2⌋, ⌊(n−1)/2⌋]`, proved by
//    computing `s = r + ⌊n/2⌋` and verifying:
//      - s ∈ [0, 2^k)   (s ≥ 0 → lower bound on r)
//      - (n−1) − s ∈ [0, 2^k)  (s ≤ n−1 → upper bound on r)
//    where `k = ⌈log₂(n_valid)⌉`.  Together these prove `s ∈ [0, n−1]` for
//    any kernel size, including non-powers-of-two.
//
// 4. **Output range check**: `y ∈ [0, 2^n_bits)` (non-negative activation).
//
// Pads are ONNX-style [begin_d0, begin_d1, ..., end_d0, end_d1, ...].
// Only positions inside the input boundary count toward `n_valid`
// (`count_include_pad = 0` semantic, which is the ONNX default).

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::averagepool::AVERAGEPOOL_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

// -------- Struct --------

#[derive(Debug)]
pub struct AveragePoolLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    kernel_shape: Vec<usize>,
    strides: Vec<usize>,
    pads: Vec<usize>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    n_bits: usize,
}

// -------- Helpers --------

/// Returns ⌈log₂(n)⌉, minimum 1.  Used to size the residual range check.
fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize
    }
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for AveragePoolLayer {
    #[allow(clippy::cast_possible_wrap)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::AveragePool, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::AveragePool,
            name: x_name.to_string(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::AveragePool,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        // Require 4-D input [N, C, H, W].
        if self.input_shape.len() != 4 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::AveragePool,
                msg: format!(
                    "AveragePool expects 4-D input [N,C,H,W], got rank {}",
                    self.input_shape.len()
                ),
            }
            .into());
        }
        let (big_n, c, h_in, w_in) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let (k_h, k_w) = (
            self.kernel_shape.first().copied().unwrap_or(1),
            self.kernel_shape.get(1).copied().unwrap_or(1),
        );
        let (s_h, s_w) = (
            self.strides.first().copied().unwrap_or(1),
            self.strides.get(1).copied().unwrap_or(1),
        );
        // ONNX pads: [begin_H, begin_W, end_H, end_W]
        let (ph_b, pw_b) = (
            self.pads.first().copied().unwrap_or(0),
            self.pads.get(1).copied().unwrap_or(0),
        );
        let (ph_e, pw_e) = (
            self.pads.get(2).copied().unwrap_or(0),
            self.pads.get(3).copied().unwrap_or(0),
        );

        let out_h = (h_in + ph_b + ph_e - k_h) / s_h + 1;
        let out_w = (w_in + pw_b + pw_e - k_w) / s_w + 1;

        let n_bits = self.n_bits;
        let mut out_storage: Vec<Variable> = Vec::with_capacity(big_n * c * out_h * out_w);

        for n_idx in 0..big_n {
            for c_idx in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = (oh * s_h) as isize - ph_b as isize;
                        let iw_start = (ow * s_w) as isize - pw_b as isize;

                        // Collect variables at valid (non-padded) positions.
                        let mut valid_vars: Vec<Variable> = Vec::with_capacity(k_h * k_w);
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = ih_start + kh as isize;
                                let iw = iw_start + kw as isize;
                                if ih >= 0
                                    && (ih as usize) < h_in
                                    && iw >= 0
                                    && (iw as usize) < w_in
                                {
                                    let flat = n_idx * (c * h_in * w_in)
                                        + c_idx * (h_in * w_in)
                                        + ih as usize * w_in
                                        + iw as usize;
                                    valid_vars.push(data_flat[flat]);
                                }
                                // count_include_pad=0: skip out-of-bounds positions
                            }
                        }

                        let n_valid = valid_vars.len();
                        if n_valid == 0 {
                            out_storage
                                .push(api.constant(CircuitField::<C>::from_u256(U256::from(0u64))));
                            continue;
                        }

                        // n_valid = 1: mean is the element itself, no rounding.
                        if n_valid == 1 {
                            let y = valid_vars[0];
                            logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;
                            out_storage.push(y);
                            continue;
                        }

                        // ── Input range checks (enforce non-negative activations) ─
                        // Sound because AveragePool is always after non-negative ops
                        // (e.g. ReLU). Enforcing this in the circuit makes the y >= 0
                        // range check sound; a malicious prover cannot supply negative
                        // inputs that average to a clamped zero.
                        for &var in &valid_vars {
                            logup_ctx.range_check::<C, Builder>(api, var, n_bits)?;
                        }

                        // ── Hint ────────────────────────────────────────────
                        let hint_out = api.new_hint(AVERAGEPOOL_HINT_KEY, &valid_vars, 1);
                        let y = hint_out[0];

                        // ── Sum constraint ───────────────────────────────────
                        // sum = sum(x_i) — computed in circuit
                        let sum_var = valid_vars[1..]
                            .iter()
                            .fold(valid_vars[0], |acc, &x| api.add(acc, x));

                        // n_valid * y
                        let n_const =
                            api.constant(CircuitField::<C>::from_u256(U256::from(n_valid as u64)));
                        let n_y = api.mul(n_const, y);

                        // r = sum − n*y
                        let r = api.sub(sum_var, n_y);

                        // ── Residual range check ─────────────────────────────
                        // s = r + ⌊n/2⌋  ∈ [0, n−1]
                        // proved by:
                        //   (a) s ∈ [0, 2^k)         → s ≥ 0
                        //   (b) (n−1) − s ∈ [0, 2^k) → s ≤ n−1
                        let half_n = (n_valid / 2) as u64;
                        let half_const =
                            api.constant(CircuitField::<C>::from_u256(U256::from(half_n)));
                        let s = api.add(r, half_const);

                        let k = ceil_log2(n_valid);

                        logup_ctx.range_check::<C, Builder>(api, s, k)?;

                        let nm1_const = api.constant(CircuitField::<C>::from_u256(U256::from(
                            (n_valid - 1) as u64,
                        )));
                        let t = api.sub(nm1_const, s);
                        logup_ctx.range_check::<C, Builder>(api, t, k)?;

                        // ── Output range check ───────────────────────────────
                        logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;

                        out_storage.push(y);
                    }
                }
            }
        }

        let result =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_storage).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::AveragePool,
                    msg: format!("AveragePool output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), result))
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
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
                layer: LayerKind::AveragePool,
                param: "input tensor".to_string(),
            })?;
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::AveragePool,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::AveragePool,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::AveragePool,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        // Parse kernel_shape from params.
        let mk_err = |msg: String| LayerError::InvalidShape {
            layer: LayerKind::AveragePool,
            msg,
        };
        let kernel_shape = parse_usize_list(layer, "kernel_shape")
            .map_err(mk_err)?
            .unwrap_or_else(|| vec![1]);
        let strides = parse_usize_list(layer, "strides")
            .map_err(mk_err)?
            .unwrap_or_else(|| vec![1; kernel_shape.len()]);
        let pads = parse_usize_list(layer, "pads")
            .map_err(mk_err)?
            .unwrap_or_else(|| vec![0; kernel_shape.len() * 2]);

        // Reject unsupported pooling attributes.
        let ceil_mode = get_int_param(layer, "ceil_mode").unwrap_or(0);
        if ceil_mode != 0 {
            return Err(LayerError::Other {
                layer: LayerKind::AveragePool,
                msg: format!(
                    "AveragePool ceil_mode={ceil_mode} is not supported; only ceil_mode=0 is implemented"
                ),
            }
            .into());
        }
        let count_include_pad = get_int_param(layer, "count_include_pad").unwrap_or(0);
        if count_include_pad != 0 {
            return Err(LayerError::Other {
                layer: LayerKind::AveragePool,
                msg: "AveragePool count_include_pad=1 is not supported; only count_include_pad=0 is implemented".to_string(),
            }
            .into());
        }
        let auto_pad = get_str_param(layer, "auto_pad");
        if auto_pad
            .as_deref()
            .is_some_and(|s| s != "NOTSET" && !s.is_empty())
        {
            return Err(LayerError::Other {
                layer: LayerKind::AveragePool,
                msg: format!(
                    "AveragePool auto_pad='{}' is not supported; only auto_pad=NOTSET is implemented",
                    auto_pad.unwrap()
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            kernel_shape,
            strides,
            pads,
            input_shape,
            output_shape,
            n_bits,
        }))
    }
}

/// Returns `Ok(None)` when the key is absent, `Ok(Some(vals))` when valid,
/// and `Err(message)` when the key is present but contains an invalid (negative) value.
pub(crate) fn parse_usize_list(
    layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
    key: &str,
) -> Result<Option<Vec<usize>>, String> {
    let map = match layer.params.as_ref() {
        Some(rmpv::Value::Map(m)) => m,
        _ => return Ok(None),
    };
    let entry = map
        .iter()
        .find(|(k, _)| k == &rmpv::Value::String(key.into()));
    let v = match entry {
        Some((_, v)) => v,
        None => return Ok(None),
    };
    let parse_one = |x: &rmpv::Value| -> Result<usize, String> {
        if let rmpv::Value::Integer(i) = x {
            match i.as_i64() {
                Some(n) if n >= 0 => Ok(n as usize),
                Some(n) => Err(format!("'{key}' contains negative value {n}")),
                None => Err(format!(
                    "'{key}' contains an integer that cannot be read as i64"
                )),
            }
        } else {
            Err(format!("'{key}' contains a non-integer value"))
        }
    };
    match v {
        rmpv::Value::Array(arr) => arr
            .iter()
            .map(parse_one)
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        rmpv::Value::Integer(_) => parse_one(v).map(|n| Some(vec![n])),
        _ => Err(format!("'{key}' has an unexpected msgpack type")),
    }
}

/// Returns the integer value of a scalar attribute, or `None` if absent/non-integer.
fn get_int_param(
    layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
    key: &str,
) -> Option<i64> {
    if let Some(rmpv::Value::Map(m)) = layer.params.as_ref() {
        m.iter().find_map(|(k, v)| {
            if k == &rmpv::Value::String(key.into()) {
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
}

/// Returns the string value of a scalar attribute, or `None` if absent/non-string.
fn get_str_param(
    layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
    key: &str,
) -> Option<String> {
    if let Some(rmpv::Value::Map(m)) = layer.params.as_ref() {
        m.iter().find_map(|(k, v)| {
            if k == &rmpv::Value::String(key.into()) {
                if let rmpv::Value::String(s) = v {
                    s.as_str().map(|x| x.to_string())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_functions::utils::onnx_types::ONNXLayer;
    use rmpv::Value;
    use std::collections::HashMap;

    fn make_layer(key: &str, vals: Vec<i64>) -> ONNXLayer {
        let arr = Value::Array(
            vals.into_iter()
                .map(|v| Value::Integer(rmpv::Integer::from(v)))
                .collect(),
        );
        let params = Value::Map(vec![(Value::String(key.into()), arr)]);
        ONNXLayer {
            id: 0,
            name: "test".to_string(),
            op_type: "AveragePool".to_string(),
            inputs: vec![],
            outputs: vec![],
            shape: HashMap::new(),
            tensor: None,
            params: Some(params),
            opset_version_number: 13,
        }
    }

    #[test]
    fn parse_usize_list_array() {
        let layer = make_layer("kernel_shape", vec![3, 3]);
        let result = parse_usize_list(&layer, "kernel_shape");
        assert_eq!(result, Ok(Some(vec![3, 3])));
    }

    #[test]
    fn parse_usize_list_missing_key() {
        let layer = make_layer("kernel_shape", vec![3, 3]);
        let result = parse_usize_list(&layer, "strides");
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn parse_usize_list_single_element() {
        let layer = make_layer("pads", vec![1]);
        let result = parse_usize_list(&layer, "pads");
        assert_eq!(result, Ok(Some(vec![1])));
    }

    #[test]
    fn parse_usize_list_no_params() {
        let layer = ONNXLayer {
            id: 0,
            name: "test".to_string(),
            op_type: "AveragePool".to_string(),
            inputs: vec![],
            outputs: vec![],
            shape: HashMap::new(),
            tensor: None,
            params: None,
            opset_version_number: 13,
        };
        assert_eq!(parse_usize_list(&layer, "kernel_shape"), Ok(None));
    }

    #[test]
    fn parse_usize_list_negative_value_is_err() {
        let layer = make_layer("kernel_shape", vec![-1, 3]);
        let result = parse_usize_list(&layer, "kernel_shape");
        assert!(result.is_err());
    }
}
