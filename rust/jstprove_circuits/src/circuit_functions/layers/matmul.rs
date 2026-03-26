//! ONNX `MatMul` layer implementation for JSTprove.
//!
//! MatMul computes `Y = A @ B` — a pure matrix multiplication with no alpha,
//! beta, bias, or transpose attributes. This distinguishes it from ONNX `Gemm`
//! which computes `Y = alpha * A' * B' + beta * C`.
//!
//! For quantized integer tensors, the product of two scale-encoded values
//! produces a double-scaled result, so rescaling is applied when configured.
//!
//! Like `GemmLayer`, this implementation uses Freivalds' algorithm when the
//! cost model determines it is cheaper than a fully-constrained matmul.
//!
//! Batched MatMul (ONNX semantics) is supported: leading batch dimensions are
//! broadcast per numpy rules and 2-D matrix multiplication is performed on the
//! last two axes of each operand.

use std::collections::HashMap;

use ndarray::{ArrayD, Axis, Ix2, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    gadgets::linear_algebra::{
        freivalds_verify_matrix_product, matrix_multiplication, unconstrained_matrix_multiplication,
    },
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
        quantization::{MaybeRescaleParams, maybe_rescale},
        tensor_ops::load_array_constants_or_get_inputs,
    },
};

#[derive(Debug)]
pub struct MatMulLayer {
    name: String,
    weights: Option<ArrayD<i64>>,
    is_rescale: bool,
    source_scale_exponent: usize,
    scaling: u64,
    inputs: Vec<String>,
    outputs: Vec<String>,
    freivalds_reps: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for MatMulLayer {
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::MatMul, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::MatMul,
                name: input_name.clone(),
            })?
            .clone();

        if layer_input.ndim() < 2 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::MatMul,
                msg: format!(
                    "Expected rank >= 2 for input of layer '{}', got rank {}",
                    self.name,
                    layer_input.ndim()
                ),
            }
            .into());
        }

        let w_name = get_input_name(&self.inputs, 1, LayerKind::MatMul, "weights")?;
        let weights_array = load_array_constants_or_get_inputs(
            api,
            input,
            w_name,
            &self.weights,
            LayerKind::MatMul,
        )?;

        if weights_array.ndim() < 2 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::MatMul,
                msg: format!(
                    "Expected rank >= 2 for weights of layer '{}', got rank {}",
                    self.name,
                    weights_array.ndim()
                ),
            }
            .into());
        }

        let core_product = compute_matmul_product(
            api,
            &layer_input,
            &weights_array,
            LayerKind::MatMul,
            self.freivalds_reps,
        )?;

        let out_array = maybe_rescale(
            api,
            logup_ctx,
            core_product,
            &MaybeRescaleParams {
                is_rescale: self.is_rescale,
                scaling_exponent: self.scaling,
                n_bits: self.source_scale_exponent,
                is_relu: false,
                layer_kind: LayerKind::MatMul,
                layer_name: self.name.clone(),
            },
        )?;
        Ok((self.outputs.clone(), out_array))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let freivalds_reps = circuit_params.freivalds_reps;
        // freivalds_reps == 0 is valid: it disables Freivalds and falls back to
        // direct matrix multiplication (should_use_freivalds returns false when reps == 0).

        let w_name = get_input_name(&layer.inputs, 1, LayerKind::MatMul, "weights")?;

        // Only fetch from the initializer map when the name is actually present there.
        // If it is absent, the RHS is a dynamic activation tensor supplied at apply time.
        let weights = if layer_context.weights_as_inputs {
            None
        } else if layer_context.w_and_b_map.contains_key(w_name) {
            Some(get_w_or_b(layer_context.w_and_b_map, w_name)?)
        } else {
            None
        };

        let matmul = Self {
            name: layer.name.clone(),
            weights,
            is_rescale,
            source_scale_exponent: layer_context.n_bits_for(&layer.name),
            scaling: circuit_params.scale_exponent.into(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            freivalds_reps,
        };
        Ok(Box::new(matmul))
    }
}

// ---------------------------------------------------------------------------
// Batch-shape helpers
// ---------------------------------------------------------------------------

/// Computes the broadcast shape for the leading batch dimensions of a MatMul.
///
/// Follows numpy broadcasting rules: dimensions are aligned from the right,
/// missing leading dimensions are treated as 1, and two dimensions are
/// compatible when they are equal or one of them is 1.
fn broadcast_batch_shapes(
    a: &[usize],
    b: &[usize],
    layer_kind: &LayerKind,
) -> Result<Vec<usize>, CircuitError> {
    let out_rank = a.len().max(b.len());
    let mut out = vec![0usize; out_rank];
    for i in 0..out_rank {
        let a_dim = if i + a.len() < out_rank {
            1
        } else {
            a[i + a.len() - out_rank]
        };
        let b_dim = if i + b.len() < out_rank {
            1
        } else {
            b[i + b.len() - out_rank]
        };
        out[i] = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(LayerError::ShapeMismatch {
                layer: layer_kind.clone(),
                expected: a.to_vec(),
                got: b.to_vec(),
                var_name: format!("MatMul batch dim {i}: {a_dim} vs {b_dim} are not broadcastable"),
            }
            .into());
        };
    }
    Ok(out)
}

/// Converts a flat (row-major) index into a multi-dimensional index.
fn flat_to_nd_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = vec![0usize; shape.len()];
    let mut rem = flat;
    for d in (0..shape.len()).rev() {
        if shape[d] == 0 {
            break;
        }
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    idx
}

/// Maps an output batch multi-index back to a flat index in `batch`, respecting
/// broadcasting (size-1 dimensions always map to index 0).
fn nd_broadcast_to_flat(out_idx: &[usize], batch: &[usize], out_batch: &[usize]) -> usize {
    let pad = out_batch.len().saturating_sub(batch.len());
    let mut flat = 0usize;
    for (d, &dim) in batch.iter().enumerate() {
        let out_d = d + pad;
        let i = if dim == 1 { 0 } else { out_idx[out_d] };
        flat = flat * dim + i;
    }
    flat
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

fn compute_matmul_product<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_array: &ArrayD<Variable>,
    weights_array: &ArrayD<Variable>,
    layer_kind: LayerKind,
    freivalds_reps: usize,
) -> Result<ArrayD<Variable>, CircuitError> {
    let a_rank = input_array.ndim();
    let b_rank = weights_array.ndim();

    // Extract matrix dims from the last two axes.
    let m = input_array.shape()[a_rank - 2];
    let k = input_array.shape()[a_rank - 1];
    let k2 = weights_array.shape()[b_rank - 2];
    let n = weights_array.shape()[b_rank - 1];

    if k != k2 {
        return Err(LayerError::ShapeMismatch {
            layer: layer_kind,
            expected: vec![k],
            got: vec![k2],
            var_name: "MatMul: A.cols != B.rows".to_string(),
        }
        .into());
    }

    let a_batch = input_array.shape()[..a_rank - 2].to_vec();
    let b_batch = weights_array.shape()[..b_rank - 2].to_vec();
    let out_batch = broadcast_batch_shapes(&a_batch, &b_batch, &layer_kind)?;

    let prod_a: usize = a_batch.iter().product::<usize>().max(1);
    let prod_b: usize = b_batch.iter().product::<usize>().max(1);
    let prod_out: usize = out_batch.iter().product::<usize>().max(1);

    // Flatten batch dimensions for indexed slicing: [batch, row, col].
    let a_flat: ArrayD<Variable> = input_array
        .view()
        .into_shape_with_order(IxDyn(&[prod_a, m, k]))
        .map_err(|_| LayerError::InvalidShape {
            layer: layer_kind.clone(),
            msg: format!("cannot reshape A to [{prod_a}, {m}, {k}]"),
        })?
        .into_owned();

    let b_flat: ArrayD<Variable> = weights_array
        .view()
        .into_shape_with_order(IxDyn(&[prod_b, k, n]))
        .map_err(|_| LayerError::InvalidShape {
            layer: layer_kind.clone(),
            msg: format!("cannot reshape B to [{prod_b}, {k}, {n}]"),
        })?
        .into_owned();

    // Build the output array.
    let mut out_shape = out_batch.clone();
    out_shape.extend([m, n]);
    let zero = api.constant(0u32);
    let mut output = ArrayD::from_elem(IxDyn(&out_shape), zero);

    let use_freivalds = should_use_freivalds(m, k, n, freivalds_reps);

    for flat_out in 0..prod_out {
        let out_idx = flat_to_nd_index(flat_out, &out_batch);
        let a_flat_idx = nd_broadcast_to_flat(&out_idx, &a_batch, &out_batch);
        let b_flat_idx = nd_broadcast_to_flat(&out_idx, &b_batch, &out_batch);

        // Extract 2-D slices.
        let a_slice: ArrayD<Variable> = a_flat
            .index_axis(Axis(0), a_flat_idx)
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_kind.clone(),
                msg: "A batch slice is not 2D after reshape".to_string(),
            })?
            .into_dyn()
            .into_owned();

        let b_slice: ArrayD<Variable> = b_flat
            .index_axis(Axis(0), b_flat_idx)
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_kind.clone(),
                msg: "B batch slice is not 2D after reshape".to_string(),
            })?
            .into_dyn()
            .into_owned();

        let result_dyn = if use_freivalds {
            let core = unconstrained_matrix_multiplication(
                api,
                a_slice.clone(),
                b_slice.clone(),
                layer_kind.clone(),
            )?;
            freivalds_verify_matrix_product(
                api,
                &a_slice,
                &b_slice,
                &core,
                layer_kind.clone(),
                freivalds_reps,
            )?;
            core
        } else {
            matrix_multiplication(api, a_slice, b_slice, layer_kind.clone())?
        };

        let result_2d =
            result_dyn
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: layer_kind.clone(),
                    msg: "matmul result is not 2D".to_string(),
                })?;

        // Write the 2-D result into the output at the correct batch position.
        for i in 0..m {
            for j in 0..n {
                let mut full_idx = out_idx.clone();
                full_idx.push(i);
                full_idx.push(j);
                output[IxDyn(&full_idx)] = result_2d[[i, j]];
            }
        }
    }

    Ok(output)
}

fn should_use_freivalds(ell: usize, m: usize, n: usize, reps: usize) -> bool {
    if reps == 0 {
        return false;
    }

    let ell_u = ell as u128;
    let m_u = m as u128;
    let n_u = n as u128;
    let reps_u = reps as u128;

    let cost_full = (2u128)
        .saturating_mul(ell_u)
        .saturating_mul(m_u)
        .saturating_mul(n_u)
        .saturating_sub(ell_u.saturating_mul(n_u));

    let s = ell_u
        .saturating_mul(m_u)
        .saturating_add(ell_u.saturating_mul(n_u))
        .saturating_add(m_u.saturating_mul(n_u));

    let d = (2u128)
        .saturating_mul(s)
        .saturating_sub(m_u.saturating_add((2u128).saturating_mul(ell_u)));

    if d == 0 {
        return false;
    }

    let cost_f = reps_u.saturating_mul(d);
    cost_f < cost_full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freivalds_zero_reps_always_false() {
        assert!(!should_use_freivalds(100, 200, 50, 0));
    }

    #[test]
    fn freivalds_small_matrix_prefers_direct() {
        assert!(!should_use_freivalds(2, 2, 2, 1));
    }

    #[test]
    fn freivalds_large_matrix_prefers_freivalds() {
        assert!(should_use_freivalds(100, 200, 50, 1));
    }

    #[test]
    fn broadcast_batch_shapes_pure_2d() {
        let out = broadcast_batch_shapes(&[], &[], &LayerKind::MatMul).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn broadcast_batch_shapes_equal() {
        let out = broadcast_batch_shapes(&[3, 4], &[3, 4], &LayerKind::MatMul).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn broadcast_batch_shapes_broadcast_left() {
        let out = broadcast_batch_shapes(&[1, 4], &[3, 4], &LayerKind::MatMul).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn broadcast_batch_shapes_incompatible_errors() {
        let res = broadcast_batch_shapes(&[2, 4], &[3, 4], &LayerKind::MatMul);
        assert!(res.is_err());
    }

    #[test]
    fn flat_to_nd_and_back_roundtrip() {
        let shape = [2usize, 3, 4];
        for flat in 0..24 {
            let idx = flat_to_nd_index(flat, &shape);
            let back = nd_broadcast_to_flat(&idx, &shape, &shape);
            assert_eq!(flat, back, "roundtrip failed for flat={flat}");
        }
    }
}
