//! GEMM (ONNX `Gemm`) layer implementation for JSTprove.
//!
//! This module defines `GemmLayer` and its `LayerOp` implementation, providing a
//! circuit-level execution of an ONNX `Gemm` node using ExpanderCompilerCollection.
//!
//! ## ONNX semantics (conceptual)
//! ONNX defines GEMM as
//!
//!     Y = alpha * A * B + beta * C
//!
//! with optional transpose flags `transA`, `transB` applied to `A`, `B` prior to
//! multiplication, and with `C` typically used as a bias term (potentially
//! broadcastable).
//!
//! ## JSTprove semantics (this implementation)
//! This implementation assumes the quantization pipeline has produced integer-valued
//! tensors encoded as field elements, and it currently enforces the common restricted
//! case `alpha = beta = 1` (validated at build/apply time).
//!
//! The layer performs:
//! 1) Optional transposition of `A` and/or `B` according to ONNX attributes `transA` and `transB`.
//! 2) Computation of the core product `C_core = A * B`.
//! 3) Constrained addition of the bias/`C` term.
//! 4) Optional fixed-point rescaling when required by the quantization configuration.
//! 5) Optional Freivalds verification of the core matrix product, which can
//!    probabilistically enforce `A * B == C_core` using fewer multiplication
//!    constraints than a fully-constrained matmul when dimensions are favorable.
//!
//! ## Pipeline boundaries
//! This file contains only circuit logic for GEMM execution. Shape checks, quantizer
//! logic, ONNX attribute parsing, and graph-level optimizations occur earlier in the
//! pipeline. Correctness is enforced in-circuit via Expander constraints on:
//! - bias addition and optional rescaling, and
//! - either a fully constrained matmul, or a Freivalds check linking `C_core` to `A` and `B`.
//!
//! Note: Freivalds is probabilistic; soundness depends on the field size and the
//! number of repetitions (`freivalds_reps`).

use std::collections::HashMap;

/// External crate imports
use ndarray::{ArrayD, Ix2};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::{
    CircuitError,
    gadgets::linear_algebra::{
        freivalds_verify_matrix_product, matrix_addition, matrix_multiplication,
        unconstrained_matrix_multiplication,
    },
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{ALPHA, BETA, INPUT, TRANS_A, TRANS_B},
        graph_pattern_matching::PatternRegistry,
        onnx_model::{
            extract_params_and_expected_shape, get_input_name, get_param_or_default, get_w_or_b,
        },
        quantization::rescale_array,
        shaping::check_and_apply_transpose_array,
        tensor_ops::load_array_constants_or_get_inputs,
    },
};

// -----------------------------------------------------------------------------
// STRUCT: GemmLayer
// -----------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug)]
pub struct GemmLayer {
    name: String,
    index: usize,
    weights: Option<ArrayD<i64>>,
    bias: Option<ArrayD<i64>>,
    is_rescale: bool,
    source_scale_exponent: usize,
    optimization_pattern: PatternRegistry,
    scaling: u64,
    input_shape: Vec<usize>,
    alpha: f32,
    beta: f32,
    transa: usize,
    transb: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
    freivalds_reps: usize,
}

// -----------------------------------------------------------------------------
// IMPL: LayerOp for GemmLayer
// -----------------------------------------------------------------------------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GemmLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let is_relu = matches!(self.optimization_pattern, PatternRegistry::GemmRelu);

        let input_name = get_input_name(&self.inputs, 0, LayerKind::Gemm, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Gemm,
                name: input_name.clone(),
            })?
            .clone();

        // Load and shape input and weights as 2D matrices.
        let mut input_array =
            layer_input
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::Gemm,
                    msg: format!("Expected 2D input for layer {}", self.name),
                })?;
        let w_name = get_input_name(&self.inputs, 1, LayerKind::Gemm, "weights")?;
        let mut weights_array = load_array_constants_or_get_inputs(
            api,
            input,
            w_name,
            &self.weights,
            LayerKind::Gemm,
        )?
        .into_dimensionality::<Ix2>()
        .map_err(|_| LayerError::InvalidShape {
            layer: LayerKind::Gemm,
            msg: format!("Expected 2D weights array for layer {}", self.name),
        })?;

        // Apply transposes according to ONNX attributes.
        input_array = check_and_apply_transpose_array(
            input_array,
            self.transa,
            TRANS_A,
            &LayerKind::Gemm,
            &self.name,
        )?;
        weights_array = check_and_apply_transpose_array(
            weights_array,
            self.transb,
            TRANS_B,
            &LayerKind::Gemm,
            &self.name,
        )?;

        let b_name = get_input_name(&self.inputs, 2, LayerKind::Gemm, "bias")?;
        let bias_array =
            load_array_constants_or_get_inputs(api, input, b_name, &self.bias, LayerKind::Gemm)?;

        // Sanity check alpha and beta.
        check_alpha_beta(self.alpha, ALPHA, LayerKind::Gemm, &self.name)?;
        check_alpha_beta(self.beta, BETA, LayerKind::Gemm, &self.name)?;

        let core_product = compute_core_product(
            api,
            &input_array,
            &weights_array,
            LayerKind::Gemm,
            self.freivalds_reps,
        )?;

        // Add bias (constrained) on top of the core product.
        let result = matrix_addition(api, &core_product, bias_array, LayerKind::Gemm)?;

        // Optional rescaling (quantized fixed-point).
        let mut out_array = result.into_dyn();
        if self.is_rescale {
            let k = usize::try_from(self.scaling).map_err(|_| LayerError::Other {
                layer: LayerKind::Gemm,
                msg: "Cannot convert scaling to usize".to_string(),
            })?;
            let s = self.source_scale_exponent.checked_sub(1).ok_or_else(|| {
                LayerError::InvalidParameterValue {
                    layer: LayerKind::Gemm,
                    layer_name: self.name.clone(),
                    param_name: "source_scale_exponent".to_string(),
                    value: self.source_scale_exponent.to_string(),
                }
            })?;

            out_array =
                rescale_array(api, out_array, k, s, is_relu).map_err(|e| LayerError::Other {
                    layer: LayerKind::Gemm,
                    msg: format!("Rescale failed: {e}"),
                })?;
        }

        Ok((self.outputs.clone(), out_array))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Gemm,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;
        let freivalds_reps = circuit_params.freivalds_reps;
        if freivalds_reps == 0 {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::Gemm,
                layer_name: layer.name.clone(),
                param_name: "freivalds_reps".to_string(),
                value: freivalds_reps.to_string(),
            }
            .into());
        }

        let w_name = get_input_name(&layer.inputs, 1, LayerKind::Gemm, "weights")?;
        let b_name = get_input_name(&layer.inputs, 2, LayerKind::Gemm, "bias")?;

        let (weights, bias) = if layer_context.weights_as_inputs {
            (None, None)
        } else {
            (
                Some(get_w_or_b(&layer_context.w_and_b_map, w_name)?),
                Some(get_w_or_b(&layer_context.w_and_b_map, b_name)?),
            )
        };

        let gemm = Self {
            name: layer.name.clone(),
            index,
            weights,
            bias,
            is_rescale,
            source_scale_exponent: layer_context.n_bits_for(&layer.name),
            optimization_pattern,
            scaling: circuit_params.scale_exponent.into(),
            input_shape: expected_shape.clone(),
            alpha: get_param_or_default(&layer.name, ALPHA, &params, Some(&1.0))?,
            beta: get_param_or_default(&layer.name, BETA, &params, Some(&1.0))?,
            transa: get_param_or_default(&layer.name, TRANS_A, &params, Some(&0))?,
            transb: get_param_or_default(&layer.name, TRANS_B, &params, Some(&0))?,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            freivalds_reps,
        };
        Ok(Box::new(gemm))
    }
}

// -----------------------------------------------------------------------------
// FUNCTION: compute_core_product
// -----------------------------------------------------------------------------

/// Computes the core matrix product C = A * B for GEMM, optionally using
/// Freivalds' algorithm to reduce constraint cost.
///
/// If Freivalds is enabled and cheaper under the internal cost model, this
/// function:
///   1) computes C using unconstrained operations, and
///   2) enforces correctness via a constrained Freivalds check.
///
/// Otherwise, it falls back to a fully constrained matrix multiplication.
///
/// Arguments:
///   - api: circuit builder
///   - matrix_a: left matrix A (2D)
///   - matrix_b: right matrix B (2D)
///   - layer_type: layer identifier for error reporting
///   - freivalds_reps: number of Freivalds repetitions
///
/// Returns:
///   - Ok(ArrayD<Variable>) representing C = A * B
///
/// # Errors
/// Returns a LayerError if:
///   - inputs are not 2D,
///   - inner dimensions do not match,
///   - freivalds_reps is zero when Freivalds is selected.
fn compute_core_product<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_array: &ndarray::Array2<Variable>,
    weights_array: &ndarray::Array2<Variable>,
    layer_kind: LayerKind,
    freivalds_reps: usize,
) -> Result<ArrayD<Variable>, CircuitError> {
    let (ell, m) = input_array.dim();
    let (m2, n) = weights_array.dim();
    if m != m2 {
        return Err(LayerError::ShapeMismatch {
            layer: layer_kind,
            expected: vec![m],
            got: vec![m2],
            var_name: "GEMM: A.cols != B.rows".to_string(),
        }
        .into());
    }

    let use_freivalds = should_use_freivalds(ell, m, n, freivalds_reps);

    if use_freivalds {
        let core_dyn = unconstrained_matrix_multiplication(
            api,
            input_array.clone().into_dyn(),
            weights_array.clone().into_dyn(),
            layer_kind.clone(),
        )?;

        freivalds_verify_matrix_product(
            api,
            &input_array.clone().into_dyn(),
            &weights_array.clone().into_dyn(),
            &core_dyn,
            layer_kind,
            freivalds_reps,
        )?;

        Ok(core_dyn)
    } else {
        matrix_multiplication(
            api,
            input_array.clone().into_dyn(),
            weights_array.clone().into_dyn(),
            layer_kind,
        )
        .map_err(Into::into)
    }
}

// -----------------------------------------------------------------------------
// FUNCTION: check_alpha_beta
// -----------------------------------------------------------------------------

fn check_alpha_beta(
    val: f32,
    var_name: &str,
    layer_type: LayerKind,
    layer_name: &str,
) -> Result<(), LayerError> {
    if (val - 1.0).abs() > 1e-6 {
        return Err(LayerError::InvalidParameterValue {
            layer: layer_type,
            layer_name: layer_name.to_string(),
            param_name: var_name.to_string(),
            value: val.to_string(),
        });
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// FUNCTION: should_use_freivalds
// -----------------------------------------------------------------------------
//
// Decide whether using Freivalds' algorithm (with a given number of
// repetitions) is cheaper than performing a fully constrained matrix
// multiplication inside the circuit.
//
// COST MODEL (treating add and mul as equal unit operations):
//
//   Full constrained matmul for A (ell x m) and B (m x n):
//       cost_full = 2 * ell * m * n - ell * n
//
//   One Freivalds repetition:
//       S = ell*m + ell*n + m*n
//       D = 2*S - (m + 2*ell)
//
//   With `reps` repetitions:
//       cost_f = reps * D
//
// Decision rule:
//       use Freivalds  <=>  cost_f < cost_full
//
// All internal arithmetic is promoted to u128 to reduce overflow risk.
fn should_use_freivalds(ell: usize, m: usize, n: usize, reps: usize) -> bool {
    // reps == 0 must never allow Freivalds, otherwise core_product can be left unconstrained.
    if reps == 0 {
        return false;
    }

    let ell_u = ell as u128;
    let m_u = m as u128;
    let n_u = n as u128;
    let reps_u = reps as u128;

    // Full deterministic matmul cost:
    // cost_full = 2*ell*m*n - ell*n
    let cost_full = (2u128)
        .saturating_mul(ell_u)
        .saturating_mul(m_u)
        .saturating_mul(n_u)
        .saturating_sub(ell_u.saturating_mul(n_u));

    // Freivalds per-repetition "work" estimate:
    // s = ell*m + ell*n + m*n
    // d = 2*s - (m + 2*ell)
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

    // Total Freivalds cost
    let cost_f = reps_u.saturating_mul(d);

    cost_f < cost_full
}
