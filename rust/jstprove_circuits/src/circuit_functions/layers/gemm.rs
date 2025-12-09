//! GEMM (General Matrix Multiplication) layer implementation for JSTprove.
//!
//! This module defines the `GemmLayer` struct and its `LayerOp` implementation,
//! providing the circuit-level execution of an ONNX Gemm node using
//! ExpanderCompilerCollection. The layer performs:
//!
//!   1) Optional transposition of inputs according to ONNX attributes `transA` and `transB`.
//!   2) Integer matrix multiplication of the input and weight tensors.
//!   3) Addition of the bias tensor.
//!   4) Optional fixed-point rescaling, applied when the quantization pipeline
//!      indicates that the GEMM output must be shifted to match downstream scale.
//!   5) Optional Freivalds verification of the matrix product, which can
//!      probabilistically enforce `A * B == C` using asymptotically fewer
//!      multiplication constraints when the dimensions are favorable.
//!
//! The layer interfaces with:
//!
//!   * the quantized ONNX representation produced on the Python side,
//!   * utility modules for tensor loading, shaping, and quantized arithmetic,
//!   * Expander's `RootAPI` for constraint construction,
//!   * JSTprove's optimization patterns (for example, folding GEMM+ReLU).
//!
//! This file contains only the circuit logic for GEMM execution. Shape checks,
//! quantizer logic, kernel attributes, and graph-level optimizations occur
//! earlier in the pipeline. Runtime correctness is enforced in-circuit via
//! Expander constraints on the matrix multiplication, bias addition, optional
//! rescaling, and (when enabled) a Freivalds check on the core matrix product.

use std::collections::HashMap;

/// External crate imports
use ndarray::{ArrayD, Ix2};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::{
    CircuitError,
    layers::{
        LayerError, LayerKind,
        layer_ops::LayerOp,
        math::{
            freivalds_verify_matrix_product, matrix_addition, matrix_multiplication,
            unconstrained_matrix_multiplication,
        },
    },
    utils::{
        constants::{ALPHA, BETA, INPUT, TRANS_A, TRANS_B},
        graph_pattern_matching::PatternRegistry,
        onnx_model::{
            extract_params_and_expected_shape, get_input_name, get_param_or_default, get_w_or_b,
        },
        quantization::rescale_array,
        shaping::check_and_apply_transpose_array,
        tensor_ops::load_array_constants,
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
    weights: ArrayD<i64>,
    bias: ArrayD<i64>,
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
        input: HashMap<String, ArrayD<Variable>>,
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
        let mut weights_array = load_array_constants(api, &self.weights)
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

        let bias_array = load_array_constants(api, &self.bias);

        // Sanity check alpha and beta.
        check_alpha_beta(self.alpha, ALPHA, LayerKind::Gemm, &self.name)?;
        check_alpha_beta(self.beta, BETA, LayerKind::Gemm, &self.name)?;

        // Decide whether to use Freivalds or a fully constrained matmul.
        let (ell, m) = input_array.dim();
        let (m2, n) = weights_array.dim();
        if m != m2 {
            return Err(LayerError::ShapeMismatch {
                layer: LayerKind::Gemm,
                expected: vec![m],
                got: vec![m2],
                var_name: "GEMM: A.cols != B.rows".to_string(),
            }
            .into());
        }

        // Decide whether to use Freivalds, now with a configurable repetition count.
        let use_freivalds = should_use_freivalds(ell, m, n, self.freivalds_reps);

        if use_freivalds {
            println!(
                "Using Freivalds (reps = {}) for GEMM '{}' with dims ell={}, m={}, n={}",
                self.freivalds_reps, self.name, ell, m, n
            );
        }

        // Compute the core matrix product C_core = A * B.
        //
        // If Freivalds is enabled and beneficial, we compute C_core using
        // unconstrained operations and then verify A * B == C_core using
        // Freivalds. Otherwise, we fall back to a fully constrained matmul.
        let core_product: ArrayD<Variable> = if use_freivalds {
            // Unconstrained matmul: C_core = A * B
            let input_dyn_for_unconstrained = input_array.clone().into_dyn();
            let weights_dyn_for_unconstrained = weights_array.clone().into_dyn();

            let core_dyn = unconstrained_matrix_multiplication(
                api,
                input_dyn_for_unconstrained,
                weights_dyn_for_unconstrained,
                LayerKind::Gemm,
            )?;

            // Constrained Freivalds check: A * B == C_core with self.freivalds_reps repetitions.
            let input_dyn_for_check = input_array.clone().into_dyn();
            let weights_dyn_for_check = weights_array.clone().into_dyn();
            freivalds_verify_matrix_product(
                api,
                &input_dyn_for_check,
                &weights_dyn_for_check,
                &core_dyn,
                LayerKind::Gemm,
                self.freivalds_reps,
            )?;

            core_dyn
        } else {
            // Fully constrained matmul, as before.
            matrix_multiplication(
                api,
                input_array.into_dyn(),
                weights_array.into_dyn(),
                LayerKind::Gemm,
            )?
        };

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
        let gemm = Self {
            name: layer.name.clone(),
            index,
            weights: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[1])?,
            bias: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[2])?,
            is_rescale,
            source_scale_exponent: layer_context.n_bits,
            optimization_pattern,
            scaling: circuit_params.scale_exponent.into(),
            input_shape: expected_shape.clone(),
            alpha: get_param_or_default(&layer.name, ALPHA, &params, Some(&1.0))?,
            beta: get_param_or_default(&layer.name, BETA, &params, Some(&1.0))?,
            transa: get_param_or_default(&layer.name, TRANS_A, &params, Some(&0))?,
            transb: get_param_or_default(&layer.name, TRANS_B, &params, Some(&0))?,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            freivalds_reps: circuit_params.freivalds_reps,
        };
        Ok(Box::new(gemm))
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
    let ell_u = ell as u128;
    let m_u = m as u128;
    let n_u = n as u128;
    let reps_u = reps as u128;

    // Full deterministic matmul cost
    let cost_full = 2 * ell_u * m_u * n_u - ell_u * n_u;

    // Freivalds cost terms
    let s = ell_u * m_u + ell_u * n_u + m_u * n_u;
    let d = 2 * s - (m_u + 2 * ell_u);

    // If D is zero, Freivalds does not do useful arithmetic work
    if d == 0 {
        return false;
    }

    // Total Freivalds cost for `reps` repetitions
    let cost_f = reps_u * d;

    cost_f < cost_full
}
