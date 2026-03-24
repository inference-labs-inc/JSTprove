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

use std::collections::HashMap;

use ndarray::{ArrayD, Ix2};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    gadgets::linear_algebra::{
        freivalds_verify_matrix_product, matrix_multiplication,
        unconstrained_matrix_multiplication,
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

        let input_array =
            layer_input
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("Expected 2D input for layer {}", self.name),
                })?;

        let w_name = get_input_name(&self.inputs, 1, LayerKind::MatMul, "weights")?;
        let weights_array =
            load_array_constants_or_get_inputs(api, input, w_name, &self.weights, LayerKind::MatMul)?
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("Expected 2D weights for layer {}", self.name),
                })?;

        let core_product = compute_matmul_product(
            api,
            &input_array,
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
        if freivalds_reps == 0 {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::MatMul,
                layer_name: layer.name.clone(),
                param_name: "freivalds_reps".to_string(),
                value: freivalds_reps.to_string(),
            }
            .into());
        }

        let w_name = get_input_name(&layer.inputs, 1, LayerKind::MatMul, "weights")?;

        let weights = if layer_context.weights_as_inputs {
            None
        } else {
            Some(get_w_or_b(layer_context.w_and_b_map, w_name)?)
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

fn compute_matmul_product<C: Config, Builder: RootAPI<C>>(
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
            var_name: "MatMul: A.cols != B.rows".to_string(),
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
}
