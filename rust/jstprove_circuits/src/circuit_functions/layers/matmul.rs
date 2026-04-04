// ONNX `MatMul` layer for ZK circuits.
//
// # ZK approach
// MatMul computes C = A * B where A and B are matrices (or batched matrices).
// This uses the same approach as GemmLayer: circuit multiplication + addition constraints.
// The output is rescaled if needed (multiplying two α-scaled values gives α²).
//
// # Supported configurations
// - 2D: [M, K] @ [K, N] → [M, N]
// - Higher-rank inputs are rejected at build time.

use std::collections::HashMap;

use ndarray::{ArrayD, Ix2};

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
    n_bits: usize,
    scaling: u64,
    inputs: Vec<String>,
    outputs: Vec<String>,
    freivalds_reps: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for MatMulLayer {
    #[allow(clippy::too_many_lines)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::MatMul, INPUT)?;
        let layer_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::MatMul,
                name: input_name.to_string(),
            })?
            .clone();

        let input_array =
            layer_input
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("Expected 2D input for MatMul layer {}", self.name),
                })?;

        let w_name = get_input_name(&self.inputs, 1, LayerKind::MatMul, "weights")?;
        let weights_array = load_array_constants_or_get_inputs(
            api,
            input,
            w_name,
            &self.weights,
            LayerKind::MatMul,
        )?
        .into_dimensionality::<Ix2>()
        .map_err(|_| LayerError::InvalidShape {
            layer: LayerKind::MatMul,
            msg: format!("Expected 2D weights for MatMul layer {}", self.name),
        })?;

        let (ell, m) = input_array.dim();
        let (m2, n) = weights_array.dim();
        if m != m2 {
            return Err(LayerError::ShapeMismatch {
                layer: LayerKind::MatMul,
                expected: vec![m],
                got: vec![m2],
                var_name: "MatMul: A.cols != B.rows".to_string(),
            }
            .into());
        }

        let use_freivalds = self.freivalds_reps > 0 && {
            let ell_u = ell as u128;
            let m_u = m as u128;
            let n_u = n as u128;
            let reps_u = self.freivalds_reps as u128;
            let cost_full = 2u128 * ell_u * m_u * n_u - ell_u * n_u;
            let s = ell_u * m_u + ell_u * n_u + m_u * n_u;
            let d = 2u128 * s - (m_u + 2u128 * ell_u);
            d > 0 && reps_u * d < cost_full
        };

        let core = if use_freivalds {
            let core_dyn = unconstrained_matrix_multiplication(
                api,
                input_array.clone().into_dyn(),
                weights_array.clone().into_dyn(),
                LayerKind::MatMul,
            )?;
            freivalds_verify_matrix_product(
                api,
                &input_array.clone().into_dyn(),
                &weights_array.clone().into_dyn(),
                &core_dyn,
                LayerKind::MatMul,
                self.freivalds_reps,
            )?;
            core_dyn
        } else {
            matrix_multiplication(
                api,
                input_array.clone().into_dyn(),
                weights_array.clone().into_dyn(),
                LayerKind::MatMul,
            )
            .map_err(CircuitError::from)?
        };

        let out_array = maybe_rescale(
            api,
            logup_ctx,
            core,
            &MaybeRescaleParams {
                is_rescale: self.is_rescale,
                scaling_exponent: self.scaling,
                n_bits: self.n_bits,
                is_relu: false,
                layer_kind: LayerKind::MatMul,
                layer_name: self.name.clone(),
            },
        )?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(
        clippy::too_many_lines,
        clippy::uninlined_format_args,
        clippy::redundant_closure_for_method_calls
    )]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let freivalds_reps = circuit_params.freivalds_reps;

        // Validate we have two inputs.
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::MatMul,
                name: "input A".to_string(),
            })?;
        let w_name = get_input_name(&layer.inputs, 1, LayerKind::MatMul, "input B")?;

        // Validate 2D shapes.
        let input_name = layer.inputs.first().unwrap();
        let a_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::MatMul,
                msg: format!("missing input shape for '{input_name}'"),
            })?;
        if a_shape.len() != 2 {
            return Err(LayerError::Other {
                layer: LayerKind::MatMul,
                msg: format!(
                    "MatMul only supports 2D inputs in the Expander backend; \
                     input A has rank {} (shape {:?}). Use Gemm for 2D or reshape inputs.",
                    a_shape.len(),
                    a_shape
                ),
            }
            .into());
        }

        let weights = if layer_context.weights_as_inputs {
            None
        } else {
            get_w_or_b(layer_context.w_and_b_map, w_name).ok()
        };

        let scaling: u64 = circuit_params.scale_exponent.into();

        Ok(Box::new(Self {
            name: layer.name.clone(),
            weights,
            is_rescale,
            n_bits: layer_context.n_bits_for(&layer.name),
            scaling,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            freivalds_reps,
        }))
    }
}

#[cfg(test)]
mod tests {
    /// Evaluate the Freivalds cost check used in MatMulLayer::apply.
    ///
    /// Returns `true` when Freivalds verification with `reps` repetitions is
    /// cheaper than computing the full matrix product directly.
    fn freivalds_cheaper(ell: u128, m: u128, n: u128, reps: u128) -> bool {
        if reps == 0 {
            return false;
        }
        let cost_full = 2u128 * ell * m * n - ell * n;
        let s = ell * m + ell * n + m * n;
        let d = 2u128 * s - (m + 2u128 * ell);
        d > 0 && reps * d < cost_full
    }

    #[test]
    fn freivalds_beneficial_for_large_matrices() {
        // 100×100 × 100×100 with 3 reps — Freivalds should win.
        assert!(freivalds_cheaper(100, 100, 100, 3));
    }

    #[test]
    fn freivalds_not_beneficial_for_tiny_matrices() {
        // 2×2 × 2×2 with 3 reps — Freivalds is more expensive.
        // cost_full = 2*2*2*2 - 2*2 = 12
        // s = 4+4+4 = 12, d = 24 - 6 = 18, 3*18=54 > 12
        assert!(!freivalds_cheaper(2, 2, 2, 3));
    }

    #[test]
    fn freivalds_zero_reps_not_beneficial() {
        assert!(!freivalds_cheaper(1000, 1000, 1000, 0));
    }

    #[test]
    fn freivalds_single_row_not_beneficial() {
        // ell=1: d = 2*(m+n+m*n) - (m+2), and for tiny matrices it is still
        // too costly even when positive.
        // For m=n=1: cost_full=1, s=3, d=3, 1*3=3 > 1 -> not beneficial.
        assert!(!freivalds_cheaper(1, 1, 1, 1));
    }
}
