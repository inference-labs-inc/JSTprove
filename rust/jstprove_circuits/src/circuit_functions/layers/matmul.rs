// ONNX `MatMul` layer for ZK circuits.
//
// # ZK approach
// MatMul computes C = A * B where A and B are matrices (or batched matrices).
// This uses the same approach as GemmLayer: circuit multiplication + addition constraints.
// The output is rescaled if needed (multiplying two α-scaled values gives α²).
//
// # Supported configurations
// - 2D: [M, K] @ [K, N] → [M, N]
// - 3D: [B, M, K] @ [K, N] → [B, M, N]  (B broadcast over batch)
//        [B, M, K] @ [B, K, N] → [B, M, N]  (batched both)

use std::collections::HashMap;

use ndarray::{ArrayD, Ix2, Ix3, IxDyn, s};

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

        let w_name = get_input_name(&self.inputs, 1, LayerKind::MatMul, "weights")?;
        let weights_dyn = load_array_constants_or_get_inputs(
            api,
            input,
            w_name,
            &self.weights,
            LayerKind::MatMul,
        )?;

        let core = if layer_input.ndim() == 3 {
            self.apply_batched(api, layer_input, weights_dyn)?
        } else {
            self.apply_2d(api, layer_input, weights_dyn)?
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

        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::MatMul,
                name: "input A".to_string(),
            })?;
        let w_name = get_input_name(&layer.inputs, 1, LayerKind::MatMul, "input B")?;

        let input_name = layer.inputs.first().unwrap();
        let a_shape: Vec<usize> = if let Some(s) = layer_context.shapes_map.get(input_name.as_str())
        {
            s.clone()
        } else {
            let out_name = layer
                .outputs
                .first()
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("missing input shape for '{input_name}'"),
                })?;
            let out_shape = layer_context.shapes_map.get(out_name.as_str());
            let b_shape_ref = layer_context.shapes_map.get(w_name);
            match (out_shape, b_shape_ref) {
                (Some(os), Some(bs)) if os.len() >= 2 && !bs.is_empty() => {
                    let k = bs[bs.len().saturating_sub(2)];
                    let m = os[os.len() - 2];
                    let batch = &os[..os.len() - 2];
                    let mut shape = batch.to_vec();
                    shape.push(m);
                    shape.push(k);
                    shape
                }
                _ => {
                    return Err(LayerError::InvalidShape {
                        layer: LayerKind::MatMul,
                        msg: format!("missing input shape for '{input_name}'"),
                    }
                    .into());
                }
            }
        };
        if a_shape.len() != 2 && a_shape.len() != 3 {
            return Err(LayerError::Other {
                layer: LayerKind::MatMul,
                msg: format!(
                    "MatMul supports rank-2 and rank-3 inputs in the Expander backend; \
                     input A has rank {} (shape {:?}).",
                    a_shape.len(),
                    a_shape
                ),
            }
            .into());
        }

        let b_shape: Vec<usize> = if let Some(s) = layer_context.shapes_map.get(w_name) {
            s.clone()
        } else {
            let out_name = layer
                .outputs
                .first()
                .ok_or_else(|| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("missing input shape for '{w_name}'"),
                })?;
            let out_shape = layer_context.shapes_map.get(out_name.as_str());
            match out_shape {
                Some(os) if os.len() >= 2 => {
                    let k = a_shape[a_shape.len() - 1];
                    let n = os[os.len() - 1];
                    if a_shape.len() == 3 {
                        vec![a_shape[0], k, n]
                    } else {
                        vec![k, n]
                    }
                }
                _ => {
                    return Err(LayerError::InvalidShape {
                        layer: LayerKind::MatMul,
                        msg: format!("missing input shape for '{w_name}'"),
                    }
                    .into());
                }
            }
        };
        if b_shape.len() != 2 && b_shape.len() != 3 {
            return Err(LayerError::Other {
                layer: LayerKind::MatMul,
                msg: format!(
                    "MatMul supports rank-2 and rank-3 inputs in the Expander backend; \
                     input B has rank {} (shape {:?}).",
                    b_shape.len(),
                    b_shape
                ),
            }
            .into());
        }

        if a_shape.len() == 2 && b_shape.len() == 3 {
            return Err(LayerError::Other {
                layer: LayerKind::MatMul,
                msg: format!(
                    "MatMul: rank-2 input A (shape {:?}) with rank-3 input B (shape {:?}) \
                     is not supported; reshape A to rank-3 or B to rank-2.",
                    a_shape, b_shape
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

impl MatMulLayer {
    fn matmul_2d_core<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        a_2d: ArrayD<Variable>,
        b_2d: ArrayD<Variable>,
    ) -> Result<ArrayD<Variable>, CircuitError> {
        let (ell, m) = {
            let sh = a_2d.shape();
            (sh[0], sh[1])
        };
        let n = b_2d.shape()[1];

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

        if use_freivalds {
            let core_dyn = unconstrained_matrix_multiplication(
                api,
                a_2d.clone(),
                b_2d.clone(),
                LayerKind::MatMul,
            )?;
            freivalds_verify_matrix_product(
                api,
                &a_2d,
                &b_2d,
                &core_dyn,
                LayerKind::MatMul,
                self.freivalds_reps,
            )?;
            Ok(core_dyn)
        } else {
            matrix_multiplication(api, a_2d, b_2d, LayerKind::MatMul).map_err(CircuitError::from)
        }
    }

    fn apply_2d<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        layer_input: ArrayD<Variable>,
        weights_dyn: ArrayD<Variable>,
    ) -> Result<ArrayD<Variable>, CircuitError> {
        let input_array =
            layer_input
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("Expected 2D input for MatMul layer {}", self.name),
                })?;

        let weights_array =
            weights_dyn
                .into_dimensionality::<Ix2>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("Expected 2D weights for MatMul layer {}", self.name),
                })?;

        let (_, m) = input_array.dim();
        let (m2, _) = weights_array.dim();
        if m != m2 {
            return Err(LayerError::ShapeMismatch {
                layer: LayerKind::MatMul,
                expected: vec![m],
                got: vec![m2],
                var_name: "MatMul: A.cols != B.rows".to_string(),
            }
            .into());
        }

        self.matmul_2d_core(api, input_array.into_dyn(), weights_array.into_dyn())
    }

    fn apply_batched<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        layer_input: ArrayD<Variable>,
        weights_dyn: ArrayD<Variable>,
    ) -> Result<ArrayD<Variable>, CircuitError> {
        let input_3d =
            layer_input
                .into_dimensionality::<Ix3>()
                .map_err(|_| LayerError::InvalidShape {
                    layer: LayerKind::MatMul,
                    msg: format!("Expected 3D input for batched MatMul layer {}", self.name),
                })?;

        let (batch, rows_a, k_a) = input_3d.dim();
        let b_is_batched = weights_dyn.ndim() == 3;

        if b_is_batched {
            let weights_3d =
                weights_dyn
                    .into_dimensionality::<Ix3>()
                    .map_err(|_| LayerError::InvalidShape {
                        layer: LayerKind::MatMul,
                        msg: format!("Expected 3D weights for batched MatMul layer {}", self.name),
                    })?;
            let (batch_b, k_b, cols_b) = weights_3d.dim();
            if batch != batch_b {
                return Err(LayerError::ShapeMismatch {
                    layer: LayerKind::MatMul,
                    expected: vec![batch],
                    got: vec![batch_b],
                    var_name: "MatMul: A.batch != B.batch".to_string(),
                }
                .into());
            }
            if k_a != k_b {
                return Err(LayerError::ShapeMismatch {
                    layer: LayerKind::MatMul,
                    expected: vec![k_a],
                    got: vec![k_b],
                    var_name: "MatMul: A.K != B.K".to_string(),
                }
                .into());
            }

            let mut slices: Vec<ArrayD<Variable>> = Vec::with_capacity(batch);
            for b in 0..batch {
                let a_slice = input_3d.slice(s![b, .., ..]).into_owned().into_dyn();
                let b_slice = weights_3d.slice(s![b, .., ..]).into_owned().into_dyn();
                slices.push(self.matmul_2d_core(api, a_slice, b_slice)?);
            }
            stack_batch_slices(batch, rows_a, cols_b, &slices)
        } else {
            let weights_2d =
                weights_dyn
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| LayerError::InvalidShape {
                        layer: LayerKind::MatMul,
                        msg: format!("Expected 2D weights for MatMul layer {}", self.name),
                    })?;
            let (k_b, cols_b) = weights_2d.dim();
            if k_a != k_b {
                return Err(LayerError::ShapeMismatch {
                    layer: LayerKind::MatMul,
                    expected: vec![k_a],
                    got: vec![k_b],
                    var_name: "MatMul: A.K != B.K".to_string(),
                }
                .into());
            }

            let b_dyn = weights_2d.into_dyn();
            let mut slices: Vec<ArrayD<Variable>> = Vec::with_capacity(batch);
            for b_idx in 0..batch {
                let a_slice = input_3d.slice(s![b_idx, .., ..]).into_owned().into_dyn();
                slices.push(self.matmul_2d_core(api, a_slice, b_dyn.clone())?);
            }
            stack_batch_slices(batch, rows_a, cols_b, &slices)
        }
    }
}

fn stack_batch_slices(
    batch: usize,
    rows: usize,
    cols: usize,
    slices: &[ArrayD<Variable>],
) -> Result<ArrayD<Variable>, CircuitError> {
    let mut flat: Vec<Variable> = Vec::with_capacity(batch * rows * cols);
    for slice in slices {
        flat.extend(slice.iter().copied());
    }
    ArrayD::from_shape_vec(IxDyn(&[batch, rows, cols]), flat).map_err(|_| {
        CircuitError::from(LayerError::InvalidShape {
            layer: LayerKind::MatMul,
            msg: "Failed to stack batched MatMul results".to_string(),
        })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use expander_compiler::frontend::{
        BasicAPI, CircuitField, EmptyHintCaller, GoldilocksConfig, Variable, extra::DebugBuilder,
    };
    use ndarray::{ArrayD, IxDyn};

    use crate::circuit_functions::{
        gadgets::LogupRangeCheckContext,
        layers::layer_ops::LayerOp,
        utils::{
            build_layers::{BuildLayerContext, default_n_bits_for_config},
            graph_pattern_matching::PatternRegistry,
            onnx_model::CircuitParams,
            onnx_types::ONNXLayer,
        },
    };

    use super::MatMulLayer;

    type TestBuilder = DebugBuilder<GoldilocksConfig, EmptyHintCaller>;

    fn test_params() -> CircuitParams {
        CircuitParams {
            scale_base: 2,
            scale_exponent: 0,
            rescale_config: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            freivalds_reps: 0,
            n_bits_config: HashMap::new(),
            weights_as_inputs: true,
            proof_system: crate::proof_system::ProofSystem::Expander,
            curve: None,
            logup_chunk_bits: None,
        }
    }

    fn matmul_layer(
        inputs: Vec<String>,
        shapes_map: &HashMap<String, Vec<usize>>,
    ) -> Result<
        Box<dyn LayerOp<GoldilocksConfig, TestBuilder>>,
        crate::circuit_functions::CircuitError,
    > {
        let layer = ONNXLayer {
            id: 0,
            name: "test_matmul".to_string(),
            op_type: "MatMul".to_string(),
            inputs,
            outputs: vec!["y".to_string()],
            shape: HashMap::new(),
            tensor: None,
            params: None,
            opset_version_number: 13,
        };
        let w_and_b_map: HashMap<String, &ONNXLayer> = HashMap::new();
        let n_bits_config = HashMap::new();
        let ctx = BuildLayerContext {
            w_and_b_map: &w_and_b_map,
            shapes_map,
            n_bits_config: &n_bits_config,
            default_n_bits: default_n_bits_for_config::<GoldilocksConfig>(),
            weights_as_inputs: true,
        };
        <MatMulLayer as LayerOp<GoldilocksConfig, TestBuilder>>::build(
            &layer,
            &test_params(),
            PatternRegistry::None,
            false,
            0,
            &ctx,
        )
    }

    fn tensor_vars(api: &mut TestBuilder, values: &[u32], shape: &[usize]) -> ArrayD<Variable> {
        let vars: Vec<Variable> = values
            .iter()
            .map(|&v| api.constant(CircuitField::<GoldilocksConfig>::from(v)))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), vars).expect("shape must be valid")
    }

    type F = CircuitField<GoldilocksConfig>;

    fn extract_values(api: &mut TestBuilder, arr: &ArrayD<Variable>) -> Vec<F> {
        arr.iter()
            .map(|&v| {
                api.constant_value(v)
                    .expect("debug builder always has constants")
            })
            .collect()
    }

    fn expected_fields(vals: &[u32]) -> Vec<F> {
        vals.iter().map(|&v| F::from(v)).collect()
    }

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
        assert!(freivalds_cheaper(100, 100, 100, 3));
    }

    #[test]
    fn freivalds_not_beneficial_for_tiny_matrices() {
        assert!(!freivalds_cheaper(2, 2, 2, 3));
    }

    #[test]
    fn freivalds_zero_reps_not_beneficial() {
        assert!(!freivalds_cheaper(1000, 1000, 1000, 0));
    }

    #[test]
    fn freivalds_single_row_not_beneficial() {
        assert!(!freivalds_cheaper(1, 1, 1, 1));
    }

    #[test]
    fn batched_matmul_broadcast_rhs() {
        // A: [2, 3, 4] × B: [4, 5] → [2, 3, 5]
        // B is broadcast across the batch dimension.
        let mut shapes_map = HashMap::new();
        shapes_map.insert("a".to_string(), vec![2, 3, 4]);
        shapes_map.insert("b".to_string(), vec![4, 5]);
        shapes_map.insert("y".to_string(), vec![2, 3, 5]);

        let built = matmul_layer(vec!["a".to_string(), "b".to_string()], &shapes_map)
            .expect("build should succeed for rank-3 × rank-2");

        let (mut api, _, _) = TestBuilder::new(vec![], vec![], EmptyHintCaller::new());

        // A[0] = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]  (3×4, first 3 rows of identity)
        // A[1] = [[0,0,0,1],[0,0,1,0],[0,1,0,0]]  (3×4, reversed selection)
        #[rustfmt::skip]
        let a_vals: Vec<u32> = vec![
            1,0,0,0, 0,1,0,0, 0,0,1,0,
            0,0,0,1, 0,0,1,0, 0,1,0,0,
        ];
        // B = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
        #[rustfmt::skip]
        let b_vals: Vec<u32> = vec![
            1,2,3,4,5,
            6,7,8,9,10,
            11,12,13,14,15,
            16,17,18,19,20,
        ];

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), tensor_vars(&mut api, &a_vals, &[2, 3, 4]));
        inputs.insert("b".to_string(), tensor_vars(&mut api, &b_vals, &[4, 5]));
        let mut logup_ctx = LogupRangeCheckContext::new_default();

        let (_, out) = built
            .apply(&mut api, &mut logup_ctx, &inputs)
            .expect("apply should succeed");

        assert_eq!(out.shape(), &[2, 3, 5]);

        // A[0] selects rows 0,1,2 of B; A[1] selects rows 3,2,1 of B.
        #[rustfmt::skip]
        let expected: &[u32] = &[
            1,2,3,4,5,       6,7,8,9,10,     11,12,13,14,15,
            16,17,18,19,20,   11,12,13,14,15,  6,7,8,9,10,
        ];
        assert_eq!(extract_values(&mut api, &out), expected_fields(expected));
    }

    #[test]
    fn batched_matmul_both_batched() {
        // A: [2, 3, 4] × B: [2, 4, 5] → [2, 3, 5]
        let mut shapes_map = HashMap::new();
        shapes_map.insert("a".to_string(), vec![2, 3, 4]);
        shapes_map.insert("b".to_string(), vec![2, 4, 5]);
        shapes_map.insert("y".to_string(), vec![2, 3, 5]);

        let built = matmul_layer(vec!["a".to_string(), "b".to_string()], &shapes_map)
            .expect("build should succeed for rank-3 × rank-3");

        let (mut api, _, _) = TestBuilder::new(vec![], vec![], EmptyHintCaller::new());

        // A[0] = identity-like 3×4, A[1] = all-ones 3×4
        #[rustfmt::skip]
        let a_vals: Vec<u32> = vec![
            1,0,0,0, 0,1,0,0, 0,0,1,0,
            1,1,1,1, 1,1,1,1, 1,1,1,1,
        ];
        // B[0] same as before, B[1] = all-twos 4×5
        #[rustfmt::skip]
        let b_vals: Vec<u32> = vec![
            1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20,
            2,2,2,2,2, 2,2,2,2,2,  2,2,2,2,2,      2,2,2,2,2,
        ];

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), tensor_vars(&mut api, &a_vals, &[2, 3, 4]));
        inputs.insert("b".to_string(), tensor_vars(&mut api, &b_vals, &[2, 4, 5]));
        let mut logup_ctx = LogupRangeCheckContext::new_default();

        let (_, out) = built
            .apply(&mut api, &mut logup_ctx, &inputs)
            .expect("apply should succeed");

        assert_eq!(out.shape(), &[2, 3, 5]);

        // Batch 0: identity-like × B[0] → rows 0,1,2 of B[0]
        // Batch 1: all-ones × all-twos → each element = 4*2 = 8
        #[rustfmt::skip]
        let expected: &[u32] = &[
            1,2,3,4,5,   6,7,8,9,10,   11,12,13,14,15,
            8,8,8,8,8,   8,8,8,8,8,    8,8,8,8,8,
        ];
        assert_eq!(extract_values(&mut api, &out), expected_fields(expected));
    }

    #[test]
    fn build_rejects_rank2_a_rank3_b() {
        let mut shapes_map = HashMap::new();
        shapes_map.insert("a".to_string(), vec![3, 4]);
        shapes_map.insert("b".to_string(), vec![2, 4, 5]);

        let result = matmul_layer(vec!["a".to_string(), "b".to_string()], &shapes_map);
        let err = result.err().expect("should fail for rank-2 A × rank-3 B");
        let msg = format!("{err}");
        assert!(
            msg.contains("rank-2 input A") && msg.contains("rank-3 input B"),
            "unexpected error: {msg}"
        );
    }
}
