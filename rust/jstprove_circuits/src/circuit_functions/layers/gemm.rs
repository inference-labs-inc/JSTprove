use std::collections::HashMap;

/// External crate imports
use ndarray::{Array2, ArrayD, Ix2, IxDyn};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{ALPHA, BETA, INPUT, TRANS_A, TRANS_B},
        graph_pattern_matching::GraphPattern,
        onnx_model::{
            extract_params_and_expected_shape, get_input_name, get_param_or_default, get_w_or_b,
        },
        quantization::rescale_array,
        shaping::check_and_apply_transpose_array,
        tensor_ops::load_array_constants,
    },
};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct GemmLayer {
    name: String,
    index: usize,
    weights: ArrayD<i64>,
    bias: ArrayD<i64>,
    is_rescale: bool,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: u64,
    optimization_pattern: GraphPattern,
    scaling: u64,
    input_shape: Vec<usize>,
    alpha: f32,
    beta: f32,
    transa: usize,
    transb: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GemmLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let is_relu = matches!(self.optimization_pattern.name, "Gemm+Relu");

        let input_name = get_input_name(&self.inputs, 0, LayerKind::Gemm, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Gemm,
                name: input_name.clone(),
            })?
            .clone();

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

        // Sanity check alpha and beta
        check_alpha_beta(self.alpha, ALPHA, LayerKind::Gemm, &self.name)?;
        check_alpha_beta(self.beta, BETA, LayerKind::Gemm, &self.name)?;

        // Matrix multiplication and bias addition
        let mut result =
            matrix_multiplication(api, input_array.into_dyn(), weights_array.into_dyn())?;
        result = matrix_addition(api, &result, bias_array)?;

        let mut out_array = result.into_dyn(); // back to ArrayD<Variable>
        if self.is_rescale {
            let k = usize::try_from(self.scaling).map_err(|_| LayerError::Other {
                layer: LayerKind::Gemm,
                msg: "Cannot convert scaling to usize".to_string(),
            })?;
            let s = self.v_plus_one.checked_sub(1).ok_or_else(|| {
                LayerError::InvalidParameterValue {
                    layer: LayerKind::Gemm,
                    layer_name: self.name.clone(),
                    param_name: "v_plus_one".to_string(),
                    value: self.v_plus_one.to_string(),
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
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
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
            v_plus_one: layer_context.n_bits,
            two_v: layer_context.two_v,
            alpha_two_v: layer_context.alpha_two_v,
            optimization_pattern,
            scaling: circuit_params.scale_exponent.into(), // TODO: Becomes scaling_in?
            input_shape: expected_shape.clone(),
            alpha: get_param_or_default(&layer.name, ALPHA, &params, Some(&1.0))?,
            beta: get_param_or_default(&layer.name, BETA, &params, Some(&1.0))?,
            transa: get_param_or_default(&layer.name, TRANS_A, &params, Some(&0))?,
            transb: get_param_or_default(&layer.name, TRANS_B, &params, Some(&0))?,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };
        Ok(Box::new(gemm))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: dot
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the dot product of two 1D `Vec<Variable>` vectors using circuit constraints.
///
/// # Arguments
/// - `api`: The circuit builder.
/// - `vector_a`: First input vector.
/// - `vector_b`: Second input vector (must have the same length).
///
/// # Returns
/// A `Variable` representing the scalar dot product result:
/// `sum_i` (\\(`a_i` \\cdot `b_i`\\))
///
/// # Error
/// Raises Error if the vectors are of unequal length.
///
/// # Example
/// ```ignore
/// let dot_result = dot(api, vec![a1, a2], vec![b1, b2]);
/// ```
pub fn dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    vector_a: &ArrayD<&Variable>,
    vector_b: &ArrayD<&Variable>,
) -> Variable {
    let mut row_col_product: Variable = api.constant(0);
    for k in 0..vector_a.len() {
        let element_product = api.mul(vector_a[k], vector_b[k]);
        row_col_product = api.add(row_col_product, element_product);
    }
    row_col_product
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: matrix_addition
// ─────────────────────────────────────────────────────────────────────────────

/// Adds two `ArrayD<Variable>` tensors elementwise using circuit constraints.
///
/// If the shapes differ but the total number of elements matches, this function
/// attempts to reshape `matrix_b` to match `matrix_a`. This is useful for adding
/// broadcasted constants (e.g., bias terms) with higher-dimensional arrays.
///
/// # Arguments
/// - `api`: The circuit builder.
/// - `matrix_a`: First input tensor.
/// - `matrix_b`: Second input tensor, possibly with a different shape.
///
/// # Returns
/// An `ArrayD<Variable>` of the same shape as `matrix_a`, representing the elementwise sum.
///
/// # Errors
/// - If the total number of elements in `matrix_a` and `matrix_b` do not match.
/// - If reshaping `matrix_b` to `matrix_a`'s shape fails.
///
/// # Example
/// ```ignore
/// let result = matrix_addition(api, input_tensor, bias_tensor);
/// ```
pub fn matrix_addition<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    mut matrix_b: ArrayD<Variable>,
) -> Result<ArrayD<Variable>, LayerError> {
    let shape_a = matrix_a.shape().to_vec();

    // Attempt to reshape if shape differs but total elements match
    if matrix_b.shape() != shape_a {
        if matrix_b.len() == matrix_a.len() {
            matrix_b = matrix_b
                .clone()
                .into_shape_with_order(IxDyn(&shape_a))
                .map_err(|_| LayerError::ShapeMismatch {
                    layer: LayerKind::Gemm,
                    expected: shape_a.clone(),
                    got: matrix_b.shape().to_vec(),
                    var_name: "matrix_b".to_string(),
                })?;
        } else {
            return Err(LayerError::ShapeMismatch {
                layer: LayerKind::Gemm,
                expected: shape_a,
                got: matrix_b.shape().to_vec(),
                var_name: "matrix_b".to_string(),
            });
        }
    }

    let result = matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(&a, &b)| api.add(a, b))
        .collect::<Vec<_>>();

    ArrayD::from_shape_vec(IxDyn(&shape_a), result).map_err(|_| LayerError::InvalidShape {
        layer: LayerKind::Gemm,
        msg: "Failed to build result array after matrix_addition".to_string(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: matrix_multiplication
// ─────────────────────────────────────────────────────────────────────────────

/// Performs 2D matrix multiplication using circuit constraints.
///
/// The input tensors must be 2-dimensional. This function computes
/// the standard matrix product of `matrix_a` (shape \\( m \times n \\))
/// and `matrix_b` (shape \\( n \times p \\)), resulting in a tensor of shape \\( m \times p \\).
///
/// # Arguments
/// - `api`: The circuit builder.
/// - `matrix_a`: Left-hand matrix (must be 2D).
/// - `matrix_b`: Right-hand matrix (must be 2D).
///
/// # Returns
/// An `ArrayD<Variable>` (2D) representing the result of matrix multiplication.
///
/// # Errors
/// - [`LayerError::InvalidShape`] if `matrix_a` or `matrix_b` is not 2-dimensional.
/// - [`LayerError::ShapeMismatch`] if the inner dimensions of the matrices do not match
///   (i.e., `matrix_a` columns != `matrix_b` rows).
///
/// # Example
/// ```ignore
/// let product = matrix_multiplication(api, weights, input);
/// ```
pub fn matrix_multiplication<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: ArrayD<Variable>,
    matrix_b: ArrayD<Variable>,
) -> Result<ArrayD<Variable>, LayerError> {
    let a = matrix_a
        .into_dimensionality::<Ix2>()
        .map_err(|_| LayerError::InvalidShape {
            layer: LayerKind::Gemm,
            msg: "matrix_a must be 2D".to_string(),
        })?;
    let b = matrix_b
        .into_dimensionality::<Ix2>()
        .map_err(|_| LayerError::InvalidShape {
            layer: LayerKind::Gemm,
            msg: "matrix_b must be 2D".to_string(),
        })?;

    let (dim_m, dim_n) = a.dim();
    let (dim_n2, dim_p) = b.dim();
    if dim_n != dim_n2 {
        return Err(LayerError::ShapeMismatch {
            layer: LayerKind::Gemm,
            expected: vec![dim_n],
            got: vec![dim_n2],
            var_name: "a_dim[1] != b_dim[0]".to_string(),
        });
    }

    let mut result = Array2::default((dim_m, dim_p));

    for i in 0..dim_m {
        for j in 0..dim_p {
            let mut acc = api.constant(0);
            for k in 0..dim_n {
                let mul = api.mul(a[(i, k)], b[(k, j)]);
                acc = api.add(acc, mul);
            }
            result[(i, j)] = acc;
        }
    }

    Ok(result.into_dyn())
}

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
