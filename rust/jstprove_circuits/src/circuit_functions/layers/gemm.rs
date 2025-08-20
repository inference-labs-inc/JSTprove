use std::collections::HashMap;

/// External crate imports
use ndarray::{Array2, ArrayD, Ix2, IxDyn};

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::{constants::{ALPHA, BETA, GEMM, TRANS_A, TRANS_B}, graph_pattern_matching::GraphPattern, onnx_model::{extract_params_and_expected_shape, get_param_or_default, get_w_or_b}, quantization::rescale_array, shaping::check_and_apply_transpose_array, tensor_ops::load_array_constants}};

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
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let is_relu = match self.optimization_pattern.name{
                    "Gemm+Relu" => true,
                    _ => false
                };

        let layer_input = input.get(&self.inputs[0])
        .ok_or_else(|| panic!("Missing input {}", self.inputs[0].clone())).unwrap()
    .clone();
        let mut input_array = layer_input
            .into_dimensionality::<Ix2>()
            .map_err(|_| format!("Expected 2D input for layer {}", self.name))?;
        let mut weights_array = load_array_constants(api, &self.weights)
        .into_dimensionality::<Ix2>()
            .map_err(|_| format!("Expected 2D input for layer {}", self.name))?;

        input_array = check_and_apply_transpose_array(input_array, self.transa, TRANS_A, GEMM, &self.name);
        weights_array = check_and_apply_transpose_array(weights_array, self.transb, TRANS_B, GEMM, &self.name);

        let bias_array = load_array_constants(api, &self.bias);

        // Sanity check alpha and beta
        check_alpha_beta(self.alpha, ALPHA, GEMM, &self.name);
        check_alpha_beta(self.beta, BETA, GEMM, &self.name);

        // Matrix multiplication and bias addition
        let mut result = matrix_multiplication(api, input_array.into_dyn(), weights_array.into_dyn());
        result = matrix_addition(api, result, bias_array);

        api.display("3", result[[0, 0]]);

        let mut out_array = result.into_dyn(); // back to ArrayD<Variable>
        if self.is_rescale {
            let k = self.scaling as usize;
            let s = self.v_plus_one.checked_sub(1).expect("v_plus_one must be at least 1");
            out_array = rescale_array(api, out_array, k, s, is_relu);
        }

        Ok((self.outputs.clone(), out_array))
    }
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {
        
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer);
        let gemm = Self {
            name: layer.name.clone(),
            index: index,
            weights: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[1]),
            bias: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[2]),
            is_rescale: is_rescale.clone(),
            v_plus_one: layer_context.n_bits,
            two_v: layer_context.two_v,
            alpha_two_v: layer_context.alpha_two_v,
            optimization_pattern: optimization_pattern,
            scaling: circuit_params.scale_exponent.into(), // TODO: Becomes scaling_in?
            input_shape: expected_shape.to_vec(),
            alpha: get_param_or_default(&layer.name, ALPHA, &params, Some(&1.0)),
            beta: get_param_or_default(&layer.name, BETA, &params, Some(&1.0)),
            transa: get_param_or_default(&layer.name, TRANS_A, &params, Some(&0)),
            transb: get_param_or_default(&layer.name, TRANS_B, &params, Some(&0)),
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
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
/// \\( \sum_i a_i \cdot b_i \\)
///
/// # Panics
/// Panics if the vectors are of unequal length.
///
/// # Example
/// ```ignore
/// let dot_result = dot(api, vec![a1, a2], vec![b1, b2]);
/// ```

pub fn dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    vector_a: ArrayD<&Variable>,
    vector_b: ArrayD<&Variable>,
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
/// # Panics
/// - If the total number of elements in `matrix_a` and `matrix_b` do not match.
/// - If reshaping `matrix_b` to `matrix_a`'s shape fails.
///
/// # Example
/// ```ignore
/// let result = matrix_addition(api, input_tensor, bias_tensor);
/// ```
pub fn matrix_addition<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: ArrayD<Variable>,
    mut matrix_b: ArrayD<Variable>,
) -> ArrayD<Variable> {
    let shape_a = matrix_a.shape().to_vec();

    // Attempt to reshape if shape differs but total elements match
    if matrix_b.shape() != shape_a {
        if matrix_b.len() == matrix_a.len() {
            matrix_b = matrix_b
                .into_shape_with_order(IxDyn(&shape_a))
                .expect("Reshape failed: bias shape is not compatible with input shape");
        } else {
            panic!(
                "Shape mismatch in matrix_addition: matrix_a shape = {:?}, matrix_b shape = {:?}",
                shape_a,
                matrix_b.shape()
            );
        }
    }

    let result = matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(&a, &b)| api.add(a, b))
        .collect::<Vec<_>>();

    ArrayD::from_shape_vec(IxDyn(&shape_a), result)
        .expect("Failed to build result array after matrix_addition")
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
/// # Panics
/// - If either input tensor is not 2D.
/// - If the inner dimensions do not match: `matrix_a.shape()[1] != matrix_b.shape()[0]`.
///
/// # Example
/// ```ignore
/// let product = matrix_multiplication(api, weights, input);
/// ```
pub fn matrix_multiplication<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: ArrayD<Variable>,
    matrix_b: ArrayD<Variable>,
) -> ArrayD<Variable> {
    let a = matrix_a
        .into_dimensionality::<Ix2>()
        .expect("matrix_multiplication: matrix_a must be 2D");
    let b = matrix_b
        .into_dimensionality::<Ix2>()
        .expect("matrix_multiplication: matrix_b must be 2D");

    let (m, n) = a.dim();
    let (n2, p) = b.dim();
    assert_eq!(
        n, n2,
        "Inner dimensions must match for matrix multiplication"
    );

    let mut result = Array2::default((m, p));

    for i in 0..m {
        for j in 0..p {
            let mut acc = api.constant(0);
            for k in 0..n {
                let mul = api.mul(a[(i, k)], b[(k, j)]);
                acc = api.add(acc, mul);
            }
            result[(i, j)] = acc;
        }
    }

    result.into_dyn()
}

fn check_alpha_beta(val: f32, var_name: &str, layer_type: &str, layer_name: &str) {
    if val != 1.0{
        panic!("Only {} = 1 is currently supported for {} layers: {}", var_name, layer_type, layer_name);
    }
}
