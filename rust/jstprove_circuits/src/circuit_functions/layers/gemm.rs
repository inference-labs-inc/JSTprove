/// External crate imports
use ndarray::{Array2, ArrayD, Ix2, IxDyn};

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

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
