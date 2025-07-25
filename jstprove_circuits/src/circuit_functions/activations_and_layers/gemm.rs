use ndarray::{ArrayD, Array2, Ix2, IxDyn};
use expander_compiler::frontend::*;

pub fn dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    vector_a: Vec<Variable>,
    vector_b: Vec<Variable>,
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

/// Adds two `Array2<Variable>` matrices elementwise using circuit constraints.
///
/// # Arguments
/// - `api`: The circuit builder.
/// - `matrix_a`: First matrix.
/// - `matrix_b`: Second matrix.
///
/// # Returns
/// An `Array2<Variable>` representing the elementwise sum of the two input matrices.
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
                .into_shape(IxDyn(&shape_a))
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

/// Multiplies two `Array2<Variable>` matrices using naïve nested loops and circuit constraints.
///
/// # Arguments
/// - `api`: The circuit builder.
/// - `matrix_a`: Matrix of shape (m × n).
/// - `matrix_b`: Matrix of shape (n × p).
///
/// # Returns
/// A matrix of shape (m × p) representing the matrix product.
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
    assert_eq!(n, n2, "Inner dimensions must match for matrix multiplication");

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