use ndarray::{ArrayD, Array2, Ix2};
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
    matrix_b: ArrayD<Variable>,
) -> ArrayD<Variable> {
    let a = matrix_a
        .into_dimensionality::<Ix2>()
        .expect("matrix_addition: matrix_a must be 2D");
    let b = matrix_b
        .into_dimensionality::<Ix2>()
        .expect("matrix_addition: matrix_b must be 2D");

    let shape = a.dim();
    assert_eq!(shape, b.dim(), "Shape mismatch in matrix_addition");

    let mut result = Array2::default(shape);
    for ((i, j), out_elem) in result.indexed_iter_mut() {
        *out_elem = api.add(a[(i, j)], b[(i, j)]);
    }

    result.into_dyn()
}


/// Naive Matrix addition with vectors
pub fn matrix_addition_vec<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new(); // or [[Variable::default(); N]; M]
    for (i, row_a) in matrix_a.iter().enumerate() {
        let mut row_out: Vec<Variable> = Vec::new();
        for (j, value) in row_a.iter().enumerate() {
            row_out.push(api.add(value, matrix_b[i][j]));
        }
        out.push(row_out);
    }
    out
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


/// Matrix multiplciation of vector of vectors naive version 2
pub fn matrix_multplication_naive2<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut out_row: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out_row.push(row_col_product);
        }
        out.push(out_row);
    }
    out
}