use ndarray::ArrayD;
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
    matrix_a: Array2<Variable>,
    matrix_b: Array2<Variable>,
) -> Array2<Variable> {
    let shape = matrix_a.dim();
    assert_eq!(shape, matrix_b.dim(), "Shape mismatch in matrix_addition");

    let mut result = Array2::default(shape);
    for ((i, j), out_elem) in result.indexed_iter_mut() {
        *out_elem = api.add(matrix_a[(i, j)], matrix_b[(i, j)]);
    }

    result
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
    matrix_a: Array2<Variable>,
    matrix_b: Array2<Variable>,
) -> Array2<Variable> {
    let (m, n) = matrix_a.dim();
    let (n2, p) = matrix_b.dim();
    assert_eq!(n, n2, "Inner dimensions must match for matrix multiplication");

    let mut result = Array2::default((m, p));

    for i in 0..m {
        for j in 0..p {
            let mut acc = api.constant(0);
            for k in 0..n {
                let mul = api.mul(matrix_a[(i, k)], matrix_b[(k, j)]);
                acc = api.add(acc, mul);
            }
            result[(i, j)] = acc;
        }
    }

    result
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