// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: dot
// ─────────────────────────────────────────────────────────────────────────────

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::{Array2, ArrayD, Ix2, IxDyn};

use crate::circuit_functions::layers::{LayerError, LayerKind};

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
#[allow(dead_code)]
pub fn dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    vector_a: &ArrayD<&Variable>,
    vector_b: &ArrayD<&Variable>,
    layer_type: LayerKind,
) -> Result<Variable, LayerError> {
    if vector_a.shape() != vector_b.shape() {
        return Err(LayerError::InvalidShape {
            layer: layer_type,
            msg: format!(
                "Dot product requires two vectors of the same length. Got {:?}, {:?}",
                vector_a.shape(),
                vector_b.shape()
            ),
        });
    }
    if vector_a.ndim() != 1 {
        return Err(LayerError::InvalidShape {
            layer: layer_type,
            msg: format!("Dot product requires 1D vectors. Got {}", vector_a.ndim()),
        });
    }

    let mut row_col_product: Variable = api.constant(0);
    for k in 0..vector_a.len() {
        let element_product = api.mul(vector_a[k], vector_b[k]);
        row_col_product = api.add(row_col_product, element_product);
    }
    Ok(row_col_product)
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
    layer_type: LayerKind,
) -> Result<ArrayD<Variable>, LayerError> {
    let shape_a = matrix_a.shape().to_vec();
    let shape_b = matrix_b.shape().to_vec();

    // Attempt to reshape if shape differs but total elements match
    if shape_b != shape_a {
        if matrix_b.len() == matrix_a.len() {
            matrix_b = matrix_b
                .into_shape_with_order(IxDyn(&shape_a))
                .map_err(|_| LayerError::ShapeMismatch {
                    layer: layer_type.clone(),
                    expected: shape_a.clone(),
                    got: shape_b,
                    var_name: "matrix_b".to_string(),
                })?;
        } else {
            return Err(LayerError::ShapeMismatch {
                layer: layer_type.clone(),
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
        layer: layer_type,
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
    layer_type: LayerKind,
) -> Result<ArrayD<Variable>, LayerError> {
    let a = matrix_a
        .into_dimensionality::<Ix2>()
        .map_err(|_| LayerError::InvalidShape {
            layer: layer_type.clone(),
            msg: "matrix_a must be 2D".to_string(),
        })?;
    let b = matrix_b
        .into_dimensionality::<Ix2>()
        .map_err(|_| LayerError::InvalidShape {
            layer: layer_type.clone(),
            msg: "matrix_b must be 2D".to_string(),
        })?;

    let (dim_m, dim_n) = a.dim();
    let (dim_n2, dim_p) = b.dim();
    if dim_n != dim_n2 {
        return Err(LayerError::ShapeMismatch {
            layer: layer_type,
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
