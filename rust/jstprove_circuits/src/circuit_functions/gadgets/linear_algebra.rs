//! Linear algebra gadgets used by circuit layers.
//!
//! This module provides reusable tensor / matrix / vector operations over
//! `expander_compiler::frontend::Variable` values.
//!
//! Two kinds of functionality appear here:
//! - **Constrained gadgets** (using `api.mul`, `api.add`, `api.assert_is_equal`, ...),
//!   which generate constraints.
//! - **Unconstrained gadgets** (using `api.unconstrained_*`), which compute witness
//!   values without adding constraints and therefore must be linked to the circuit
//!   by a separate constrained check.
//!
//! In particular, this module includes a Freivalds-based verifier for matrix products,
//! allowing probabilistic checking of `A * B = C` with substantially fewer constraints
//! than fully constraining the matrix multiplication in many regimes.
//!
//! For convenience these functions report failures via `LayerError` / `LayerKind`.

/// External crate imports
use ndarray::{Array2, ArrayD, Ix2, IxDyn};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::layers::{LayerError, LayerKind};

// -----------------------------------------------------------------------------
// FUNCTION: dot
// -----------------------------------------------------------------------------
//
// Computes the dot product of two 1D ArrayD<&Variable> views using circuit
// constraints.
//
// Arguments:
//   - api: circuit builder
//   - vector_a: first input vector
//   - vector_b: second input vector (must have the same shape)
//
// Returns:
//   - Ok(Variable) holding the dot product
//   - Err(LayerError) if shapes are incompatible or not 1D
//
// The implementation is a simple sum over k of (a[k] * b[k]).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// FUNCTION: matrix_addition
// -----------------------------------------------------------------------------
//
// Elementwise addition of two ArrayD<Variable> tensors using circuit
// constraints.
//
// If shapes differ but total element counts match, matrix_b is reshaped
// to match matrix_a. This is useful for adding broadcast-style constants
// (for example, bias terms) with higher dimensional arrays.
//
// Arguments:
//   - api: circuit builder
//   - matrix_a: first tensor
//   - matrix_b: second tensor, possibly with a different shape
//   - layer_type: identifier for error reporting
//
// Returns:
//   - Ok(ArrayD<Variable>) with the same shape as matrix_a
//   - Err(LayerError) if reshape is impossible or shapes are incompatible
// -----------------------------------------------------------------------------
pub fn matrix_addition<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    matrix_b: ArrayD<Variable>,
    layer_type: LayerKind,
) -> Result<ArrayD<Variable>, LayerError> {
    elementwise_op(
        api,
        matrix_a,
        matrix_b,
        layer_type,
        expander_compiler::frontend::BasicAPI::add,
        "matrix_addition",
    )
}

// -----------------------------------------------------------------------------
// FUNCTION: matrix_hadamard_product
// -----------------------------------------------------------------------------
//
// Elementwise multiplication of two ArrayD<Variable> tensors using circuit
// constraints.
//
// If shapes differ but total element counts match, matrix_b is reshaped
// to match matrix_a. This is useful for multiplying broadcast-style
// constants with higher dimensional arrays.
//
// Arguments:
//   - api: circuit builder
//   - matrix_a: first tensor
//   - matrix_b: second tensor, possibly with a different shape
//   - layer_type: identifier for error reporting
//
// Returns:
//   - Ok(ArrayD<Variable>) with the same shape as matrix_a
//   - Err(LayerError) if reshape is impossible or shapes are incompatible
// -----------------------------------------------------------------------------
pub fn matrix_hadamard_product<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    matrix_b: ArrayD<Variable>,
    layer_type: LayerKind,
) -> Result<ArrayD<Variable>, LayerError> {
    elementwise_op(
        api,
        matrix_a,
        matrix_b,
        layer_type,
        expander_compiler::frontend::BasicAPI::mul,
        "matrix_hadamard_product",
    )
}

// -----------------------------------------------------------------------------
// FUNCTION: matrix_subtraction
// -----------------------------------------------------------------------------
//
// Elementwise subtraction of two ArrayD<Variable> tensors using circuit
// constraints.
//
// If shapes differ but total element counts match, matrix_b is reshaped
// to match matrix_a. This is useful for subtracting broadcast-style
// constants from higher dimensional arrays.
//
// Arguments:
//   - api: circuit builder
//   - matrix_a: first tensor
//   - matrix_b: second tensor, possibly with a different shape
//   - layer_type: identifier for error reporting
//
// Returns:
//   - Ok(ArrayD<Variable>) with the same shape as matrix_a
//   - Err(LayerError) if reshape is impossible or shapes are incompatible
// -----------------------------------------------------------------------------
pub fn matrix_subtraction<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    matrix_b: ArrayD<Variable>,
    layer_type: LayerKind,
) -> Result<ArrayD<Variable>, LayerError> {
    elementwise_op(
        api,
        matrix_a,
        matrix_b,
        layer_type,
        expander_compiler::frontend::BasicAPI::sub,
        "matrix_subtraction",
    )
}

// -----------------------------------------------------------------------------
// FUNCTION: elementwise_op (internal helper)
// -----------------------------------------------------------------------------
//
// Internal helper for elementwise tensor operations (add, mul, sub, etc.).
//
// Shape rules:
//   - If shapes are identical, no reshape occurs.
//   - If shapes differ but total element counts match, matrix_b is reshaped
//     to match matrix_a.
//   - Otherwise, LayerError::ShapeMismatch is returned.
//
// Arguments:
//   - api: circuit builder
//   - matrix_a: left operand; its shape defines the output shape
//   - matrix_b: right operand, possibly reshaped
//   - layer_type: identifier for error reporting
//   - op: function implementing the elementwise operation
//   - op_name: name for debug and error messages
//
// Returns:
//   - Ok(ArrayD<Variable>) with the same shape as matrix_a
//   - Err(LayerError) if reshape is impossible or array construction fails
// -----------------------------------------------------------------------------
fn elementwise_op<C: Config, Builder: RootAPI<C>, F>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    mut matrix_b: ArrayD<Variable>,
    layer_type: LayerKind,
    op: F,
    op_name: &'static str,
) -> Result<ArrayD<Variable>, LayerError>
where
    F: Fn(&mut Builder, Variable, Variable) -> Variable,
{
    let shape_a = matrix_a.shape().to_vec();
    let shape_b = matrix_b.shape().to_vec();

    // Shared shape match / reshape logic
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
                expected: shape_a.clone(),
                got: shape_b,
                var_name: "matrix_b".to_string(),
            });
        }
    }

    // Elementwise operation
    let result = matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(&a, &b)| op(api, a, b))
        .collect::<Vec<_>>();

    ArrayD::from_shape_vec(IxDyn(&shape_a), result).map_err(|_| LayerError::InvalidShape {
        layer: layer_type,
        msg: format!("Failed to build result array after {op_name}"),
    })
}

// -----------------------------------------------------------------------------
// FUNCTION: matrix_multiplication
// -----------------------------------------------------------------------------
//
// Performs 2D matrix multiplication using circuit constraints.
//
// The input tensors must be 2-dimensional. This function computes the
// standard matrix product of matrix_a with shape (m, n) and matrix_b
// with shape (n, p), yielding an ArrayD<Variable> with shape (m, p).
//
// Arguments:
//   - api: circuit builder
//   - matrix_a: left matrix, must be 2D
//   - matrix_b: right matrix, must be 2D
//   - layer_type: identifier for error reporting
//
// Returns:
//   - Ok(ArrayD<Variable>) representing the product
//   - Err(LayerError::InvalidShape) if either input is not 2D
//   - Err(LayerError::ShapeMismatch) if inner dimensions do not match
// -----------------------------------------------------------------------------
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

// ............................................................................
// FUNCTION: unconstrained_matrix_multiplication
// ............................................................................

/// Computes a 2D matrix product using **unconstrained** arithmetic.
///
/// Given `A` of shape `(m, n)` and `B` of shape `(n, p)`, returns `C = A * B`
/// with shape `(m, p)`.
///
/// Security / correctness note:
/// This uses `unconstrained_mul` / `unconstrained_add` and therefore **adds no
/// constraints** relating `C` to `A` and `B`. The returned `C` is only a witness
/// suggestion. It must be linked back to `A` and `B` via a constrained relation,
/// e.g. `freivalds_verify_matrix_product`, or by fully constraining the matmul.
///
/// Returns `LayerError` on shape mismatch or if inputs are not 2D.
pub fn unconstrained_matrix_multiplication<C: Config, Builder: RootAPI<C>>(
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
            let mut acc = api.unconstrained_identity(0);
            for k in 0..dim_n {
                let mul = api.unconstrained_mul(a[(i, k)], b[(k, j)]);
                acc = api.unconstrained_add(acc, mul);
            }
            result[(i, j)] = acc;
        }
    }

    Ok(result.into_dyn())
}

// -----------------------------------------------------------------------------
// FUNCTION: freivalds_verify_matrix_product
// -----------------------------------------------------------------------------

/// Probabilistically verifies a matrix product `A * B = C` using Freivalds' algorithm.
///
/// All matrices are over the circuit field `F`, represented as `Variable`s.
/// Expected shapes:
/// - `A`: `(ell, m)`
/// - `B`: `(m, n)`
/// - `C`: `(ell, n)`
///
/// For each repetition:
/// 1) Sample a challenge vector `x ∈ F^n`.
/// 2) Compute `v = Bx ∈ F^m`.
/// 3) Compute `w = Cx ∈ F^ell`.
/// 4) Compute `u = Av ∈ F^ell`.
/// 5) Constrain `u == w` entrywise.
///
/// ## Soundness (standard Freivalds guarantee)
/// If `A * B != C` and `x` is sampled uniformly at random from `F^n`, then for a
/// single repetition:
///
///     Pr[u = w] ≤ 1 / |F|.
///
/// With `k = num_repetitions` independent repetitions, the soundness error is at most
///
///     (1 / |F|)^k.
///
/// **Important:** This guarantee relies on how `x` is generated. The challenges must be
/// prover-unbiased (i.e., not chosen adversarially after seeing `A, B, C`) and should be
/// independent across repetitions.
///
/// ## Constraint costs
/// This function constrains only the matrix-vector products and the final equality checks.
/// In typical usage, `C` is computed cheaply (e.g., via unconstrained arithmetic) and then
/// linked back to `A` and `B` via this verifier, avoiding the cost of fully constraining
/// the `ell × m × n` matmul.
///
/// Returns `LayerError` on invalid dimensions or shape mismatch, and rejects
/// `num_repetitions == 0` to avoid accidentally disabling verification.
pub fn freivalds_verify_matrix_product<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    matrix_b: &ArrayD<Variable>,
    matrix_c: &ArrayD<Variable>,
    layer_type: LayerKind,
    num_repetitions: usize,
) -> Result<(), LayerError> {
    // Convert to 2D matrices (Ix2) and check dimensions.
    let a: Array2<Variable> =
        matrix_a
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_type.clone(),
                msg: "Freivalds: matrix_a must be 2D".to_string(),
            })?;

    let b: Array2<Variable> =
        matrix_b
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_type.clone(),
                msg: "Freivalds: matrix_b must be 2D".to_string(),
            })?;

    let c: Array2<Variable> =
        matrix_c
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_type.clone(),
                msg: "Freivalds: matrix_c must be 2D".to_string(),
            })?;

    // Dimensions: A is (ell, m), B is (m, n), C is (ell, n)
    let (ell, m) = a.dim();
    let (m2, n) = b.dim();
    let (ell2, n2) = c.dim();

    if m != m2 {
        return Err(LayerError::ShapeMismatch {
            layer: layer_type.clone(),
            expected: vec![m],
            got: vec![m2],
            var_name: "Freivalds: A.cols != B.rows".to_string(),
        });
    }

    if ell != ell2 || n != n2 {
        return Err(LayerError::ShapeMismatch {
            layer: layer_type.clone(),
            expected: vec![ell, n],
            got: vec![ell2, n2],
            var_name: "Freivalds: C shape mismatch".to_string(),
        });
    }

    if num_repetitions == 0 {
        // Degenerate case: nothing to prove; treat as error to avoid
        // accidentally disabling the check.
        return Err(LayerError::InvalidShape {
            layer: layer_type,
            msg: "Freivalds: num_repetitions must be >= 1".to_string(),
        });
    }

    for _rep in 0..num_repetitions {
        // 1) Sample random x in F^n
        let mut x: Vec<Variable> = Vec::with_capacity(n);
        for _j in 0..n {
            let r = api.get_random_value();
            x.push(r);
        }

        // 2) v = B x, length m
        let mut v: Vec<Variable> = Vec::with_capacity(m);
        for i in 0..m {
            let mut acc = api.constant(0);
            for j in 0..n {
                let mul = api.mul(b[(i, j)], x[j]);
                acc = api.add(acc, mul);
            }
            v.push(acc);
        }

        // 3) w = C x, length ell
        let mut w: Vec<Variable> = Vec::with_capacity(ell);
        for i in 0..ell {
            let mut acc = api.constant(0);
            for j in 0..n {
                let mul = api.mul(c[(i, j)], x[j]);
                acc = api.add(acc, mul);
            }
            w.push(acc);
        }

        // 4) u = A v, length ell
        let mut u: Vec<Variable> = Vec::with_capacity(ell);
        for i in 0..ell {
            let mut acc = api.constant(0);
            for k in 0..m {
                let mul = api.mul(a[(i, k)], v[k]);
                acc = api.add(acc, mul);
            }
            u.push(acc);
        }

        // 5) Enforce u == w entrywise
        for i in 0..ell {
            api.assert_is_equal(u[i], w[i]);
        }
    }

    Ok(())
}
