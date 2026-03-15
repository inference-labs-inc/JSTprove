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
use ndarray::{Array2, ArrayD, Ix2, IxDyn, s};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::layers::{LayerError, LayerKind};

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
/// # Errors
/// Returns `LayerError::ShapeMismatch` if the tensors are not the same shape and cannot be reshaped
/// (by total element count) to match. Returns `LayerError::InvalidShape` if the output array
/// cannot be constructed.
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
/// # Errors
/// Returns `LayerError::ShapeMismatch` if the tensors are not the same shape and cannot be reshaped
/// (by total element count) to match. Returns `LayerError::InvalidShape` if the output array
/// cannot be constructed.
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
/// # Errors
/// Returns `LayerError::ShapeMismatch` if the tensors are not the same shape and cannot be reshaped
/// (by total element count) to match. Returns `LayerError::InvalidShape` if the output array
/// cannot be constructed.
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
/// # Errors
/// Returns `LayerError::InvalidShape` if either input is not 2D.
/// Returns `LayerError::ShapeMismatch` if A.cols != B.rows.
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
/// # Errors
/// Returns `LayerError` if the input shapes are incompatible or cannot be interpreted
/// as the required dimensionality for the operation.
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
            let mut acc = api.constant(0);
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
/// single repetition: `Pr[u = w] ≤ 1 / |F|`.
///
/// With `k = num_repetitions` independent repetitions, the soundness error is at most
/// `(1 / |F|)^k`.
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
/// # Errors
/// Returns `LayerError::InvalidShape` if any matrix is not 2D or if `num_repetitions == 0`.
/// Returns `LayerError::ShapeMismatch` if the dimensions do not satisfy A:(ell,m), B:(m,n), C:(ell,n).
pub fn freivalds_verify_matrix_product<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: &ArrayD<Variable>,
    matrix_b: &ArrayD<Variable>,
    matrix_c: &ArrayD<Variable>,
    layer_type: LayerKind,
    num_repetitions: usize,
) -> Result<(), LayerError> {
    // Convert to 2D matrices (Ix2) and check dimensions.
    let mat_a: Array2<Variable> =
        matrix_a
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_type.clone(),
                msg: "Freivalds: matrix_a must be 2D".to_string(),
            })?;

    let mat_b: Array2<Variable> =
        matrix_b
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_type.clone(),
                msg: "Freivalds: matrix_b must be 2D".to_string(),
            })?;

    let mat_c: Array2<Variable> =
        matrix_c
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: layer_type.clone(),
                msg: "Freivalds: matrix_c must be 2D".to_string(),
            })?;

    // Dimensions: A is (ell, m), B is (m, n), C is (ell, n)
    let (num_rows_a, num_cols_a) = mat_a.dim();
    let (num_rows_b, num_cols_b) = mat_b.dim();
    let (num_rows_c, num_cols_c) = mat_c.dim();

    if num_cols_a != num_rows_b {
        return Err(LayerError::ShapeMismatch {
            layer: layer_type.clone(),
            expected: vec![num_cols_a],
            got: vec![num_rows_b],
            var_name: "Freivalds: A.cols != B.rows".to_string(),
        });
    }

    if num_rows_a != num_rows_c || num_cols_b != num_cols_c {
        return Err(LayerError::ShapeMismatch {
            layer: layer_type.clone(),
            expected: vec![num_rows_a, num_cols_b],
            got: vec![num_rows_c, num_cols_c],
            var_name: "Freivalds: C shape mismatch".to_string(),
        });
    }

    if num_repetitions == 0 {
        return Err(LayerError::InvalidShape {
            layer: layer_type,
            msg: "Freivalds: num_repetitions must be >= 1".to_string(),
        });
    }

    for _rep in 0..num_repetitions {
        // 1) Sample random vector in F^n
        let mut rand_vec: Vec<Variable> = Vec::with_capacity(num_cols_b);
        for _idx in 0..num_cols_b {
            let rand_entry = api.get_random_value();
            rand_vec.push(rand_entry);
        }

        // 2) b_times_rand = B * rand_vec, length m
        let mut b_times_rand: Vec<Variable> = Vec::with_capacity(num_rows_b);
        for row_b in 0..num_rows_b {
            let mut acc = api.constant(0);
            for col_b in 0..num_cols_b {
                let mul = api.mul(mat_b[(row_b, col_b)], rand_vec[col_b]);
                acc = api.add(acc, mul);
            }
            b_times_rand.push(acc);
        }

        // 3) c_times_rand = C * rand_vec, length ell
        let mut c_times_rand: Vec<Variable> = Vec::with_capacity(num_rows_c);
        for row_c in 0..num_rows_c {
            let mut acc = api.constant(0);
            for col_c in 0..num_cols_c {
                let mul = api.mul(mat_c[(row_c, col_c)], rand_vec[col_c]);
                acc = api.add(acc, mul);
            }
            c_times_rand.push(acc);
        }

        // 4) a_times_b_times_rand = A * (B * rand_vec), length ell
        let mut a_times_b_times_rand: Vec<Variable> = Vec::with_capacity(num_rows_a);
        for row_a in 0..num_rows_a {
            let mut acc = api.constant(0);
            for col_a in 0..num_cols_a {
                let mul = api.mul(mat_a[(row_a, col_a)], b_times_rand[col_a]);
                acc = api.add(acc, mul);
            }
            a_times_b_times_rand.push(acc);
        }

        // 5) Enforce equality entrywise
        for row in 0..num_rows_a {
            api.assert_is_equal(a_times_b_times_rand[row], c_times_rand[row]);
        }
    }

    Ok(())
}

const STRASSEN_THRESHOLD: usize = 64;

fn should_use_strassen(m: usize, n: usize, p: usize) -> bool {
    let min_dim = m.min(n).min(p);
    let max_dim = m.max(n).max(p);
    min_dim >= STRASSEN_THRESHOLD && max_dim <= 4 * min_dim
}

fn block_add<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Array2<Variable>,
    b: &Array2<Variable>,
) -> Array2<Variable> {
    let (rows, cols) = a.dim();
    let mut result = Array2::default((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            result[(i, j)] = api.add(a[(i, j)], b[(i, j)]);
        }
    }
    result
}

fn block_sub<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Array2<Variable>,
    b: &Array2<Variable>,
) -> Array2<Variable> {
    let (rows, cols) = a.dim();
    let mut result = Array2::default((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            result[(i, j)] = api.sub(a[(i, j)], b[(i, j)]);
        }
    }
    result
}

fn strassen_recurse<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    mat_left: &Array2<Variable>,
    mat_right: &Array2<Variable>,
    layer_type: LayerKind,
) -> Result<Array2<Variable>, LayerError> {
    let (rows, inner) = mat_left.dim();
    let (_, cols) = mat_right.dim();

    if rows < STRASSEN_THRESHOLD || inner < STRASSEN_THRESHOLD || cols < STRASSEN_THRESHOLD {
        let result = matrix_multiplication(
            api,
            mat_left.clone().into_dyn(),
            mat_right.clone().into_dyn(),
            layer_type,
        )?;
        return result
            .into_dimensionality::<Ix2>()
            .map_err(|_| LayerError::InvalidShape {
                layer: LayerKind::Gemm,
                msg: "strassen base case result not 2D".to_string(),
            });
    }

    let pad_rows = if rows % 2 != 0 { rows + 1 } else { rows };
    let pad_inner = if inner % 2 != 0 { inner + 1 } else { inner };
    let pad_cols = if cols % 2 != 0 { cols + 1 } else { cols };

    let needs_pad = pad_rows != rows || pad_inner != inner || pad_cols != cols;

    let (left_padded, right_padded) = if needs_pad {
        let zero = api.constant(0);
        let mut lp = Array2::from_elem((pad_rows, pad_inner), zero);
        for i in 0..rows {
            for j in 0..inner {
                lp[(i, j)] = mat_left[(i, j)];
            }
        }
        let mut rp = Array2::from_elem((pad_inner, pad_cols), zero);
        for i in 0..inner {
            for j in 0..cols {
                rp[(i, j)] = mat_right[(i, j)];
            }
        }
        (lp, rp)
    } else {
        (mat_left.clone(), mat_right.clone())
    };

    let half_rows = pad_rows / 2;
    let half_inner = pad_inner / 2;
    let half_cols = pad_cols / 2;

    let a11 = left_padded.slice(s![..half_rows, ..half_inner]).to_owned();
    let a12 = left_padded.slice(s![..half_rows, half_inner..]).to_owned();
    let a21 = left_padded.slice(s![half_rows.., ..half_inner]).to_owned();
    let a22 = left_padded.slice(s![half_rows.., half_inner..]).to_owned();

    let b11 = right_padded.slice(s![..half_inner, ..half_cols]).to_owned();
    let b12 = right_padded.slice(s![..half_inner, half_cols..]).to_owned();
    let b21 = right_padded.slice(s![half_inner.., ..half_cols]).to_owned();
    let b22 = right_padded.slice(s![half_inner.., half_cols..]).to_owned();

    let s1 = block_add::<C, Builder>(api, &a11, &a22);
    let s2 = block_add::<C, Builder>(api, &b11, &b22);
    let m1 = strassen_recurse::<C, Builder>(api, &s1, &s2, layer_type.clone())?;

    let s3 = block_add::<C, Builder>(api, &a21, &a22);
    let m2 = strassen_recurse::<C, Builder>(api, &s3, &b11, layer_type.clone())?;

    let s4 = block_sub::<C, Builder>(api, &b12, &b22);
    let m3 = strassen_recurse::<C, Builder>(api, &a11, &s4, layer_type.clone())?;

    let s5 = block_sub::<C, Builder>(api, &b21, &b11);
    let m4 = strassen_recurse::<C, Builder>(api, &a22, &s5, layer_type.clone())?;

    let s6 = block_add::<C, Builder>(api, &a11, &a12);
    let m5 = strassen_recurse::<C, Builder>(api, &s6, &b22, layer_type.clone())?;

    let s7 = block_sub::<C, Builder>(api, &a21, &a11);
    let s8 = block_add::<C, Builder>(api, &b11, &b12);
    let m6 = strassen_recurse::<C, Builder>(api, &s7, &s8, layer_type.clone())?;

    let s9 = block_sub::<C, Builder>(api, &a12, &a22);
    let s10 = block_add::<C, Builder>(api, &b21, &b22);
    let m7 = strassen_recurse::<C, Builder>(api, &s9, &s10, layer_type)?;

    let t1 = block_add::<C, Builder>(api, &m1, &m4);
    let t2 = block_sub::<C, Builder>(api, &t1, &m5);
    let c11 = block_add::<C, Builder>(api, &t2, &m7);

    let c12 = block_add::<C, Builder>(api, &m3, &m5);

    let c21 = block_add::<C, Builder>(api, &m2, &m4);

    let t3 = block_sub::<C, Builder>(api, &m1, &m2);
    let t4 = block_add::<C, Builder>(api, &t3, &m3);
    let c22 = block_add::<C, Builder>(api, &t4, &m6);

    let mut result = Array2::default((pad_rows, pad_cols));
    for i in 0..half_rows {
        for j in 0..half_cols {
            result[(i, j)] = c11[(i, j)];
            result[(i, j + half_cols)] = c12[(i, j)];
            result[(i + half_rows, j)] = c21[(i, j)];
            result[(i + half_rows, j + half_cols)] = c22[(i, j)];
        }
    }

    if needs_pad {
        Ok(result.slice(s![..rows, ..cols]).to_owned())
    } else {
        Ok(result)
    }
}

/// # Errors
/// Returns `LayerError::InvalidShape` if either input is not 2D.
/// Returns `LayerError::ShapeMismatch` if A.cols != B.rows.
pub fn strassen_matrix_multiplication<C: Config, Builder: RootAPI<C>>(
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

    if !should_use_strassen(dim_m, dim_n, dim_p) {
        return matrix_multiplication(api, a.into_dyn(), b.into_dyn(), layer_type);
    }

    let result = strassen_recurse::<C, Builder>(api, &a, &b, layer_type)?;
    Ok(result.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strassen_below_threshold_returns_false() {
        assert!(!should_use_strassen(32, 32, 32));
        assert!(!should_use_strassen(63, 63, 63));
        assert!(!should_use_strassen(64, 32, 64));
    }

    #[test]
    fn strassen_at_threshold_returns_true() {
        assert!(should_use_strassen(64, 64, 64));
        assert!(should_use_strassen(128, 128, 128));
        assert!(should_use_strassen(64, 100, 80));
    }

    #[test]
    fn strassen_highly_rectangular_returns_false() {
        assert!(!should_use_strassen(64, 256 + 1, 64));
        assert!(!should_use_strassen(1000, 100, 100));
    }

    #[test]
    fn strassen_moderately_rectangular_returns_true() {
        assert!(should_use_strassen(64, 128, 64));
        assert!(should_use_strassen(100, 200, 150));
    }
}
