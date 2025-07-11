use expander_compiler::frontend::*;
use crate::circuit_functions::{
    assert_is_max,
    rescale,
};

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: rescale_matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Applies [`rescale`] elementwise to a 2D matrix of [`Variable`]s,
/// using a shared [`RescalingContext`] to avoid redundant computation.
///
/// # Arguments
/// - `api`: Circuit builder implementing [`RootAPI<C>`].
/// - `context`: Precomputed [`RescalingContext`] holding scaling constants.
/// - `dividend_matrix`: 2D matrix of [`Variable`]s to be rescaled.
/// - `apply_relu`: If `true`, outputs `max(q, 0)` elementwise instead of `q`.
///
/// # Returns
/// A 2D matrix with each element rescaled via [`rescale`].
///
/// # Example
/// ```ignore
/// let scaling_exponent = 3; // κ = 3 ⇒ α = 2³ = 8
/// let shift_exponent = 2;   // s = 2 ⇒ S = 2² = 4
/// let apply_relu = true;
///
/// let context = RescalingContext::new(api, scaling_exponent, shift_exponent);
/// let input_matrix: Vec<Vec<Variable>> = vec![
///     vec![x00, x01],
///     vec![x10, x11],
/// ];
///
/// let output_matrix = rescale_matrix(api, &context, input_matrix, apply_relu);
/// ```
pub fn rescale_matrix<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &RescalingContext<C>,
    dividend_matrix: Vec<Vec<Variable>>,
    apply_relu: bool,
) -> Vec<Vec<Variable>> {
    dividend_matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|dividend| rescale(api, context, dividend, apply_relu))
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: rescale_4d_vector
// ─────────────────────────────────────────────────────────────────────────────

/// Applies `rescale` elementwise to a 4D tensor of `Variable`s.
///
/// # Arguments
/// - `api`: Circuit builder implementing `RootAPI<C>`.
/// - `input_tensor`: 4D vector (Vec of Vec of Vec of Vec) of `Variable`s to be rescaled.
/// - `context`: Precomputed [`RescalingContext`] containing constants α, S, and α·S.
/// - `apply_relu`: If `true`, applies `max(q, 0)` elementwise.
///
/// # Returns
/// A 4D tensor with each element rescaled via `rescale`.
///
/// # Example
/// ```ignore
/// let tensor: Vec<Vec<Vec<Vec<Variable>>>> = vec![
///     vec![
///         vec![vec![x0000, x0001], vec![x0010, x0011]],
///         vec![vec![x0100, x0101], vec![x0110, x0111]],
///     ],
/// ];
/// let context = RescalingContext::new(api, scaling_exponent, shift_exponent);
/// let out = rescale_4d_vector(api, tensor, &context, true);
/// ```
pub fn rescale_4d_vector<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_tensor: Vec<Vec<Vec<Vec<Variable>>>>,
    context: &RescalingContext<C>,
    apply_relu: bool,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    input_tensor
        .into_iter()
        .map(|dim1| {
            dim1.into_iter()
                .map(|dim2| {
                    dim2.into_iter()
                        .map(|dim3| {
                            dim3.into_iter()
                                .map(|x| rescale(api, context, x, apply_relu))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}