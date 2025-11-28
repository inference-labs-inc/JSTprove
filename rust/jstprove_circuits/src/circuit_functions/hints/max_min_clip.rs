use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::CircuitError;

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_max
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the maximum value in a nonempty slice of field elements (interpreted as integers in `[0, p−1]`),
/// using only unconstrained witness operations and explicit selection logic.
///
/// Internally, this function performs pairwise comparisons using `unconstrained_greater` and `unconstrained_lesser_eq`,
/// and selects the maximum via weighted sums:
/// `current_max ← v·(v > current_max) + current_max·(v ≤ current_max)`
///
/// # Errors
/// - If `values` is empty.
///
/// # Arguments
/// - `api`: A mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `values`: A slice of `Variable`s, each assumed to lie in the range `[0, p−1]`.
///
/// # Returns
/// A `Variable` encoding `max_i values[i]`, the maximum value in the slice.
///
/// # Example
/// ```ignore
/// // For values = [7, 2, 9, 5], returns 9.
/// ```
pub fn unconstrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        // This is a precondition failure, not an ONNX-layer semantic error.
        return Err(CircuitError::Other(
            "unconstrained_max: input slice must be nonempty".to_string(),
        ));
    }

    // Initialize with the first element
    let mut current_max = values[0];
    for &v in &values[1..] {
        // Compute indicators: is_greater = 1 if v > current_max, else 0
        let is_greater = api.unconstrained_greater(v, current_max);
        let is_not_greater = api.unconstrained_lesser_eq(v, current_max);

        // Select either v or current_max based on indicator bits
        let take_v = api.unconstrained_mul(v, is_greater);
        let keep_old = api.unconstrained_mul(current_max, is_not_greater);

        // Update current_max
        current_max = api.unconstrained_add(take_v, keep_old);
    }

    Ok(current_max)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_min
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the minimum value in a nonempty slice of field elements (interpreted as integers in `[0, p−1]`),
/// using only unconstrained witness operations and explicit selection logic.
///
/// Internally, this function performs pairwise comparisons using `unconstrained_greater` and `unconstrained_lesser_eq`,
/// and selects the minimum via weighted sums:
/// `current_min ← v·(current_min > v) + current_min·(current_min ≤ v)`
///
/// # Errors
/// - If `values` is empty.
///
/// # Arguments
/// - `api`: A mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `values`: A slice of `Variable`s, each assumed to lie in the range `[0, p−1]`.
///
/// # Returns
/// A `Variable` encoding `min_i values[i]`, the minimum value in the slice.
///
/// # Example
/// ```ignore
/// // For values = [7, 2, 9, 5], returns 2.
/// ```
pub fn unconstrained_min<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        // This is a precondition failure, not an ONNX-layer semantic error.
        return Err(CircuitError::Other(
            "unconstrained_max: input slice must be nonempty".to_string(),
        ));
    }

    // Initialize with the first element
    let mut current_min = values[0];
    for &v in &values[1..] {
        // Compute indicators: is_lesser = 1 if v < current_min, else 0
        let is_lesser = api.unconstrained_lesser(v, current_min);
        let is_not_lesser = api.unconstrained_greater_eq(v, current_min);

        // Select either v or current_min based on indicator bits
        let take_v = api.unconstrained_mul(v, is_lesser);
        let keep_old = api.unconstrained_mul(current_min, is_not_lesser);

        // Update current_min
        current_min = api.unconstrained_add(take_v, keep_old);
    }

    Ok(current_min)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_clip
// ─────────────────────────────────────────────────────────────────────────────

/// Computes `clip(x; min, max)` at the witness level using only unconstrained
/// operations and the existing `unconstrained_max` / `unconstrained_min` gadgets.
///
/// Semantics (elementwise, assuming `min <= max` in the intended integer semantics):
/// - If both `lower` and `upper` are present:
///       y = min(max(x, lower), upper)
/// - If only `lower` is present:
///       y = max(x, lower)
/// - If only `upper` is present:
///       y = min(x, upper)
/// - If neither is present:
///       y = x
///
/// All variables are field elements (least nonnegative residues), interpreted
/// as signed integers in a fixed range consistent with the surrounding circuit.
pub fn unconstrained_clip<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    lower: Option<Variable>,
    upper: Option<Variable>,
) -> Result<Variable, CircuitError> {
    // Start from x and apply lower / upper bounds as needed.
    let mut cur = x;

    if let Some(a) = lower {
        // cur <- max(cur, a)
        cur = unconstrained_max(api, &[cur, a])?;
    }

    if let Some(b) = upper {
        // cur <- min(cur, b)
        cur = unconstrained_min(api, &[cur, b])?;
    }

    Ok(cur)
}
