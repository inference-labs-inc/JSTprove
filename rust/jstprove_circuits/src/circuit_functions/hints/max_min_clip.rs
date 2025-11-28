use expander_compiler::frontend::{Config, RootAPI, Variable};

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
        return Err(LayerError::Other {
            layer: LayerKind::Max,
            msg: "unconstrained_max: input slice must be nonempty".to_string(),
        }
        .into());
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
