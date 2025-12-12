//! Unconstrained max/min/clip helpers.
//!
//! These are NOT used for soundness — they only compute witnesses.
//! Constrained counterparts live in `gadgets/max_min_clip.rs`.

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::CircuitError;

// -----------------------------------------------------------------------------
// FUNCTION: unconstrained_max
// -----------------------------------------------------------------------------

/// Computes `max(values)` using *only unconstrained witness operations*
/// (`unconstrained_greater`, `unconstrained_lesser_eq`, and
/// `unconstrained_mul`). No constraints are added to the circuit.
///
/// This is a helper for constructing witnesses in max-selection gadgets.
/// It is **not** sound on its own; all correctness of the selection must be
/// enforced later by a constrained gadget such as `constrained_max`.
///
/// # Overview
///
/// Given an input slice `[x_0, x_1, ..., x_{n-1}]`, the function iteratively
/// updates:
///
/// ```text
/// current_max = x_0
/// For each v in values[1..]:
///     if v > current_max: current_max = v
/// ```
///
/// Since we cannot branch in a circuit, this is implemented with witness-side
/// selectors:
///
/// ```text
/// is_greater     = (v > current_max)
/// is_not_greater = (v <= current_max)
/// current_max = v * is_greater + current_max * is_not_greater
/// ```
///
/// These selectors are witness values (not constrained to be boolean), so this
/// routine performs no correctness checks. The constrained layer is responsible
/// for verifying the max-selection proof.
///
/// # Assumptions
///
/// - All inputs represent nonnegative integers in `[0, p-1]` under least
///   nonnegative residue interpretation.
/// - The caller will later enforce correctness in a constrained gadget.
///
/// # Errors
///
/// - Returns `CircuitError::Other` if `values` is empty.
///
/// # Arguments
///
/// - `api`: circuit builder supporting unconstrained operations.
/// - `values`: nonempty slice of `Variable`s.
///
/// # Returns
///
/// The unconstrained witness representing the maximum.
///
/// # Example
///
/// ```ignore
/// values = [7, 2, 9, 5]  --> returns 9
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

// -----------------------------------------------------------------------------
// FUNCTION: unconstrained_min
// -----------------------------------------------------------------------------

/// Computes `min(values)` using only *unconstrained witness operations*
/// (`unconstrained_lesser`, `unconstrained_greater_eq`, and
/// `unconstrained_mul`). No constraints are added.
///
/// This function mirrors `unconstrained_max`, but selects the minimum.
///
/// # Overview
///
/// For `[x_0, x_1, ..., x_{n-1}]`, we update:
///
/// ```text
/// current_min = x_0
/// For each v in values[1..]:
///     if v < current_min: current_min = v
/// ```
///
/// Selector form:
///
/// ```text
/// is_lesser     = (v < current_min)
/// is_not_lesser = (v >= current_min)
/// current_min = v * is_lesser + current_min * is_not_lesser
/// ```
///
/// As with `unconstrained_max`, this performs **no correctness enforcement**.
/// Constrained checking (e.g. via `constrained_min`) must be applied later.
///
/// # Assumptions
///
/// - Inputs represent integers in `[0, p-1]`.
/// - Caller will enforce correctness using a constrained gadget.
///
/// # Errors
///
/// - Returns `CircuitError::Other` if `values` is empty.
///
/// # Arguments
///
/// - `api`: circuit builder.
/// - `values`: nonempty slice of field elements.
///
/// # Returns
///
/// An unconstrained witness representing the minimum.
///
/// # Example
///
/// ```ignore
/// values = [7, 2, 9, 5]  --> returns 2
/// ```
pub fn unconstrained_min<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        // This is a precondition failure, not an ONNX-layer semantic error.
        return Err(CircuitError::Other(
            "unconstrained_min: input slice must be nonempty".to_string(),
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

// -----------------------------------------------------------------------------
// FUNCTION: unconstrained_clip
// -----------------------------------------------------------------------------

/// Computes the witness value for `clip(x; lower, upper)` using only
/// *unconstrained* operations.
///
/// This function is intended to generate witnesses for `constrained_clip`,
/// `constrained_max`, and `constrained_min`. It enforces **no constraints**
/// on the output.
///
/// # Semantics
///
/// ```text
/// if lower and upper exist:
///     y = min(max(x, lower), upper)
/// else if only lower exists:
///     y = max(x, lower)
/// else if only upper exists:
///     y = min(x, upper)
/// else:
///     y = x
/// ```
///
/// # Important Notes
///
/// - This routine does NOT verify that `lower <= upper`.
/// - It does NOT enforce booleanity of selection signals.
/// - All correctness must be enforced later by the constrained Clip layer.
///
/// # Assumptions
///
/// - `x`, `lower`, and `upper` represent integers in a compatible signed range.
/// - The surrounding circuit will use constrained gadgets to verify Clip logic.
///
/// # Errors
///
/// - Propagates any `CircuitError` returned by `unconstrained_max` or
///   `unconstrained_min`.
///
/// # Arguments
///
/// - `api`: circuit builder with unconstrained operators.
/// - `x`: the input value to be clipped.
/// - `lower`: optional lower bound.
/// - `upper`: optional upper bound.
///
/// # Returns
///
/// The unconstrained witness for the clipped value.
///
/// # Example
///
/// ```ignore
/// x = 7, lower = 2, upper = 5 --> returns 5
/// x = 1, lower = 2            --> returns 2
/// ```
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
