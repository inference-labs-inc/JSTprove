/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
};

use crate::circuit_functions::gadgets::LogupRangeCheckContext;

// ─────────────────────────────────────────────────────────────────────────────
// CONTEXT: ShiftRangeContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for applying `constrained_max` / `constrained_min` with a fixed
/// shift exponent `s`, to avoid recomputing constants in repeated calls.
///
/// This is shared across:
///   - MaxPool (windowed max over tensors),
///   - MaxLayer (elementwise max),
///   - MinLayer (elementwise min).
pub struct ShiftRangeContext {
    /// The exponent `s` such that `S = 2^s`.
    pub shift_exponent: usize,

    /// The offset `S = 2^s`, lifted as a constant into the circuit.
    pub offset: Variable,
}

impl ShiftRangeContext {
    /// Creates a new context for asserting maximums/minimums, given a
    /// `shift_exponent = s`.
    ///
    /// # Arguments
    /// - `api`: Mutable reference to a builder for creating constants.
    /// - `shift_exponent`: Exponent `s` used to compute `2^s`.
    ///
    /// # Errors
    /// - [`LayerError::Other`] if `shift_exponent` is too large to fit in a `u32`.
    /// - [`LayerError::InvalidParameterValue`] if the computed offset overflows `u32`.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        shift_exponent: usize,
    ) -> Result<Self, LayerError> {
        let offset_: u32 = 1u32
            .checked_shl(
                u32::try_from(shift_exponent).map_err(|_| LayerError::Other {
                    layer: LayerKind::MaxPool,
                    msg: format!("Shift exponent {shift_exponent} is too large for type: u32"),
                })?,
            )
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "ShiftRangeContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: shift_exponent.to_string(),
            })?;
        let offset = api.constant(offset_);
        Ok(Self {
            shift_exponent,
            offset,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_max
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts that a given slice of `Variable`s contains a maximum value `M`,
/// by verifying that some `x_i` satisfies `M = max_i x_i`, using a combination of
/// unconstrained helper functions and explicit constraint assertions,
/// along with an offset-shifting technique to reduce comparisons to the
/// nonnegative range `[0, 2^(s + 1) − 1]`.
///
/// # Idea
/// Each `x_i` is a field element (i.e., a `Variable` representing the least nonnegative residue mod `p`)
/// that is **assumed** to encode a signed integer in the interval `[-S, T − S] = [−2^s, 2^s − 1]`,
/// where `S = 2^s` and `T = 2·S - 1 = 2^(s + 1) − 1]`.
///
/// Since all circuit operations take place in `𝔽_p` and each `x_i` is already reduced modulo `p`,
/// we shift each value by `S` on-circuit to ensure that the quantity `x_i + S` lands in `[0, T]`.
/// Under the assumption that `x_i ∈ [−S, T − S]`, this shift does **not** wrap around modulo `p`,
/// so `x_i + S` in `𝔽_p` reflects the true integer sum.
///
/// We then compute:
/// ```text
///     M^♯ = max_i (x_i + S)
///     M   = M^♯ − S mod p
/// ```
/// to recover the **least nonnegative residue** of the maximum value, `M`.
///
/// To verify that `M` is indeed the maximum:
/// - For each `x_i`, we compute `Δ_i = M − x_i`, and use bit decomposition to enforce
///   `Δ_i ∈ [0, T]`, using `s + 1` bits.
/// - Then we constrain the product `∏_i Δ_i` to be zero. This ensures that at least one
///   `Δ_i = 0`, i.e., that some `x_i = M`.
///
/// # Example
/// Suppose the input slice encodes the signed integers `[-2, 0, 3]`, and `s = 2`, so `S = 4`, `T = 7`.
///
/// - Shift:
///   `x_0 = -2` ⇒ `x_0 + S = 2`
///   `x_1 =  0` ⇒ `x_1 + S = 4`
///   `x_2 =  3` ⇒ `x_2 + S = 7`
///
/// - Compute:
///   `M^♯ = max{x_i + S} = 7`
///   `M   = M^♯ − S = 3`
///
/// - Verify:
///   For each `x_i`, compute `Δ_i = M − x_i ∈ [0, 7]`
///   The values are: `Δ = [5, 3, 0]`
///   Since one `Δ_i = 0`, we conclude that some `x_i = M`.
///
/// # Assumptions
/// - All values `x_i` are `Variable`s in `𝔽_p` that **encode signed integers** in `[-S, T − S]`.
/// - The prime `p` satisfies `p > T = 2^(s + 1) − 1`, so no wraparound occurs in `x_i + S`.
///
/// # Errors
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `context`: A `ShiftRangeContext` holding shift-related parameters.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
pub fn constrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    shift_ctx: &ShiftRangeContext, // S = 2^s = shift_ctx.offset
    logup_ctx: &mut LogupRangeCheckContext, // reused LogUp table
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    // 0) Require nonempty input
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Max,
            msg: "constrained_max: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, shift_ctx.offset));
    }

    // 2) Compute max_i (x_i^♯), which equals M^♯ = M + S
    let max_offset = unconstrained_max(api, &values_offset)?;

    // 3) Recover M = M^♯ − S
    let max_raw = api.sub(max_offset, shift_ctx.offset);

    // 4) For each x_i, range-check Δ_i = M − x_i ∈ [0, T] using s + 1 bits
    let n_bits = shift_ctx.shift_exponent.checked_add(1).ok_or_else(|| {
        LayerError::InvalidParameterValue {
            layer: LayerKind::Max,
            layer_name: "ShiftRangeContext".to_string(),
            param_name: "shift_exponent".to_string(),
            value: shift_ctx.shift_exponent.to_string(),
        }
    })?;

    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        // Δ ∈ [0, T] = [0, 2^{s + 1} - 1] via *shared* LogUp-based range proof
        logup_ctx
            .range_check::<C, Builder>(api, delta, n_bits)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Max,
                msg: format!("logup_range_check_pow2_unsigned (LogUp) failed: {e}"),
            })?;

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
    Ok(max_raw)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_min
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts that a given slice of `Variable`s contains a minimum value `M`,
/// by verifying that some `x_i` satisfies `M = min_i x_i`, using a combination of
/// unconstrained helper functions and explicit constraint assertions,
/// along with an offset-shifting technique to reduce comparisons to the
/// nonnegative range `[0, 2^(s + 1) − 1]`.
///
/// # Idea
/// Each `x_i` is a field element (i.e., a `Variable` representing the least nonnegative residue mod `p`)
/// that is **assumed** to encode a signed integer in the interval `[-S, T − S] = [−2^s, 2^s − 1]`,
/// where `S = 2^s` and `T = 2·S - 1 = 2^(s + 1) − 1]`.
///
/// Since all circuit operations take place in `𝔽_p` and each `x_i` is already reduced modulo `p`,
/// we shift each value by `S` on-circuit to ensure that the quantity `x_i + S` lands in `[0, T]`.
/// Under the assumption that `x_i ∈ [−S, T − S]`, this shift does **not** wrap around modulo `p`,
/// so `x_i + S` in `𝔽_p` reflects the true integer sum.
///
/// We then compute:
/// ```text
///     M^♯ = min_i (x_i + S)
///     M   = M^♯ − S mod p
/// ```
/// to recover the **least nonnegative residue** of the minimum value, `M`.
///
/// To verify that `M` is indeed the minimum:
/// - For each `x_i`, we compute `Δ_i = x_i - M`, and use bit decomposition to enforce
///   `Δ_i ∈ [0, T]`, using `s + 1` bits.
/// - Then we constrain the product `∏_i Δ_i` to be zero. This ensures that at least one
///   `Δ_i = 0`, i.e., that some `x_i = M`.
///
/// # Example
/// Suppose the input slice encodes the signed integers `[-2, 0, 3]`, and `s = 2`, so `S = 4`, `T = 7`.
///
/// - Shift:
///   `x_0 = -2` ⇒ `x_0 + S = 2`
///   `x_1 =  0` ⇒ `x_1 + S = 4`
///   `x_2 =  3` ⇒ `x_2 + S = 7`
///
/// - Compute:
///   `M^♯ = min{x_i + S} = 2`
///   `M   = M^♯ − S = -2`
///
/// - Verify:
///   For each `x_i`, compute `Δ_i = x_i - M ∈ [0, 7]`
///   The values are: `Δ = [0, 2, 5]`
///   Since one `Δ_i = 0`, we conclude that some `x_i = M`.
///
/// # Assumptions
/// - All values `x_i` are `Variable`s in `𝔽_p` that **encode signed integers** in `[-S, T − S]`.
/// - The prime `p` satisfies `p > T = 2^(s + 1) − 1`, so no wraparound occurs in `x_i + S`.
///
/// # Errors
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `context`: A `ShiftRangeContext` holding shift-related parameters.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
pub fn constrained_min<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &ShiftRangeContext,            // S = 2^s = context.offset
    logup_ctx: &mut LogupRangeCheckContext, // reused LogUp table
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    // 0) Require nonempty input
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Min,
            msg: "constrained_min: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute min_i (x_i^♯), which equals M^♯ = M + S
    let min_offset = unconstrained_min(api, &values_offset)?;

    // 3) Recover M = M^♯ − S
    let min_raw = api.sub(min_offset, context.offset);

    // 4) For each x_i, range-check Δ_i = x_i − M ∈ [0, T] using s + 1 bits
    let n_bits =
        context
            .shift_exponent
            .checked_add(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::Min,
                layer_name: "ShiftRangeContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: context.shift_exponent.to_string(),
            })?;

    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(x, min_raw);

        // Δ ∈ [0, T] = [0, 2^{s + 1} - 1] via *shared* LogUp-based range proof
        logup_ctx
            .range_check::<C, Builder>(api, delta, n_bits)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Min,
                msg: format!("logup_range_check_pow2_unsigned (LogUp) failed: {e}"),
            })?;

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
    Ok(min_raw)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_clip
// ─────────────────────────────────────────────────────────────────────────────

/// Enforces `c = clip(x; min, max)` using the existing `constrained_max` and
/// `constrained_min` gadgets under a shared [`ShiftRangeContext`].
///
/// Assumptions:
/// - `x`, `min`, and `max` (if present) all encode signed integers in the range
///   `[-S, T - S] = [-2^s, 2^s - 1]`, where `S = 2^s` is given by
///   `context.shift_exponent`.
/// - The field modulus `p` satisfies `p > 2^(s + 1) - 1` to avoid wraparound.
///
/// Semantics match ONNX `Clip`:
/// - If both `lower` and `upper` are present:
///       c = min(max(x, lower), upper)
/// - If only `lower` is present:
///       c = max(x, lower)
/// - If only `upper` is present:
///       c = min(x, upper)
/// - If neither is present:
///       c = x
///
/// Each `constrained_max` / `constrained_min` call:
/// - Checks the signed range via bit-decomposition and reconstruction.
/// - Enforces that the result equals one of the inputs.
/// - Uses a product-of-differences check to guarantee at least one exact match.
pub fn constrained_clip<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    range_ctx: &ShiftRangeContext,
    logup_ctx: &mut LogupRangeCheckContext,
    x: Variable,
    lower: Option<Variable>,
    upper: Option<Variable>,
) -> Result<Variable, CircuitError> {
    // Degenerate case: no bounds → identity
    if lower.is_none() && upper.is_none() {
        return Ok(x);
    }

    // Apply lower bound via constrained_max when present
    let mut cur = if let Some(a) = lower {
        constrained_max(api, range_ctx, logup_ctx, &[x, a])?
    } else {
        x
    };

    // Apply upper bound via constrained_min when present
    if let Some(b) = upper {
        cur = constrained_min(api, range_ctx, logup_ctx, &[cur, b])?;
    }

    Ok(cur)
}
