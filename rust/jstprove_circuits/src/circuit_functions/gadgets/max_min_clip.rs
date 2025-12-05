//! Constrained max/min/clip gadgets for int64 fixed-point arithmetic.
//!
//! These gadgets:
//! - enforce correctness of max/min selection via `LogUp` range checks,
//! - use unconstrained comparison hints internally,
//! - are used by `Max`, `Min`, `Clip`, `MaxPool`, and quantization layers.

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
};

use crate::circuit_functions::gadgets::LogupRangeCheckContext;

// Unconstrained arithmetic helpers from hints/
use crate::circuit_functions::hints::{unconstrained_max, unconstrained_min};

// -----------------------------------------------------------------------------
// CONTEXT: ShiftRangeContext
// -----------------------------------------------------------------------------

/// Context for applying `constrained_max` / `constrained_min` with a fixed
/// shift exponent `s`, so we do not recompute constants on every call.
///
/// This context is shared across:
///   - `MaxPool` (windowed max over tensors),
///   - `MaxLayer` (elementwise max),
///   - `MinLayer` (elementwise min).
///
/// Intuitively:
/// - We treat each value x as a signed integer in the range [-2^s, 2^s - 1].
/// - We define an offset S = 2^s and lift it into the circuit as a constant.
/// - Max/min gadgets work with the shifted values x + S, which are in
///   [0, 2^(s+1) - 1], and then subtract S again at the end.
///
/// This struct packages the shared parameters (s and S) so that layers and
/// gadgets can agree on a consistent signed range.
pub struct ShiftRangeContext {
    /// The exponent s such that S = 2^s.
    pub shift_exponent: usize,

    /// The offset S = 2^s, lifted as a constant into the circuit.
    pub offset: Variable,
}

impl ShiftRangeContext {
    /// Creates a new context for max/min assertion gadgets, given a
    /// `shift_exponent = s`.
    ///
    /// Internally computes S = 2^s as a `u32`, lifts it as a circuit
    /// constant, and stores both `s` and `S`.
    ///
    /// # Arguments
    /// - `api`: mutable reference to the circuit builder.
    /// - `shift_exponent`: exponent s used to compute S = 2^s.
    ///
    /// # Errors
    /// - `LayerError::Other` if `shift_exponent` is too large to fit in `u32`.
    /// - `LayerError::InvalidParameterValue` if computing `2^s` overflows `u32`.
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

// -----------------------------------------------------------------------------
// FUNCTION: constrained_max
// -----------------------------------------------------------------------------

/// Enforces that a slice of variables has a maximum value `M`, and that the
/// returned variable equals that maximum.
///
/// # High-level idea
///
/// Each input `x_i` is a field element (a `Variable` in `F_p`) that we *interpret*
/// as encoding a signed integer in the range:
///
///     x_i in [-S, 2^s - 1]
///
/// where S = 2^s is provided by `shift_ctx.offset`.
///
/// To stay inside a nonnegative interval, we shift each value:
///
///     x_i_sh = x_i + S
///
/// Under the assumption that the field modulus p satisfies:
///
///     p > 2^(s+1) - 1
///
/// this shift does not wrap modulo p, and the values lie in:
///
///     x_i_sh in [0, 2^(s+1) - 1].
///
/// We then:
///
/// 1. Compute:
///   max_sh = max_i (x_i_sh)
///    using an *unconstrained* helper (`unconstrained_max`).
///
/// 2. Recover the candidate maximum in the original signed range:
///   M = max_sh - S.
///
/// 3. For each i, define:
///   delta_i = M - x_i.
///
///    We enforce that each `delta_i` lies in:
///        [0, 2^(s+1) - 1]
///    by calling the shared `logup_ctx.range_check` with `n_bits = s + 1`.
///
/// 4. Compute the product:
///        prod = product_{i} delta_i
///    and assert `prod == 0`.
///
///    Since every `delta_i` is constrained to be in the nonnegative range,
///    the only way the product can be zero is if at least one `delta_i` is
///    actually zero. That in turn implies `x_i == M` for some i, so M really
///    is one of the inputs.
///
/// # Assumptions
/// - All inputs encode signed integers in [-S, 2^s - 1].
/// - The field modulus p is greater than 2^(s+1) - 1 so that x_i + S does not
///   wrap modulo p.
/// - A shared `LogupRangeCheckContext` is available and initialized in the
///   surrounding circuit (the caller is responsible for calling `finalize`).
///
/// # Arguments
/// - `api`: circuit builder.
/// - `shift_ctx`: precomputed shift parameters (s and S = 2^s).
/// - `logup_ctx`: shared LogUp range-check context used to prove each delta_i
///   is in [0, 2^(s+1) - 1].
/// - `values`: nonempty slice of variables to be maximized.
///
/// # Errors
/// - Returns `LayerError::Other` (wrapped as `CircuitError`) if `values` is
///   empty.
/// - Returns `LayerError::InvalidParameterValue` if `shift_exponent + 1`
///   overflows `usize` when computing the bit-length for the range check.
/// - Propagates any internal circuit errors (e.g., from LogUp calls).
///
/// # Returns
/// - A `Variable` equal to the least nonnegative residue of the maximum value.
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

    // 1) Form offset-shifted values: x_i_sh = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, shift_ctx.offset));
    }

    // 2) Compute max_i (x_i_sh), which equals M_sh = M + S
    let max_offset = unconstrained_max(api, &values_offset)?;

    // 3) Recover M = M_sh − S
    let max_raw = api.sub(max_offset, shift_ctx.offset);

    // 4) For each x_i, range-check delta_i = M − x_i in [0, T] using s + 1 bits
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

        // delta in [0, T] = [0, 2^{s + 1} - 1] via *shared* LogUp-based range proof
        logup_ctx
            .range_check::<C, Builder>(api, delta, n_bits)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Max,
                msg: format!("logup_range_check_pow2_unsigned (LogUp) failed: {e}"),
            })?;

        // Multiply all delta_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: prod delta_i = 0 iff exists x_i such that delta_i = 0 iff x_i = M
    api.assert_is_zero(prod);
    Ok(max_raw)
}

// -----------------------------------------------------------------------------
// FUNCTION: constrained_min
// -----------------------------------------------------------------------------

/// Enforces that a slice of variables has a minimum value `M`, and that the
/// returned variable equals that minimum.
///
/// # High-level idea
///
/// This is the min-analogue of `constrained_max`, using the same offset-shift
/// strategy. Again, we interpret each `x_i` as encoding a signed integer in:
///
///     x_i in [-S, 2^s - 1]
///
/// with S = 2^s given by `context.offset`. We shift to nonnegative space:
///
///     x_i_sh = x_i + S in [0, 2^(s+1) - 1]
///
/// assuming p > 2^(s+1) - 1.
///
/// We then:
///
/// 1. Compute:
///        min_sh = min_i (x_i_sh)
///    using the unconstrained helper `unconstrained_min`.
///
/// 2. Recover the candidate minimum in the original signed range:
///        M = min_sh - S.
///
/// 3. For each i, define:
///        delta_i = x_i - M.
///
///    We enforce that each `delta_i` lies in [0, 2^(s+1) - 1] using
///    the shared `logup_ctx.range_check` with `n_bits = s + 1`.
///
/// 4. Compute:
///        prod = product_{i} delta_i
///    and assert `prod == 0`. As before, with each delta_i constrained to the
///    nonnegative range, the only way the product can be zero is if some
///    delta_i is exactly zero, so some `x_i == M`.
///
/// # Assumptions
/// - Same as `constrained_max`: values encode signed integers in [-S, 2^s - 1]
///   and the field modulus p is large enough to avoid wraparound.
/// - `logup_ctx` is initialized and later finalized by the caller.
///
/// # Arguments
/// - `api`: circuit builder.
/// - `context`: shift parameters (s and S = 2^s).
/// - `logup_ctx`: shared LogUp range-check context.
/// - `values`: nonempty slice of variables.
///
/// # Errors
/// - Returns `LayerError::Other` (wrapped as `CircuitError`) if `values` is
///   empty.
/// - Returns `LayerError::InvalidParameterValue` if `shift_exponent + 1`
///   overflows when computing the bit-length.
/// - Propagates LogUp-related and other circuit errors.
///
/// # Returns
/// - A `Variable` equal to the least nonnegative residue of the minimum value.
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

    // 1) Form offset-shifted values: x_i_sh = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute min_i (x_i_sh), which equals M_sh = M + S
    let min_offset = unconstrained_min(api, &values_offset)?;

    // 3) Recover M = M_sh − S
    let min_raw = api.sub(min_offset, context.offset);

    // 4) For each x_i, range-check delta_i = x_i − M in [0, T] using s + 1 bits
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

        // delta in [0, T] = [0, 2^{s + 1} - 1] via *shared* LogUp-based range proof
        logup_ctx
            .range_check::<C, Builder>(api, delta, n_bits)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Min,
                msg: format!("logup_range_check_pow2_unsigned (LogUp) failed: {e}"),
            })?;

        // Multiply all delta_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: prod delta_i = 0 iff exists x_i such that delta_i = 0 iff x_i = M
    api.assert_is_zero(prod);
    Ok(min_raw)
}

// -----------------------------------------------------------------------------
// FUNCTION: constrained_clip
// -----------------------------------------------------------------------------

/// Enforces `c = clip(x; lower, upper)` using `constrained_max` and
/// `constrained_min` under a shared `ShiftRangeContext`.
///
/// # Semantics
///
/// This matches the ONNX Clip operator for scalar or tensor bounds:
///
/// - If both `lower` and `upper` are present:
///       c = min( max(x, lower), upper )
/// - If only `lower` is present:
///       c = max( x, lower )
/// - If only `upper` is present:
///       c = min( x, upper )
/// - If neither is present:
///       c = x
///
/// # How it works
///
/// Each of `x`, `lower`, and `upper` (if present) is treated as encoding a
/// signed integer in the same range:
///
///     [-S, 2^s - 1],  with  S = 2^s  from `range_ctx`.
///
/// The underlying `constrained_max` / `constrained_min` gadgets:
///
/// - Use the shared `ShiftRangeContext` to shift into [0, 2^(s+1) - 1].
/// - Use LogUp-based range checks (via `logup_ctx`) to constrain differences.
/// - Enforce that the result is one of the inputs (via product-of-differences).
///
/// By composing these gadgets, `constrained_clip` inherits the same guarantees:
/// the returned value is in the allowed range and equals either the original
/// `x` or one of the bounds.
///
/// # Arguments
/// - `api`: circuit builder.
/// - `range_ctx`: shift parameters (s, S = 2^s) shared with max/min gadgets.
/// - `logup_ctx`: shared LogUp context used inside max/min.
/// - `x`: value to be clipped.
/// - `lower`: optional lower bound.
/// - `upper`: optional upper bound.
///
/// # Errors
/// - Propagates errors from `constrained_max` and `constrained_min`.
/// - Propagates LogUp-related and other circuit errors.
///
/// # Returns
/// - A `Variable` equal to `clip(x; lower, upper)` in the signed range
///   convention defined by `range_ctx`.
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
