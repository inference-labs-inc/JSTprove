//! Utilities for quantization-related arithmetic in circuit construction.
//!
//! This module provides functionality for rescaling the product of two fixed-point
//! approximations, where each operand has been independently scaled and rounded.
//! We assume integers `a ≈ alpha * x` and `b ≈ alpha * y`, where `alpha = 2^kappa` is the fixed-point
//! scaling factor, and `a`, `b` are scaled versions of real numbers `x`, `y`.
//!
//! The product `c' = a * b` approximates `alpha^2 * x * y`. Since the circuit operates
//! only on field elements (nonnegative integers modulo `p`, the field modulus), we
//! simulate signed integer arithmetic using representatives in `[0, p - 1]`.
//!
//! To recover a fixed-point approximation of `x * y` at scale `alpha`, we compute
//!
//! ```text
//! q = floor((c + alpha * S) / alpha) - S
//! ```
//!
//! where `c` is the least nonnegative residue of the signed integer `c'` and
//! `S = 2^s` is a centering constant that keeps intermediate values inside the
//! field during division and remainder operations.
//!
//! The computation enforces
//!
//! ```text
//! c + alpha * S = alpha * q_shifted + r,    with    0 <= r < alpha
//! ```
//!
//! then recovers `q = q_shifted - S`, and optionally applies `ReLU` as
//!
//! ```text
//! ReLU(q) = max(q, 0)
//! ```
//!
//! using the existing `constrained_max` gadget, which itself relies on `LogUp`-based
//! range proofs instead of ad hoc most-significant-bit extraction.
//!
//! The core logic is implemented in [`rescale`] and its batched variant
//! [`rescale_array`], using a shared [`RescalingContext`] plus a
//! [`LogupRangeCheckContext`] to precompute constants and optimize performance.

use ethnum::U256;
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::euclidean_division::div_pos_integer_pow2_constant,
    layers::{LayerError, LayerKind},
    utils::{
        UtilsError,
        errors::{ArrayConversionError, RescaleError},
    },
};

// Internal modules: LogUp-based range-check helper + max gadget
use crate::circuit_functions::gadgets::{
    LogupRangeCheckContext, ShiftRangeContext, constrained_max,
};

// -----------------------------------------------------------------------------
// STRUCT: RescalingContext
// -----------------------------------------------------------------------------

/// Holds integer and circuit-level constants for rescaling by `alpha = 2^kappa`
/// and shifting by `S = 2^s`.
///
/// This context packages both native integers and circuit `Variable`s:
///
/// - `scaling_exponent` (`kappa`): exponent such that `alpha = 2^kappa`.
/// - `shift_exponent` (`s`): exponent such that `S = 2^s`.
/// - `scaling_factor_`: native `u32` equal to `alpha`.
/// - `shift_`: native `u64` equal to `S`.
/// - `scaled_shift_`: native `U256` equal to `alpha * S = 2^(kappa + s)`.
/// - `scaling_factor`, `shift`, `scaled_shift`: the same quantities lifted
///   into the circuit as `Variable`s.
///
/// These values are reused throughout rescaling to avoid recomputing constants
/// and to keep constraints consistent.
pub struct RescalingContext {
    pub scaling_exponent: usize, // kappa: exponent such that alpha = 2^kappa
    pub shift_exponent: usize,   // s: exponent such that S = 2^s

    pub scaling_factor_: u32, // alpha = 2^kappa, as a native u32
    pub shift_: u64,          // S = 2^s, as a native u64 (supports n_bits > 32)
    pub scaled_shift_: U256,  // alpha*S = 2^{kappa + s}, as a U256 for overflow safety

    pub scaling_factor: Variable, // alpha = 2^kappa, lifted to the circuit as a Variable
    pub shift: Variable,          // S = 2^s, lifted to the circuit as a Variable
    pub scaled_shift: Variable,   // alpha*S = 2^{kappa + s}, lifted to the circuit as a Variable
}

// -----------------------------------------------------------------------------
// IMPL: RescalingContext
// -----------------------------------------------------------------------------

impl RescalingContext {
    /// Construct a [`RescalingContext`] from the given scaling and shift exponents.
    ///
    /// This method precomputes:
    ///
    /// - Native powers of two:
    ///   - `scaling_factor_ = 2^kappa` as `u32`.
    ///   - `shift_ = 2^s` as `u64`.
    ///   - `scaled_shift_ = 2^(kappa + s)` as `U256` (to avoid overflow).
    /// - Circuit-lifted versions of these values:
    ///   - `scaling_factor`, `shift`, `scaled_shift` as `Variable`s.
    ///
    /// These constants are reused by [`rescale`] to keep the arithmetic and
    /// range checks consistent.
    ///
    /// # Errors
    ///
    /// - Returns [`RescaleError::ShiftExponentTooLargeError`] if either exponent
    ///   does not fit in `u32`.
    /// - Returns [`RescaleError::ScalingExponentTooLargeError`] if computing
    ///   `1u32 << scaling_exponent` overflows `u32`.
    /// - Returns [`RescaleError::ShiftExponentTooLargeError`] if computing
    ///   `1u64 << shift_exponent` overflows.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        scaling_exponent: usize,
        shift_exponent: usize,
    ) -> Result<Self, RescaleError> {
        let scaling_exponent_u32 = u32::try_from(scaling_exponent).map_err(|_| {
            RescaleError::ScalingExponentTooLargeError {
                exp: scaling_exponent,
                type_name: "u32",
            }
        })?;
        let scaling_factor_ = 1u32.checked_shl(scaling_exponent_u32).ok_or(
            RescaleError::ScalingExponentTooLargeError {
                exp: scaling_exponent,
                type_name: "u32",
            },
        )?;
        let shift_exponent_u32 = u32::try_from(shift_exponent).map_err(|_| {
            RescaleError::ShiftExponentTooLargeError {
                exp: shift_exponent,
                type_name: "u32",
            }
        })?;
        let shift_ = 1u64.checked_shl(shift_exponent_u32).ok_or(
            RescaleError::ShiftExponentTooLargeError {
                exp: shift_exponent,
                type_name: "u64",
            },
        )?;
        let scaled_shift_ = U256::from(scaling_factor_) * U256::from(shift_);

        let scaling_factor = api.constant(scaling_factor_);
        let shift = api.constant(CircuitField::<C>::from_u256(U256::from(shift_)));
        let scaled_shift = api.constant(CircuitField::<C>::from_u256(scaled_shift_));

        Ok(Self {
            scaling_exponent, // kappa
            shift_exponent,   // s
            scaling_factor_,  // alpha = 2^kappa
            shift_,           // S = 2^s
            scaled_shift_,    // alpha*S = 2^{kappa + s}
            scaling_factor,   // alpha as Variable
            shift,            // S as Variable
            scaled_shift,     // alpha*S as Variable
        })
    }
}

// -----------------------------------------------------------------------------
// FUNCTION: rescale
// -----------------------------------------------------------------------------

/// Compute `q = floor((c + alpha * S) / alpha) - S`, optionally followed by `ReLU`,
/// using a precomputed [`RescalingContext`] and a shared
/// [`LogupRangeCheckContext`].
///
/// All range checks are implemented via `LogUp` (no direct bit-decomposition
/// gadget here), and `ReLU` is implemented as `max(q, 0)` via the existing
/// `constrained_max` gadget, which itself uses `LogUp`-based range checks and
/// a product-of-differences argument to prove correctness.
///
/// # Notation
///
/// - `kappa = context.scaling_exponent` and `alpha = 2^kappa`.
/// - `s = context.shift_exponent` and `S = 2^s`.
/// - `T = 2 * S - 1 = 2^(s + 1) - 1`.
/// - `c` is the input `dividend`.
/// - `q_shifted` is the intermediate quotient `q + S`.
/// - `r` is the remainder.
///
/// # Process (high level)
///
/// 1. Form `shifted_dividend = alpha * S + c` using constants from `context`.
/// 2. Use unconstrained division to compute witnesses `q_shifted` and `r`:
///    `shifted_dividend = alpha * q_shifted + r`.
/// 3. Enforce this equality as a circuit constraint.
/// 4. Use `LogUp` to range-check `r` in `[0, alpha - 1]` using `kappa` bits.
/// 5. Use `LogUp` to range-check `q_shifted` in `[0, T] = [0, 2^(s + 1) - 1]`.
/// 6. Recover `q = q_shifted - S`.
/// 7. If `apply_relu` is `true`, compute `max(q, 0)` using `constrained_max`.
///
/// # Arguments
///
/// - `api`: mutable reference to the circuit builder.
/// - `context`: precomputed [`RescalingContext`] carrying `alpha`, `S`, and `alpha * S`.
/// - `logup_ctx`: shared [`LogupRangeCheckContext`] reused for all range checks.
/// - `dividend`: the input `Variable` representing `c`.
/// - `apply_relu`: if `true`, applies `ReLU` after rescaling.
///
/// # Errors
///
/// - Returns [`RescaleError`] if exponent handling or internal range checks fail,
///   for example:
///   - `RescaleError::ShiftExponentTooLargeError` or
///     `RescaleError::ScalingExponentTooLargeError` when deriving bit-widths.
///   - `RescaleError::BitDecompositionError` when `LogUp` range checks fail,
///     or when `ShiftRangeContext::new` or `constrained_max` fail inside the
///     `ReLU` branch.
pub fn rescale<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &RescalingContext,
    logup_ctx: &mut LogupRangeCheckContext,
    dividend: Variable,
    apply_relu: bool,
) -> Result<Variable, RescaleError> {
    let quotient = div_pos_integer_pow2_constant(
        api,
        logup_ctx,
        dividend,
        context.scaling_factor,
        context.scaled_shift,
        context.scaling_exponent,
        context.shift_exponent,
        context.shift,
    )?;

    // Step 7: If ReLU is applied, enforce ReLU(q) = max(q, 0) via constrained_max
    if apply_relu {
        // Build shift context for the max gadget (same `s` as in RescalingContext)
        let shift_ctx = ShiftRangeContext::new(api, context.shift_exponent).map_err(|e| {
            RescaleError::BitDecompositionError {
                var_name: format!("ShiftRangeContext::new in ReLU: {e}"),
                n_bits: context.shift_exponent + 1,
            }
        })?;

        let zero = api.constant(0);
        let relu_q = constrained_max::<C, Builder>(api, &shift_ctx, logup_ctx, &[quotient, zero])
            .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("ReLU via constrained_max failed: {e}"),
            n_bits: context.shift_exponent + 1,
        })?;

        Ok(relu_q)
    } else {
        Ok(quotient)
    }
}

// -----------------------------------------------------------------------------
// FUNCTION: rescale_array
// -----------------------------------------------------------------------------

/// Apply [`rescale`] elementwise to an `ArrayD<Variable>`, using the provided
/// scaling and shift exponents.
///
/// Internally, this function:
///
/// - Constructs a [`RescalingContext`] with:
///   - `scaling_exponent` `kappa` such that `alpha = 2^kappa`.
///   - `shift_exponent` `s` such that `S = 2^s`.
/// - Constructs a shared [`LogupRangeCheckContext`] used for **all** range checks
///   (both remainder and quotient range proofs, plus any `constrained_max` calls
///   used to implement `ReLU`).
/// - Flattens the input array, applies [`rescale`] to each element, and rebuilds
///   an `ArrayD<Variable>` of the same shape.
///
/// # Arguments
///
/// - `api`: mutable reference to the circuit builder.
/// - `array`: tensor (any shape) of `Variable`s to be rescaled.
/// - `scaling_exponent`: `kappa` such that `alpha = 2^kappa`.
/// - `shift_exponent`: `s` such that `S = 2^s`.
/// - `apply_relu`: if `true`, applies `ReLU` after rescaling each element.
///
/// # Returns
///
/// An `ArrayD<Variable>` of the same shape as `array`, with all entries
/// rescaled (and optionally passed through `ReLU`).
///
/// # Errors
///
/// - Returns [`UtilsError`] converted from [`RescaleError`] if rescaling fails
///   for any element.
/// - Returns [`ArrayConversionError::ShapeError`] if reconstructing the array
///   from the flattened data fails.
/// - Propagates any errors from [`RescalingContext::new`] (for example, when
///   exponent shifts overflow).
pub fn rescale_array<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    array: ArrayD<Variable>,
    scaling_exponent: usize,
    shift_exponent: usize,
    apply_relu: bool,
) -> Result<ArrayD<Variable>, UtilsError> {
    let context = RescalingContext::new(api, scaling_exponent, shift_exponent)?;

    // Shared LogUp table for all range checks in this rescale pass
    let mut logup_ctx = LogupRangeCheckContext::new_default();
    logup_ctx.init::<C, Builder>(api);

    // Convert to Vec, map with error handling, then back to ArrayD
    let shape = array.shape().to_vec();
    let results: Result<Vec<Variable>, RescaleError> = array
        .into_iter()
        .map(|x| rescale::<C, Builder>(api, &context, &mut logup_ctx, x, apply_relu))
        .collect();

    // Single final LogUp consistency check for all queries
    logup_ctx.finalize::<C, Builder>(api);

    let rescaled_data = results?;
    Ok(ArrayD::from_shape_vec(shape, rescaled_data).map_err(ArrayConversionError::ShapeError)?)
}

pub struct MaybeRescaleParams {
    pub is_rescale: bool,
    pub scaling_exponent: u64,
    pub n_bits: usize,
    pub is_relu: bool,
    pub layer_kind: LayerKind,
    pub layer_name: String,
}

/// Conditionally rescale an array when `params.is_rescale` is true.
///
/// Converts `scaling` (u64) to a `usize` scaling exponent, derives the shift
/// exponent as `n_bits - 1`, and delegates to [`rescale_array`].  Returns the
/// input unchanged when `is_rescale` is false.
///
/// # Errors
///
/// Returns [`LayerError::Other`] if `scaling` cannot be represented as `usize`.
/// Returns [`LayerError::InvalidParameterValue`] if `n_bits` is zero.
/// Propagates errors from [`rescale_array`].
pub fn maybe_rescale<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    result: ArrayD<Variable>,
    params: &MaybeRescaleParams,
) -> Result<ArrayD<Variable>, CircuitError> {
    if !params.is_rescale {
        return Ok(result);
    }

    let scaling_exponent =
        usize::try_from(params.scaling_exponent).map_err(|_| LayerError::Other {
            layer: params.layer_kind.clone(),
            msg: "Cannot convert scaling to usize".to_string(),
        })?;
    let shift_exponent =
        params
            .n_bits
            .checked_sub(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: params.layer_kind.clone(),
                layer_name: params.layer_name.clone(),
                param_name: "n_bits".to_string(),
                value: params.n_bits.to_string(),
            })?;

    Ok(rescale_array(
        api,
        result,
        scaling_exponent,
        shift_exponent,
        params.is_relu,
    )?)
}
