//! Utilities for quantization-related arithmetic in circuit construction.
//!
//! This module provides functionality for rescaling the product of two fixed-point
//! approximations, where each operand has been independently scaled and rounded.
//! That is, we assume integers `a ≈ α·x` and `b ≈ α·y`, where `α = 2^κ` is the fixed-point
//! scaling factor, and `a`, `b` are scaled versions of real numbers `x`, `y`.
//!
//! The product `c' = a·b` approximates `α²·x·y`. Since the circuit operates only on field
//! elements (nonnegative integers modulo `p`, the field modulus), we must simulate signed
//! integer arithmetic using representative values in the range `[0, p - 1]`.
//!
//! To correctly recover a fixed-point approximation of `x·y` at scale `α`, we compute:
//!
//! ```text
//! q = floor((c + α·S)/α) − S
//! ```
//!
//! where `c` is the least nonnegative residue of the signed integer `c'`,
//! and `S = 2^s` is a centering constant that ensures intermediate values
//! remain within the field during division and remainder operations.
//!
//! The computation enforces that:
//!
//! ```text
//! c + α·S = α·q^♯ + r,    with    0 ≤ r < α
//! ```
//!
//! Then recovers `q = q^♯ − S`, and optionally applies `ReLU` as
//!
//! ```text
//! ReLU(q) = max(q, 0)
//! ```
//!
//! using the existing `constrained_max` gadget, which itself uses LogUp-based
//! range proofs instead of ad-hoc MSB extraction.
//!
//! The core logic is implemented in [`rescale`] and its batched variant
//! [`rescale_array`], using a shared [`RescalingContext`] plus a
//! [`LogupRangeCheckContext`] to precompute constants and optimize performance.

use ethnum::U256;
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::utils::{
    UtilsError,
    errors::{ArrayConversionError, RescaleError},
};

// Internal modules: LogUp-based range-check helper + max gadget
use crate::circuit_functions::gadgets::{
    LogupRangeCheckContext, ShiftRangeContext, constrained_max,
};

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: RescalingContext
// ─────────────────────────────────────────────────────────────────────────────

/// Holds integer and circuit-level constants for rescaling by `α = 2^κ` and shifting by `S = 2^s`.
pub struct RescalingContext {
    pub scaling_exponent: usize, // κ: exponent such that α = 2^κ
    pub shift_exponent: usize,   // s: exponent such that S = 2^s

    pub scaling_factor_: u32, // α = 2^κ, as a native u32
    pub shift_: u32,          // S = 2^s, as a native u32
    pub scaled_shift_: U256,  // α·S = 2^{κ + s}, as a U256 for overflow safety

    pub scaling_factor: Variable, // α = 2^κ, lifted to the circuit as a Variable
    pub shift: Variable,          // S = 2^s, lifted to the circuit as a Variable
    pub scaled_shift: Variable,   // α·S = 2^{κ + s}, lifted to the circuit as a Variable
}

// ─────────────────────────────────────────────────────────────────────────────
// IMPL: RescalingContext
// ─────────────────────────────────────────────────────────────────────────────

impl RescalingContext {
    /// Constructs a [`RescalingContext`] from the given scaling and shift exponents.
    ///
    /// Precomputes:
    /// - Native powers of two:
    ///   - `scaling_factor_ = 2^κ` (`u32`)
    ///   - `shift_ = 2^s` (`u32`)
    ///   - `scaled_shift_ = 2^{κ + s}` (`U256`, to avoid overflow)
    ///
    /// - Circuit-lifted versions of these values:
    ///   - `scaling_factor`, `shift`, and `scaled_shift` (as `Variable`)
    ///
    /// These are reused throughout rescaling to avoid redundant lifting and ensure consistent constraints.
    ///
    /// # Errors
    /// - Returns [`RescaleError::ShiftExponentTooLargeError`] if either exponent does not fit in `u32`.
    /// - Returns [`RescaleError::ScalingExponentTooLargeError`] if shifting `1u32 << κ` overflows.
    /// - Returns [`RescaleError::ShiftExponentTooLargeError`] if shifting `1u32 << s` overflows.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        scaling_exponent: usize,
        shift_exponent: usize,
    ) -> Result<Self, RescaleError> {
        let scaling_exponent_u32 = u32::try_from(scaling_exponent).map_err(|_| {
            RescaleError::ShiftExponentTooLargeError {
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
        let shift_ = 1u32.checked_shl(shift_exponent_u32).ok_or(
            RescaleError::ShiftExponentTooLargeError {
                exp: scaling_exponent,
                type_name: "u32",
            },
        )?;
        let scaled_shift_ = U256::from(scaling_factor_) * U256::from(shift_); // α·S = 2^{κ + s}

        let scaling_factor = api.constant(scaling_factor_); // α as Variable
        let shift = api.constant(shift_); // S as Variable
        let scaled_shift = api.constant(CircuitField::<C>::from_u256(scaled_shift_)); // α·S as Variable

        Ok(Self {
            scaling_exponent, // κ
            shift_exponent,   // s
            scaling_factor_,  // α = 2^κ
            shift_,           // S = 2^s
            scaled_shift_,    // α·S = 2^{κ + s}
            scaling_factor,   // α as Variable
            shift,            // S as Variable
            scaled_shift,     // α·S as Variable
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: rescale
// ─────────────────────────────────────────────────────────────────────────────

/// Computes `q = floor((c + α·S)/α) − S`, optionally applying `ReLU`, using a
/// precomputed [`RescalingContext`] for efficiency and clarity, plus a shared
/// [`LogupRangeCheckContext`] for range proofs.
///
/// All range checks are implemented via LogUp (no direct bit-decomp gadgets here),
/// and `ReLU` is implemented as `max(q, 0)` via the existing `constrained_max`
/// gadget, which itself uses LogUp-based range-checking and a product-of-differences
/// condition to ensure the selected maximum is one of the inputs.
///
/// # Notation
/// - Let `κ = context.scaling_exponent`, and define `α = 2^κ`.
/// - Let `s = context.shift_exponent`, and define `S = 2^s`.
/// - Define `T = 2·S − 1 = 2^(s + 1) − 1`.
/// - `c` is the input `dividend`.
/// - `r` is the remainder.
/// - `q^♯` is the offset quotient: `q^♯ = q + S`.
///
/// # Process
/// 1. Form `shifted_dividend = α·S + c` using precomputed constants from `context`.
/// 2. Unconstrained division: `shifted_dividend = α·q^♯ + r`.
/// 3. Enforce this equality with a constraint.
/// 4. LogUp range-check `r ∈ [0, α − 1]` using `κ` bits.
/// 5. LogUp range-check `q^♯ ∈ [0, T] = [0, 2^(s + 1) − 1]`.
/// 6. Recover `q = q^♯ − S`.
/// 7. If `apply_relu`, output `max(q, 0)` via `constrained_max`.
pub fn rescale<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &RescalingContext,
    logup_ctx: &mut LogupRangeCheckContext,
    dividend: Variable,
    apply_relu: bool,
) -> Result<Variable, RescaleError> {
    // Step 1: compute shifted_dividend = α·S + c
    let shifted_dividend = api.add(context.scaled_shift, dividend);

    // Step 2: Compute unchecked witness values q^♯, r via unconstrained Euclidean division:
    //         α·S + c = α·q^♯ + r
    let shifted_q = api.unconstrained_int_div(shifted_dividend, context.scaling_factor_); // q^♯
    let remainder = api.unconstrained_mod(shifted_dividend, context.scaling_factor_); // r

    // Step 3: Enforce α·S + c = α·q^♯ + r
    let rhs_first_term = api.mul(context.scaling_factor, shifted_q);
    let rhs = api.add(rhs_first_term, remainder);
    api.assert_is_equal(shifted_dividend, rhs);

    // Step 4: LogUp range-check r ∈ [0, α − 1] using κ bits
    logup_ctx
        .range_check::<C, Builder>(api, remainder, context.scaling_exponent)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("remainder (LogUp): {e}"),
            n_bits: context.scaling_exponent,
        })?;

    // Step 5: LogUp range-check q^♯ ∈ [0, 2^(s + 1) − 1] using s + 1 bits
    let n_bits_q =
        context
            .shift_exponent
            .checked_add(1)
            .ok_or(RescaleError::ShiftExponentTooLargeError {
                exp: context.shift_exponent,
                type_name: "usize",
            })?;

    logup_ctx
        .range_check::<C, Builder>(api, shifted_q, n_bits_q)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("quotient (LogUp): {e}"),
            n_bits: n_bits_q,
        })?;

    // Step 6: Recover quotient q = q^♯ − S
    let quotient = api.sub(shifted_q, context.shift); // q = q^♯ − S

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

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: rescale_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies `rescale` elementwise to an `ArrayD<Variable>`, using provided scaling and shift exponents.
///
/// Internally constructs:
/// - a [`RescalingContext`] with the given exponents:
///   - `scaling_exponent` κ such that α = 2^κ
///   - `shift_exponent` s such that S = 2^s
/// - a shared [`LogupRangeCheckContext`] used for **all** range checks (remainder and quotient
///   range proofs, plus any `constrained_max` calls for ReLU).
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder.
/// - `array`: A tensor (of any shape) of `Variable`s to rescale.
/// - `scaling_exponent`: κ for scaling by 2^κ.
/// - `shift_exponent`: s for shifting by 2^s.
/// - `apply_relu`: Whether to apply `ReLU` after rescaling.
///
/// # Returns
/// An `ArrayD<Variable>` of the same shape with all values rescaled.
///
/// # Errors
/// - Returns [`UtilsError::from(RescaleError)`] if rescaling fails for any element.
/// - Returns [`ArrayConversionError::ShapeError`] if the reshaped array cannot be reconstructed
///   from the rescaled data.
/// - Propagates any errors from [`RescalingContext::new`], such as overflow in the exponent shifts.
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
