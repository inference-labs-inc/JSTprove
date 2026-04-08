use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{gadgets::LogupRangeCheckContext, utils::RescaleError};

/// Performs division of a positive integer variable by a **power-of-two constant**
/// inside the circuit, with range-checked quotient and remainder.
///
///
/// The division is implemented by introducing *unconstrained witnesses* for the
/// intermediate quotient and remainder, and then enforcing correctness via
/// algebraic constraints and LogUp range checks.
///
/// # Parameters
///
/// - `api`: Circuit builder API used to create variables and constraints.
/// - `logup_ctx`: LogUp context used to range-check the quotient and remainder.
/// - `dividend`: The dividend variable `c`.
/// - `divisor`: The divisor variable `alpha`, which **must be a power of two**.
/// - `divisor_by_shift`: The precomputed constant `alpha·S`.
/// - `divisor_exponent`: Exponent such that `alpha = 2^{divisor_exponent}`.
/// - `shift_exponent`: Exponent such that `S = 2^{shift_exponent}`.
/// - `shift`: The variable representing `S`.
///
/// # Returns
///
/// Returns the quotient variable `q` corresponding to division of `dividend`
/// by `divisor`, adjusted by the rescaling shift.
///
/// # Errors
///
/// Returns [`RescaleError`] if:
///
/// - The LogUp range check for the remainder fails.
/// - The LogUp range check for the quotient fails.
/// - `shift_exponent + 1` overflows `usize`.
///
/// # Preconditions
///
/// - `divisor` **must** represent a positive power of two.
/// - `divisor_by_shift` **must equal** `divisor * shift`.
/// - `logup_ctx` must be initialized before calling this function.
/// - All inputs must be non-negative and fit within the expected bit widths.
///
/// # Notes
///
/// This function does **not** itself verify that `divisor` is a power of two or
/// that `divisor_by_shift = divisor * shift`; these are assumed to be enforced
/// by the caller.
#[allow(clippy::too_many_arguments)]
pub fn div_pos_integer_pow2_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    logup_ctx: &mut LogupRangeCheckContext,
    dividend: Variable,
    divisor: Variable,
    divisor_by_shift: Variable,
    divisor_exponent: usize,
    shift_exponent: usize,
    shift: Variable,
) -> Result<Variable, RescaleError> {
    div_pos_integer_pow2_constant_inner::<C, Builder>(
        api,
        logup_ctx,
        dividend,
        divisor,
        divisor_by_shift,
        divisor_exponent,
        shift_exponent,
        shift,
        true,
    )
}

/// Like [`div_pos_integer_pow2_constant`], but optionally skips the quotient
/// range check when `range_check_quotient` is `false`.
///
/// Callers that omit the quotient range check **must** ensure that a
/// downstream constraint (e.g. a fused `ReLU` range proof) bounds the
/// quotient; otherwise the division is unsound.
///
/// # Errors
///
/// Returns [`RescaleError`] if any `LogUp` range check or overflow guard fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn div_pos_integer_pow2_constant_inner<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    logup_ctx: &mut LogupRangeCheckContext,
    dividend: Variable,
    divisor: Variable,
    divisor_by_shift: Variable,
    divisor_exponent: usize,
    shift_exponent: usize,
    shift: Variable,
    range_check_quotient: bool,
) -> Result<Variable, RescaleError> {
    let shifted_dividend = api.add(divisor_by_shift, dividend);

    let shifted_q = api.unconstrained_int_div(shifted_dividend, divisor);
    let remainder = api.unconstrained_mod(shifted_dividend, divisor);

    let rhs_first_term = api.mul(divisor, shifted_q);
    let rhs = api.add(rhs_first_term, remainder);
    api.assert_is_equal(shifted_dividend, rhs);

    logup_ctx
        .range_check::<C, Builder>(api, remainder, divisor_exponent)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("remainder (LogUp): {e}"),
            n_bits: divisor_exponent,
        })?;

    if range_check_quotient {
        let n_bits_q =
            shift_exponent
                .checked_add(1)
                .ok_or(RescaleError::ShiftExponentTooLargeError {
                    exp: shift_exponent,
                    type_name: "usize",
                })?;

        logup_ctx
            .range_check::<C, Builder>(api, shifted_q, n_bits_q)
            .map_err(|e| RescaleError::BitDecompositionError {
                var_name: format!("quotient (LogUp): {e}"),
                n_bits: n_bits_q,
            })?;
    }

    let quotient = api.sub(shifted_q, shift);
    Ok(quotient)
}

/// Performs division of a positive integer variable by an **arbitrary positive
/// constant** inside the circuit, with range-checked quotient and remainder.
///
/// When `divisor_val == 1`, returns `dividend` directly (identity division).
/// For divisors that are powers of two this delegates to
/// [`div_pos_integer_pow2_constant`]. For other divisors, the remainder
/// constraint `0 <= r < d` is enforced by range-checking both `r` and
/// `d - 1 - r` in `ceil(log2(d))` bits.
///
/// # Errors
///
/// Returns [`RescaleError::InvalidDivisor`] if `divisor_val` is zero.
/// Returns [`RescaleError`] if any LogUp range check or overflow guard fails.
#[allow(clippy::too_many_arguments)]
pub fn div_pos_integer_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    logup_ctx: &mut LogupRangeCheckContext,
    dividend: Variable,
    divisor: Variable,
    divisor_val: u32,
    divisor_by_shift: Variable,
    shift_exponent: usize,
    shift: Variable,
) -> Result<Variable, RescaleError> {
    if divisor_val == 0 {
        return Err(RescaleError::InvalidDivisor { divisor_val });
    }

    if divisor_val == 1 {
        return Ok(dividend);
    }

    if divisor_val.is_power_of_two() {
        let divisor_exponent = divisor_val.trailing_zeros() as usize;
        return div_pos_integer_pow2_constant::<C, Builder>(
            api,
            logup_ctx,
            dividend,
            divisor,
            divisor_by_shift,
            divisor_exponent,
            shift_exponent,
            shift,
        );
    }

    let shifted_dividend = api.add(divisor_by_shift, dividend);

    let shifted_q = api.unconstrained_int_div(shifted_dividend, divisor);
    let remainder = api.unconstrained_mod(shifted_dividend, divisor);

    let rhs_first_term = api.mul(divisor, shifted_q);
    let rhs = api.add(rhs_first_term, remainder);
    api.assert_is_equal(shifted_dividend, rhs);

    let remainder_bits = (u32::BITS - (divisor_val - 1).leading_zeros()) as usize;

    logup_ctx
        .range_check::<C, Builder>(api, remainder, remainder_bits)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("remainder (LogUp): {e}"),
            n_bits: remainder_bits,
        })?;

    let d_minus_1 = api.constant(CircuitField::<C>::from_u256(ethnum::U256::from(
        divisor_val - 1,
    )));
    let d_minus_1_minus_r = api.sub(d_minus_1, remainder);
    logup_ctx
        .range_check::<C, Builder>(api, d_minus_1_minus_r, remainder_bits)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("d-1-remainder (LogUp): {e}"),
            n_bits: remainder_bits,
        })?;

    let n_bits_q =
        shift_exponent
            .checked_add(1)
            .ok_or(RescaleError::ShiftExponentTooLargeError {
                exp: shift_exponent,
                type_name: "usize",
            })?;

    logup_ctx
        .range_check::<C, Builder>(api, shifted_q, n_bits_q)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("quotient (LogUp): {e}"),
            n_bits: n_bits_q,
        })?;

    let quotient = api.sub(shifted_q, shift);
    Ok(quotient)
}
