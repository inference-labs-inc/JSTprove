use expander_compiler::frontend::{Config, RootAPI, Variable};

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
    // Step 1: compute shifted_dividend = alpha*S + c
    let shifted_dividend = api.add(divisor_by_shift, dividend);

    // Step 2: Compute unchecked witness values q_shifted, r via unconstrained Euclidean division:
    //         alpha*S + c = alpha*q_shifted + r
    let shifted_q = api.unconstrained_int_div(shifted_dividend, divisor); // q_shifted
    let remainder = api.unconstrained_mod(shifted_dividend, divisor); // r

    // Step 3: Enforce alpha*S + c = alpha*q_shifted + r
    let rhs_first_term = api.mul(divisor, shifted_q);
    let rhs = api.add(rhs_first_term, remainder);
    api.assert_is_equal(shifted_dividend, rhs);

    // Step 4: LogUp range-check r in [0, alpha − 1] using kappa bits
    logup_ctx
        .range_check::<C, Builder>(api, remainder, divisor_exponent)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("remainder (LogUp): {e}"),
            n_bits: divisor_exponent,
        })?;

    // Step 5: LogUp range-check q_shifted in [0, 2^(s + 1) − 1] using s + 1 bits
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

    // Step 6: Recover quotient q = q_shifted − S
    let quotient = api.sub(shifted_q, shift); // q = q_shifted − S
    Ok(quotient)
}
