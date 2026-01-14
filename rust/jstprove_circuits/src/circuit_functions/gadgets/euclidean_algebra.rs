use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{gadgets::LogupRangeCheckContext, utils::RescaleError};

#[allow(clippy::missing_errors_doc)]
#[allow(clippy::too_many_arguments)]
pub fn div_pos_integer_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    logup_ctx: &mut LogupRangeCheckContext,
    dividend: Variable,
    divisor: Variable,
    scaled_shift: Variable,
    scaling_exponent: usize,
    shift_exponent: usize,
    shift: Variable,
) -> Result<Variable, RescaleError> {
    // Step 1: compute shifted_dividend = alpha*S + c
    let shifted_dividend = api.add(scaled_shift, dividend);

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
        .range_check::<C, Builder>(api, remainder, scaling_exponent)
        .map_err(|e| RescaleError::BitDecompositionError {
            var_name: format!("remainder (LogUp): {e}"),
            n_bits: scaling_exponent,
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
