//! Unconstrained bit-manipulation helpers.
//!
//! These helpers are *not* soundness-critical — they are witnesses only.
//! Used only for legacy or experimental bit-decomposition flows.

/// External crate imports
use ethnum::U256;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{CircuitField, Config, RootAPI, Variable};

// Trait giving CircuitField::MODULUS
use expander_compiler::field::FieldArith;

/// Internal crate imports
use crate::circuit_functions::{CircuitError, utils::UtilsError};

// -----------------------------------------------------------------------------
// FUNCTION: unconstrained_to_bits
// -----------------------------------------------------------------------------

/// Extracts the `n_bits` least significant bits of a field element, using
/// *unconstrained* bit operations, and returns them in **little-endian order**.
///
/// This is a lightweight helper used only when no constraints are required on
/// the bit pattern. It is *not* a sound range-check: higher bits of the value are
/// simply discarded.
///
/// # Overview
///
/// Given an input field element `x`, the function repeatedly:
///
/// 1. Computes `bit_0 = x AND 1` using `unconstrained_bit_and`.
/// 2. Appends `bit_0` to the output list.
/// 3. Updates `x = x >> 1` using `unconstrained_shift_r`.
///
/// After `n_bits` iterations, the result is:
///
///     bits[0] = least significant bit of input
///     bits[1] = next bit
///     ...
///     bits[n_bits - 1] = most significant of the extracted bits
///
/// This helper mirrors a CPU right-shift loop, but none of the bits are enforced
/// to be boolean and no reconstruction constraint is added. If soundness is
/// required, callers must pair this with `constrained_reconstruct_from_bits`
/// (or with LogUp-based range checks).
///
/// # Arguments
///
/// - `api`: the circuit builder, providing unconstrained bitwise operations.
/// - `input`: value from which the `n_bits` LSBs will be extracted.
/// - `n_bits`: number of least significant bits to extract.
///
/// # Returns
///
/// A `Vec<Variable>` of length `n_bits`, storing the least significant bits of
/// `input` in little-endian order.
///
/// # Errors
///
/// - Returns `CircuitError::Other` if `n_bits == 0`.
/// - Returns `UtilsError::ValueTooLarge` if `n_bits` does not fit in `u32`.
/// - Returns `CircuitError::Other` if `2^n_bits >= MODULUS/2`.
///   (This guards against extracting more bits than make sense for the field.)
///
/// # Example
///
/// For `input = 43` (binary `101011`) and `n_bits = 4`, the function returns:
///
///     [1, 1, 0, 1]
///
/// corresponding to the 4 least significant bits `1011` in little-endian form.
pub fn unconstrained_to_bits<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input: Variable,
    n_bits: usize,
) -> Result<Vec<Variable>, CircuitError> {
    if n_bits == 0 {
        return Err(CircuitError::Other("Cannot convert to 0 bits".into()));
    }
    let base: U256 = U256::from(2u32);
    if base.pow(
        u32::try_from(n_bits).map_err(|_| UtilsError::ValueTooLarge {
            value: n_bits,
            max: u128::from(u32::MAX),
        })?,
    ) >= (CircuitField::<C>::MODULUS / 2)
    {
        return Err(CircuitError::Other(
            "unconstrained_to_bits: n_bits too large (require 2^n_bits < MODULUS/2)".into(),
        ));
    }

    let mut least_significant_bits = Vec::with_capacity(n_bits);
    let mut current = input;

    for _ in 0..n_bits {
        // Extract bit 0 of `current`
        let bit = api.unconstrained_bit_and(current, 1u32);
        least_significant_bits.push(bit);
        // Shift right by one
        current = api.unconstrained_shift_r(current, 1u32);
    }

    Ok(least_significant_bits)
}
