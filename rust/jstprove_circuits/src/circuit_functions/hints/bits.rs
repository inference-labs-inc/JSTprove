use ethnum::U256;

/// Expander / circuit frontend
use expander_compiler::frontend::{CircuitField, Config, RootAPI, Variable};

// Trait that provides MODULUS for CircuitField
use expander_compiler::field::FieldArith;

use crate::circuit_functions::{CircuitError, utils::UtilsError};

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_to_bits
// ─────────────────────────────────────────────────────────────────────────────

/// Extracts the least significant `n_bits` of a field element as a bitstring, in little-endian order.
///
/// # Overview
/// Uses the Expander Compiler Collection’s unconstrained bitwise operations to extract
/// the `n_bits` least significant bits of a `Variable`. The bits are returned in little-endian order:
/// `bits[0]` is the least significant bit, `bits[n_bits - 1]` is the most significant of the truncated bits.
///
/// This function does **not** check that the input fits within `n_bits`; any higher-order bits are discarded.
///
/// # Type Parameters
/// - `C`: Circuit configuration implementing `Config`.
/// - `Builder`: Prover API implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder.
/// - `input`: A `Variable` representing the integer or field element to be bit-decomposed.
/// - `n_bits`: Number of least significant bits to extract.
///
/// # Returns
/// A vector of `n_bits` `Variable`s representing the bit decomposition of `input`,
/// in little-endian order.
///
/// # Errors
/// - [`CircuitError::Other`] if `n_bits == 0`.
/// - [`UtilsError::ValueTooLarge`] if `n_bits` cannot fit into a `u32`.
/// - [`CircuitError::Other`] if `2^n_bits >= MODULUS/2`, where `MODULUS` is the circuit field modulus.
///
/// # Example
/// ```ignore
/// // For input = 43 and n_bits = 4:
/// // Returns [1, 1, 0, 1], since 43 = 0b101011, and the 4 LSBs are 1011.
/// ```
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
        return Err(CircuitError::Other("n_bits must be ".into()));
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
