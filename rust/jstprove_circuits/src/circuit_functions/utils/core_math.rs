use ethnum::U256;
/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

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

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: assert_is_bitstring_and_reconstruct
// ─────────────────────────────────────────────────────────────────────────────

/// Checks that each element of a little-endian bitstring is in `{0,1}` and reconstructs the integer.
///
/// # Overview
/// For a given slice of variables `[b₀, b₁, ..., bₙ₋₁]` representing a bitstring in little-endian order,
/// this function:
/// 1. Enforces that each `bᵢ ∈ {0,1}` via the constraint `bᵢ(bᵢ − 1) = 0`.
/// 2. Reconstructs the integer `∑ bᵢ·2ⁱ` and returns the corresponding `Variable`.
///
/// This function panics if any shift `2ⁱ` for `i ≥ 32` overflows a `u32`.
///
/// # Arguments
/// - `api`: A mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `least_significant_bits`: A slice of `Variable`s representing a bitstring in little-endian order.
///
/// # Errors
/// - [`UtilsError::ValueTooLarge`] if the bit index `i` cannot be converted to `u32`.
/// - [`UtilsError::ValueTooLarge`] if computing `2^i` overflows a `u32` (i ≥ 32).
///
/// # Returns
/// A `Variable` encoding the integer reconstructed from the bitstring.
///
/// # Example
/// ```ignore
/// // For bits = [1, 1, 0, 1], returns 11,
/// // since 1·2⁰ + 1·2¹ + 0·2² + 1·2³ = 11.
/// ```
pub fn assert_is_bitstring_and_reconstruct<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    least_significant_bits: &[Variable],
) -> Result<Variable, CircuitError> {
    // Start with 0 and accumulate ∑ bᵢ·2ⁱ as we iterate
    let mut reconstructed = api.constant(0u32);

    for (i, &bit) in least_significant_bits.iter().enumerate() {
        // Enforce bᵢ ∈ {0, 1} via b(b − 1) = 0
        api.assert_is_bool(bit);
        // Compute bᵢ · 2ⁱ

        let weight = 1u32
            .checked_shl(u32::try_from(i).map_err(|_| UtilsError::ValueTooLarge {
                value: i,
                max: u128::from(u32::MAX),
            })?)
            .ok_or(UtilsError::ValueTooLarge {
                value: i,
                max: u128::from(u32::MAX),
            })?;
        let weight_const = api.constant(weight);
        let term = api.mul(weight_const, bit);
        reconstructed = api.add(reconstructed, term);
    }

    Ok(reconstructed)
}
