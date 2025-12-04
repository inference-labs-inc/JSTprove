//! Range-check gadgets for fixed-point arithmetic:
//! - generic bit-decomposition-based range check,
//! - LogUp-based range check with sharable context,
//! - helper gadget for reconstructing integers from boolean bitstrings.

/// External crate imports
use circuit_std_rs::logup::LogUpRangeProofTable;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::CircuitError;
use crate::circuit_functions::utils::UtilsError;

// Unconstrained bit-decomp helper
use crate::circuit_functions::hints::unconstrained_to_bits;

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_reconstruct_from_bits
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
pub fn constrained_reconstruct_from_bits<C: Config, Builder: RootAPI<C>>(
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

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: range_check_pow2_unsigned
// ─────────────────────────────────────────────────────────────────────────────

/// Range-checks that `value` lies in the interval `[0, 2^{n_bits} − 1]`.
///
/// Internally:
///   1. Computes an unconstrained bit-decomposition of length `n_bits`.
///   2. Enforces that each bit is 0/1 and reconstructs their sum.
///   3. Asserts `value == reconstructed_value`.
///
/// Returns the bit-decomposition so that callers can reuse the bits
/// (e.g., for sign extraction), but most callers can ignore it.
///
/// This is deliberately a *generic* gadget:
/// later we can swap out the internal implementation (e.g. lookup-based
/// range checks) while keeping this signature unchanged.
pub fn range_check_pow2_unsigned<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    value: Variable,
    n_bits: usize,
) -> Result<Vec<Variable>, CircuitError> {
    // 1) Bit-decompose value into n_bits bits
    let bits = unconstrained_to_bits(api, value, n_bits)?;

    // 2) Enforce bits are {0,1} and reconstruct
    let recon = constrained_reconstruct_from_bits(api, &bits)?;

    // 3) Enforce equality value == recon
    api.assert_is_equal(value, recon);

    Ok(bits)
}

// ─────────────────────────────────────────────────────────────────────────────
// LOGUP-RELATED CODE BELOW
// ─────────────────────────────────────────────────────────────────────────────

/// Default chunk size (number of bits per LogUp digit).
/// Each chunk/digit is in [0, 2^DEFAULT_LOGUP_CHUNK_BITS - 1].
pub const DEFAULT_LOGUP_CHUNK_BITS: usize = 4;

/// Context for reusing a single LogUp range-proof table across many checks.
///
/// - `chunk_bits` determines the table size: table has 2^{chunk_bits} rows.
/// - `table` holds the LogUpRangeProofTable with keys 0..2^{chunk_bits}-1.
///
/// Typical usage inside a circuit's `define`:
///
/// ```ignore
/// let mut ctx = LogupRangeCheckContext::new_default();
/// ctx.init::<C, B>(api);
/// // many calls:
/// ctx.range_check::<C, B>(api, value1, n_bits1)?;
/// ctx.range_check::<C, B>(api, value2, n_bits2)?;
/// // ...
/// ctx.finalize::<C, B>(api);
/// ```
pub struct LogupRangeCheckContext {
    pub chunk_bits: usize,
    pub table: LogUpRangeProofTable,
}

impl LogupRangeCheckContext {
    /// Create a new context with an explicit chunk size (number of bits per digit).
    pub fn new(chunk_bits: usize) -> Self {
        // You can tighten these bounds later if needed.
        assert!(
            chunk_bits > 0 && chunk_bits <= 128,
            "LogupRangeCheckContext: chunk_bits must be in 1..=128"
        );

        Self {
            chunk_bits,
            table: LogUpRangeProofTable::new(chunk_bits),
        }
    }

    /// Create a context using the default chunk size (currently 4 bits).
    pub fn new_default() -> Self {
        Self::new(DEFAULT_LOGUP_CHUNK_BITS)
    }

    /// Initialize the table with all valid keys [0, 2^{chunk_bits} - 1].
    pub fn init<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.table.initial::<C, B>(api);
    }

    /// Add a range-check constraint for `value ∈ [0, 2^{n_bits} - 1]`,
    /// using this context's chunk size.
    ///
    /// This *does not* call `final_check`; it only adds queries to the table.
    pub fn range_check<C: Config, B: RootAPI<C>>(
        &mut self,
        api: &mut B,
        value: Variable,
        n_bits: usize,
    ) -> Result<(), CircuitError> {
        if n_bits == 0 {
            return Err(CircuitError::Other(
                "logup_range_check_pow2_unsigned: n_bits must be > 0".into(),
            ));
        }

        if n_bits > 128 {
            return Err(CircuitError::Other(
                "logup_range_check_pow2_unsigned: n_bits > 128 not supported yet".into(),
            ));
        }

        // Delegate to the Expander LogUp implementation.
        // This adds the appropriate digits as queries to the shared table.
        self.table.rangeproof::<C, B>(api, value, n_bits);

        Ok(())
    }

    /// Run the final LogUp consistency check once for all queries added so far.
    pub fn finalize<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.table.final_check::<C, B>(api);
    }
}

/// One-shot helper that uses a fresh LogUp table for a single range-check.
///
/// Internally:
/// - builds a `LogupRangeCheckContext` with the default chunk size (4 bits),
/// - initializes the table,
/// - adds one range-check for `value` with `n_bits`,
/// - runs `finalize` to enforce the LogUp consistency.
pub fn logup_range_check_pow2_unsigned<C: Config, B: RootAPI<C>>(
    api: &mut B,
    value: Variable,
    n_bits: usize,
) -> Result<(), CircuitError> {
    let mut ctx = LogupRangeCheckContext::new_default();

    ctx.init::<C, B>(api);
    ctx.range_check::<C, B>(api, value, n_bits)?;
    ctx.finalize::<C, B>(api);

    Ok(())
}
