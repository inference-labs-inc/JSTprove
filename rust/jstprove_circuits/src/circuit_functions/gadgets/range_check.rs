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

// -----------------------------------------------------------------------------
// FUNCTION: constrained_reconstruct_from_bits
// -----------------------------------------------------------------------------

/// Enforces that each element of a little-endian bitstring is in {0,1} and
/// reconstructs the corresponding integer as a circuit variable.
///
/// # Overview
/// Given a slice of variables:
///     [b_0, b_1, ..., b_{n-1}]
/// representing a bitstring in little-endian order, this gadget:
///
/// 1. Enforces b_i in {0,1} using api.assert_is_bool(b_i).
/// 2. Computes the weighted sum:
///        recon = sum_{i=0}^{n-1} b_i * 2^i
///    inside the circuit.
///
/// This function only validates Booleanity of the bits and reconstructs the
/// implied integer. Callers typically assert value == recon separately to
/// enforce consistency.
///
/// # Arguments
/// - api: mutable reference to a circuit builder implementing RootAPI<C>.
/// - least_significant_bits: slice [b_0, ..., b_{n-1}].
///
/// # Errors
/// - UtilsError::ValueTooLarge if index i cannot fit into a u32.
/// - UtilsError::ValueTooLarge if computing 2^i overflows a u32.
///
/// # Returns
/// A Variable encoding sum_{i} b_i * 2^i.
///
/// # Example
/// ```ignore
/// bits = [1, 1, 0, 1]
/// recon = 1*2^0 + 1*2^1 + 0*2^2 + 1*2^3 = 11
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

// -----------------------------------------------------------------------------
// FUNCTION: range_check_pow2_unsigned
// -----------------------------------------------------------------------------

/// Enforces that `value` lies in the interval [0, 2^{n_bits} - 1], using a
/// bit-decomposition and equality check.
///
/// # Overview
/// This gadget performs:
///
/// 1. bits = unconstrained_to_bits(api, value, n_bits)
/// 2. recon = constrained_reconstruct_from_bits(api, bits)
/// 3. api.assert_is_equal(value, recon)
///
/// If the field modulus is greater than 2^{n_bits}, these constraints enforce
/// that `value` is exactly the integer represented by the bitstring.
///
/// The returned bits may be reused by the caller.
///
/// # Arguments
/// - api: circuit builder.
/// - value: field element to be range constrained.
/// - n_bits: number of unsigned bits.
///
/// # Errors
/// - Propagates errors from unconstrained_to_bits.
/// - Propagates errors from constrained_reconstruct_from_bits.
///
/// # Returns
/// Vector of bits [b_0, ..., b_{n-1}] in little-endian order.
///
/// # Example
/// ```ignore
/// bits = range_check_pow2_unsigned(api, x, 8)
/// lsb = bits[0]
/// msb = bits[7]
/// ```
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

// -----------------------------------------------------------------------------
// CONSTANT: DEFAULT_LOGUP_CHUNK_BITS
// -----------------------------------------------------------------------------

/// Default number of bits per LogUp digit.
///
/// In LogUp-based range proofs, an n-bit value is decomposed into digits in
/// base 2^{chunk_bits}. Each digit is looked up in a table of size
/// 2^{chunk_bits}. Increasing chunk_bits reduces the number of digits but
/// increases table size.
///
/// The default of 4 yields a 16-row table and is a practical middle ground.
pub const DEFAULT_LOGUP_CHUNK_BITS: usize = 4;

// -----------------------------------------------------------------------------
// STRUCT: LogupRangeCheckContext
// -----------------------------------------------------------------------------

/// Context object for amortized LogUp range checks.
///
/// A single LogUpRangeCheckContext allows many values to be range-checked
/// against a shared LogUp table, which is more efficient than performing
/// separate one-shot checks.
///
/// Fields:
/// - chunk_bits: number of bits per LogUp digit.
/// - table: underlying LogUpRangeProofTable containing 2^{chunk_bits} rows.
///
/// Typical usage:
///
/// ```ignore
/// let mut ctx = LogupRangeCheckContext::new_default();
/// ctx.init(api);
/// ctx.range_check(api, v1, n1)?;
/// ctx.range_check(api, v2, n2)?;
/// ...
/// ctx.finalize(api);
/// ```
pub struct LogupRangeCheckContext {
    pub chunk_bits: usize,
    pub table: LogUpRangeProofTable,
}

impl LogupRangeCheckContext {
    /// Create a context with user-specified chunk_bits.
    ///
    /// chunk_bits determines the table size (2^{chunk_bits}) and the number of
    /// digits used to encode a value. Must satisfy 1 <= chunk_bits <= 128.
    ///
    /// Panics if chunk_bits is out of range.
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

    /// Create a context using DEFAULT_LOGUP_CHUNK_BITS.
    ///
    /// This is recommended unless you need to adjust LogUp performance trade-offs.
    pub fn new_default() -> Self {
        Self::new(DEFAULT_LOGUP_CHUNK_BITS)
    }

    /// Initialize the LogUp table with all valid digit values.
    ///
    /// Must be called once before calling range_check.
    pub fn init<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.table.initial::<C, B>(api);
    }

    /// Add a LogUp-based range check: enforce value in [0, 2^{n_bits} - 1].
    ///
    /// This splits value into base-2^{chunk_bits} digits and registers them as
    /// queries in the shared LogUp table.
    ///
    /// This does not finalize the consistency proof; call finalize afterward.
    ///
    /// Errors:
    /// - CircuitError if n_bits == 0.
    /// - CircuitError if n_bits > 128 (current safety bound).
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

    /// Finalize all accumulated LogUp queries and enforce the global consistency
    /// constraint. Must be called exactly once after all range_check calls.
    pub fn finalize<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.table.final_check::<C, B>(api);
    }
}

// -----------------------------------------------------------------------------
// FUNCTION: logup_range_check_pow2_unsigned
// -----------------------------------------------------------------------------

/// One-shot helper for performing a LogUp-based range check.
///
/// This wraps LogupRangeCheckContext as follows:
///   1. Create a context with DEFAULT_LOGUP_CHUNK_BITS.
///   2. ctx.init(api)
///   3. ctx.range_check(api, value, n_bits)
///   4. ctx.finalize(api)
///
/// Useful when you need a single range check. For many checks, create a
/// LogupRangeCheckContext manually and reuse it.
///
/// Errors:
/// - Propagates errors from range_check.
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
