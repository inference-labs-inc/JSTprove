//! # Range-check gadgets
//!
//! - LogUp-based range checks with a shared context
//! - Generic bit-decomposition-based range checks
//! - Helper gadgets for reconstructing integers from boolean bitstrings

use ethnum::U256;

use circuit_std_rs::logup::LogUpRangeProofTable;
use expander_compiler::frontend::{CircuitField, Config, RootAPI, Variable};

// Required for accessing `CircuitField::<C>::MODULUS`
use expander_compiler::field::FieldArith;

use crate::circuit_functions::CircuitError;
use crate::circuit_functions::hints::unconstrained_to_bits;
use crate::circuit_functions::utils::UtilsError;

/// Default number of bits per LogUp digit.
///
/// In LogUp-based range proofs, an n-bit value is decomposed into digits in
/// base 2^{chunk_bits}. Each digit is looked up in a table of size
/// 2^{chunk_bits}. Increasing `chunk_bits` reduces the number of digits but
/// increases table size.
///
/// The default of 4 yields a 16-row table and is a practical middle ground.
pub const DEFAULT_LOGUP_CHUNK_BITS: usize = 4;

// Helper functions for modulus bitlength and debug-only checks

/// Bitlength of a positive `U256`, i.e. floor(log2(x)) + 1 for x > 0, else 0.
fn bitlen_u256(x: U256) -> usize {
    if x == U256::from(0u8) {
        return 0;
    }
    // `ethnum::U256` provides `leading_zeros()`
    let lz: u32 = x.leading_zeros();
    256usize.saturating_sub(lz as usize)
}

/// Returns floor(log2(p)) + 1, where `p` is the field modulus.
fn field_modulus_bitlen<C: Config>() -> usize {
    bitlen_u256(CircuitField::<C>::MODULUS)
}

/// Lift a `U256` constant into the circuit as a `Variable`.
fn const_u256<C: Config, B: RootAPI<C>>(api: &mut B, x: U256) -> Variable {
    api.constant(CircuitField::<C>::from_u256(x))
}

/// Compute `2^exp` as a `U256`, with a safety bound.
fn pow2_u256(exp: usize) -> Result<U256, CircuitError> {
    if exp > 255 {
        return Err(CircuitError::Other(format!(
            "pow2_u256: exponent too large for U256: {exp}"
        )));
    }
    Ok(U256::from(1u8) << (exp as u32))
}

/// Debug-only: check the signed-shift precondition `2^{n + 2} <= p + 1`.
fn debug_assert_signed_shift_ok<C: Config>(n: usize) {
    debug_assert!({
        // Want: 2^(n + 2) <= p + 1  iff  n + 2 <= floor(log2(p + 1))
        // Let bitlen(p + 1) = floor(log2(p + 1)) + 1.
        // We have n + 2 <= floor(log2(p + 1))  iff  n + 3 <= bitlen(p + 1)
        let p = CircuitField::<C>::MODULUS;
        let p_plus_1 = p.checked_add(U256::from(1u8)).unwrap();
        let l = bitlen_u256(p_plus_1);
        n.saturating_add(3) <= l
    });
}

/// Context object for amortized LogUp range checks.
///
/// # Semantics
///
/// This ultimately constrains the **field element** `value in F_p` to lie in the
/// canonical interval `[0, 2^{n_bits} - 1]`, assuming no wraparound occurs inside
/// the underlying ECC `rangeproof` arithmetic.
///
/// # Wraparound assumptions
///
/// Let `m = chunk_bits` and `n = n_bits`.
///
/// - ECC may shift the bound up to the next multiple of `m`, effectively using a
///   reconstruction length about `n + (m - 1)`.
/// - A conservative sufficient condition is:
///
/// ```text
/// 2^{n + m} <= p
/// ```
///
/// Equivalently, `(n + m) <= floor(log2(p))`.
///
/// In debug builds, this can be checked against
/// `CircuitField::<C>::MODULUS`.
///
/// # Upstream caveats
///
/// This relies on the correctness of ECC's `rangeproof_onechunk` when
/// `n_bits <= m`. If ECC still has the `mul_factor = 0` bug, then small-`n_bits`
/// checks are unsound. Once ECC merges the fix, this caveat goes away.
pub struct LogupRangeCheckContext {
    pub chunk_bits: usize,
    pub table: LogUpRangeProofTable,
    initialized: bool,
    finalized: bool,

    /// If true, run additional debug-only sanity checks (wrapped in `debug_assert!`)
    /// inside helpers like `range_check`.
    pub debug_checks: bool,
}

// Constructors and methods for managing a shared LogUp range-check context.
impl LogupRangeCheckContext {
    /// Create a context with user-specified `chunk_bits`.
    #[must_use]
    pub fn new(chunk_bits: usize) -> Self {
        assert!(
            chunk_bits > 0 && chunk_bits <= 128,
            "LogupRangeCheckContext: chunk_bits must be in 1..=128"
        );

        Self {
            chunk_bits,
            table: LogUpRangeProofTable::new(chunk_bits),
            initialized: false,
            finalized: false,
            debug_checks: true,
        }
    }

    /// Create a context using `DEFAULT_LOGUP_CHUNK_BITS`.
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(DEFAULT_LOGUP_CHUNK_BITS)
    }

    /// Enable or disable extra debug-only checks for this context.
    ///
    /// This does not affect correctness; it only controls `debug_assert!` blocks
    /// guarded by `self.debug_checks`.
    #[must_use]
    pub fn with_debug_checks(mut self, enabled: bool) -> Self {
        self.debug_checks = enabled;
        self
    }

    /// Initialize the LogUp digit table with all valid digit values
    /// `{0, ..., 2^{chunk_bits} - 1}`.
    ///
    /// Must be called once before calling `range_check`.
    ///
    /// This method is idempotent (calling it twice is a no-op).
    pub fn init<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        if self.initialized {
            return;
        }
        self.table.initial::<C, B>(api);
        self.initialized = true;
    }

    /// Add a LogUp-based power-of-two range check.
    ///
    /// Enforces `value` in `[0, 2^{n_bits} - 1]`.
    ///
    /// This:
    /// - decomposes `value` into base `2^{chunk_bits}` digits (via hints for
    ///   multi-digit values),
    /// - constrains each digit to lie in `{0, ..., 2^{chunk_bits} - 1}` via
    ///   LogUp queries,
    /// - constrains the reconstruction equality inside the circuit.
    ///
    /// This method does not call `finalize`; callers must invoke `finalize`
    /// once after all range checks have been added.
    pub fn range_check<C: Config, B: RootAPI<C>>(
        &mut self,
        api: &mut B,
        value: Variable,
        n_bits: usize,
    ) -> Result<(), CircuitError> {
        if !self.initialized {
            return Err(CircuitError::Other(
                "LogupRangeCheckContext not initialized; call init() first".into(),
            ));
        }
        if self.finalized {
            return Err(CircuitError::Other(
                "LogupRangeCheckContext already finalized; create a new context".into(),
            ));
        }
        if n_bits == 0 {
            return Err(CircuitError::Other(
                "LogupRangeCheckContext::range_check: n_bits must be > 0".into(),
            ));
        }
        if n_bits > 128 {
            return Err(CircuitError::Other(
                "LogupRangeCheckContext::range_check: n_bits > 128 not supported".into(),
            ));
        }

        // Debug-only safety check: conservative "no wraparound inside ECC rangeproof".
        // Require 2^{n_bits + chunk_bits} <= p.
        #[cfg(debug_assertions)]
        if self.debug_checks {
            debug_assert!({
                let p_bitlen = field_modulus_bitlen::<C>();
                let k = n_bits.saturating_add(self.chunk_bits);
                // Want 2^k <= p, i.e. k < bitlen(p).
                k < p_bitlen
            });
        }

        // Delegate to the upstream ECC LogUp implementation.
        self.table.rangeproof::<C, B>(api, value, n_bits);

        Ok(())
    }

    /// Finalize all accumulated LogUp queries and enforce the global consistency
    /// constraint.
    ///
    /// This method is idempotent (calling it twice is a no-op).
    pub fn finalize<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        if self.finalized {
            return;
        }
        self.table.final_check::<C, B>(api);
        self.finalized = true;
    }
}

/// One-shot helper that uses a fresh LogUp table for a single range check.
///
/// Uses a default context with debug checks enabled.
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

/// One-shot helper with explicit control over debug-only sanity checks.
pub fn logup_range_check_pow2_unsigned_with_debug<C: Config, B: RootAPI<C>>(
    api: &mut B,
    value: Variable,
    n_bits: usize,
    debug_checks: bool,
) -> Result<(), CircuitError> {
    let mut ctx = LogupRangeCheckContext::new_default().with_debug_checks(debug_checks);

    ctx.init::<C, B>(api);
    ctx.range_check::<C, B>(api, value, n_bits)?;
    ctx.finalize::<C, B>(api);

    Ok(())
}

/// Inputs for a signed power-of-two range check under balanced-residue semantics.
///
/// Proves the underlying integer lies in `[-2^signed_bound_bits, 2^signed_bound_bits - 1]`,
/// assuming the caller's balanced-residue precondition.
#[derive(Clone, Copy, Debug)]
pub struct SignedRangeInput {
    pub value: Variable,
    pub signed_bound_bits: usize,
}

/// Signed power-of-two range check under balanced-residue semantics.
///
/// # Semantics
///
/// - Let `p` be the field modulus (odd prime).
/// - Let `left_int` be an integer such that `left_int` lies in [-(p - 1)/2, (p - 1)/2].
/// - Let `n = signed_bound_bits`. We prove `left_int` lies in `[-2^n, 2^n - 1]`.
/// - Let `left` be the circuit variable holding the least nonnegative residue of `left_int` mod `p`.
///
/// Define:
///
/// - `left_shift` := least nonnegative residue of (`left_int + 2^n`) mod `p`.
///
/// Then:
///
/// - `-2^n <= left_int <= 2^n - 1` iff `0 <= left_shift <= 2^{n + 1} - 1`.
///
/// # Method
///
/// We implement this by:
///
/// 1. computing `left_shift = left + 2^n` in `F_p`, and
/// 2. calling the unsigned LogUp power-of-two range check on `left_shift` with
///    `n_bits = n + 1`.
///
/// # Assumptions
///
/// This proves the claim about the underlying integer `left_int` only if the caller
/// ensures `left_int` lies in [-(p - 1)/2, (p - 1)/2]. Otherwise distinct integers
/// can map to the same residue and may pass incorrectly.
///
/// In debug builds, this function asserts the modulus condition:
///
/// ```text
/// 2^{n + 1} - 1 <= (p - 1)/2
/// ```
///
/// equivalently:
///
/// ```text
/// 2^{n + 2} <= p + 1
/// ```
///
/// # Errors
///
/// - Returns error if `signed_bound_bits` overflows when forming `n_bits = signed_bound_bits + 1`.
/// - Returns error if `n_bits == 0` (defensive; unreachable in practice).
/// - Returns error if `n_bits > 128` (current LogUp safety bound in this crate).
pub fn logup_range_check_pow2_signed<C: Config, B: RootAPI<C>>(
    api: &mut B,
    logup_ctx: &mut LogupRangeCheckContext,
    input: SignedRangeInput,
) -> Result<(), CircuitError> {
    let SignedRangeInput {
        value: left,
        signed_bound_bits,
    } = input;
    // n_bits = signed_bound_bits + 1 (unsigned range on [0, 2^{signed_bound_bits + 1} - 1])
    let n_bits = signed_bound_bits.checked_add(1).ok_or_else(|| {
        CircuitError::Other("logup_range_check_pow2_signed: signed_bound_bits overflow".into())
    })?;

    // Defensive early exits (even if redundant)
    if n_bits == 0 {
        return Err(CircuitError::Other(
            "logup_range_check_pow2_signed: n_bits must be > 0".into(),
        ));
    }
    if n_bits > 128 {
        return Err(CircuitError::Other(
            "logup_range_check_pow2_signed: signed_bound_bits + 1 > 128 not supported yet".into(),
        ));
    }

    // Debug-only: enforce the no-wraparound condition needed for correctness of the shift trick.
    // (This is the important semantic precondition: 2^{signed_bound_bits + 2} <= p + 1.)
    #[cfg(debug_assertions)]
    {
        debug_assert_signed_shift_ok::<C>(signed_bound_bits);
    }

    // Compute S = 2^{signed_bound_bits} as a native integer constant (U256), then lift into the circuit.
    let s_u256: U256 = pow2_u256(signed_bound_bits)?;
    let s_var: Variable = const_u256::<C, B>(api, s_u256);

    // left_shift = left + S in F_p.
    let left_shift: Variable = api.add(left, s_var);

    // Enforce 0 <= left_shift <= 2^{signed_bound_bits + 1} - 1 via the existing unsigned LogUp range check.
    logup_ctx.range_check::<C, B>(api, left_shift, n_bits)?;

    Ok(())
}

/// One-shot signed power-of-two range check using a fresh LogUp table.
///
/// Uses a default context with debug checks enabled.
pub fn logup_range_check_pow2_signed_one_shot<C: Config, B: RootAPI<C>>(
    api: &mut B,
    left: Variable,
    signed_bound_bits: usize,
) -> Result<(), CircuitError> {
    let mut ctx = LogupRangeCheckContext::new_default();
    ctx.init::<C, B>(api);
    logup_range_check_pow2_signed::<C, B>(
        api,
        &mut ctx,
        SignedRangeInput {
            value: left,
            signed_bound_bits,
        },
    )?;
    ctx.finalize::<C, B>(api);
    Ok(())
}

/// One-shot signed power-of-two range check with explicit control over LogUp debug checks.
///
/// Note: this does not disable `debug_assert_signed_shift_ok::<C>(signed_bound_bits)`,
/// which is a semantic precondition for the signed shift trick (debug-only, but not tied
/// to LogUp internals).
pub fn logup_range_check_pow2_signed_one_shot_with_debug<C: Config, B: RootAPI<C>>(
    api: &mut B,
    left: Variable,
    signed_bound_bits: usize,
    debug_checks: bool,
) -> Result<(), CircuitError> {
    let mut ctx = LogupRangeCheckContext::new_default().with_debug_checks(debug_checks);
    ctx.init::<C, B>(api);
    logup_range_check_pow2_signed::<C, B>(
        api,
        &mut ctx,
        SignedRangeInput {
            value: left,
            signed_bound_bits,
        },
    )?;
    ctx.finalize::<C, B>(api);
    Ok(())
}

/// Policy for enabling or disabling debug-only checks.
#[derive(Clone, Copy, Debug)]
pub enum SignedPreconditionChecks {
    On,
    Off,
}

#[inline]
fn maybe_debug_assert_signed_shift_ok<C: Config>(n: usize, checks: SignedPreconditionChecks) {
    #[cfg(debug_assertions)]
    {
        if matches!(checks, SignedPreconditionChecks::On) {
            debug_assert_signed_shift_ok::<C>(n);
        }
    }
}

/// Operands and shared signed bound for comparison gadgets.
#[derive(Clone, Copy, Debug)]
pub struct SignedBounds {
    pub left: Variable,
    pub right: Variable,
    pub signed_bound_bits: usize,
}

/// Options for signed comparison gadgets (pre-checks and debug-only assertions).
#[derive(Clone, Copy, Debug)]
pub struct CmpOptions {
    pub check_left: bool,
    pub check_right: bool,
    pub debug_checks: SignedPreconditionChecks,
}

// Signed comparisons `left_int <= right_int` under assumptions.
//
// Let `n = signed_bound_bits`.
//
// Assume `left_int`, `right_int` are integers in `[-2^n, 2^n - 1]`, where:
//
// ```text
// 2^{n + 2} - 2 <= p - 1, i.e. 2^{n + 2} <= p + 1.
// ```
//
// (This assumption is NOT enforced by constraints here.)
//
// Then:
//
// ```text
// 0 <= left_int - right_int + 2^{n + 1} - 1
//   <= 2^{n + 2} - 2
//   <= p - 1,                                            (*)
// ```
//
// and:
//
// ```text
// left_int <= right_int
// ```
//
// iff:
//
// ```text
// left_int - right_int + 2^{n + 1} - 1 <= 2^{n + 1} - 1.
// ```
//
// Let `left`, `right` be the least nonnegative residues of `left_int`, `right_int` mod `p`.
// By (*), the least nonnegative residue of:
//
// ```text
// left - right + 2^{n + 1} - 1
// ```
//
// is exactly:
//
// ```text
// left_int - right_int + 2^{n + 1} - 1.
// ```
//
// Thus, an unsigned range check with:
//
// ```text
// value  = left - right + 2^{n + 1} - 1
// n_bits = n + 1
// ```
//
// enforces, under our assumption, that `left_int <= right_int`.
fn logup_leq_var_signed_assuming_bounds<C: Config, B: RootAPI<C>>(
    api: &mut B,
    logup_ctx: &mut LogupRangeCheckContext,
    bounds: SignedBounds,
    debug_checks: SignedPreconditionChecks,
) -> Result<(), CircuitError> {
    let SignedBounds {
        left,
        right,
        signed_bound_bits,
    } = bounds;
    // We will call unsigned with n_bits = signed_bound_bits + 1.
    let n_bits = signed_bound_bits.checked_add(1).ok_or_else(|| {
        CircuitError::Other(
            "logup_leq_var_signed_assuming_bounds: signed_bound_bits overflow".into(),
        )
    })?;

    if n_bits == 0 {
        return Err(CircuitError::Other(
            "logup_leq_var_signed_assuming_bounds: n_bits must be > 0".into(),
        ));
    }
    if n_bits > 128 {
        return Err(CircuitError::Other(
            "logup_leq_var_signed_assuming_bounds: signed_bound_bits + 1 > 128 not supported"
                .into(),
        ));
    }

    // Debug-only: check 2^{signed_bound_bits + 2} <= p + 1 (same precondition as signed shift checks).
    maybe_debug_assert_signed_shift_ok::<C>(signed_bound_bits, debug_checks);

    // shift := 2^{signed_bound_bits + 1} - 1 (as a U256 constant, then lifted into the circuit).
    let shift_bound_pow2: U256 = pow2_u256(n_bits)?; // 2^{signed_bound_bits + 1}
    let shift_u256: U256 = shift_bound_pow2
        .checked_sub(U256::from(1u8))
        .ok_or_else(|| {
            CircuitError::Other("shift = 2^{signed_bound_bits + 1} - 1 underflow".into())
        })?;
    let shift_var: Variable = const_u256::<C, B>(api, shift_u256);

    // shifted_diff := (left - right) + shift in F_p.
    let diff: Variable = api.sub(left, right);
    let shifted_diff: Variable = api.add(diff, shift_var);

    // Enforce shifted_diff in [0, 2^{signed_bound_bits + 1} - 1] via unsigned LogUp range check.
    logup_ctx.range_check::<C, B>(api, shifted_diff, n_bits)?;

    Ok(())
}

/// Wrapper: enforce `left_int >= right_int` by swapping arguments (`right_int <= left_int`).
#[allow(dead_code)]
fn logup_geq_var_signed_assuming_bounds<C: Config, B: RootAPI<C>>(
    api: &mut B,
    logup_ctx: &mut LogupRangeCheckContext,
    bounds: SignedBounds,
    debug_checks: SignedPreconditionChecks,
) -> Result<(), CircuitError> {
    let SignedBounds {
        left,
        right,
        signed_bound_bits,
    } = bounds;

    logup_leq_var_signed_assuming_bounds::<C, B>(
        api,
        logup_ctx,
        SignedBounds {
            left: right,
            right: left,
            signed_bound_bits,
        },
        debug_checks,
    )
}

/// Enforce `left_int <= right_int` under signed semantics.
///
/// # Preliminary checks
///
/// Let `n = signed_bound_bits`.
///
/// If `check_left` is true, first enforce `left_int` in `[-2^n, 2^n - 1]`.
///
/// If `check_right` is true, first enforce `right_int` in `[-2^n, 2^n - 1]`.
///
/// # Method
///
/// After any requested preliminary checks, this enforces
/// `left_int <= right_int` using
/// `logup_leq_var_signed_assuming_bounds`.
///
/// # Assumptions
///
/// External semantic assumption (NOT enforceable in `F_p`):
///
/// ```text
/// left_int, right_int in [-(p - 1)/2, (p - 1)/2].
/// ```
pub fn logup_leq_var_signed<C: Config, B: RootAPI<C>>(
    api: &mut B,
    logup_ctx: &mut LogupRangeCheckContext,
    bounds: SignedBounds,
    opts: CmpOptions,
) -> Result<(), CircuitError> {
    let SignedBounds {
        left,
        right,
        signed_bound_bits,
    } = bounds;

    maybe_debug_assert_signed_shift_ok::<C>(signed_bound_bits, opts.debug_checks);

    if opts.check_left {
        logup_range_check_pow2_signed::<C, B>(
            api,
            logup_ctx,
            SignedRangeInput {
                value: left,
                signed_bound_bits,
            },
        )?;
    }
    if opts.check_right {
        logup_range_check_pow2_signed::<C, B>(
            api,
            logup_ctx,
            SignedRangeInput {
                value: right,
                signed_bound_bits,
            },
        )?;
    }

    logup_leq_var_signed_assuming_bounds::<C, B>(api, logup_ctx, bounds, opts.debug_checks)
}

/// Enforce `left_int >= right_int` under signed semantics.
///
/// # Method
///
/// Implemented as `right_int <= left_int`. If preliminary checks are requested, the
/// arguments are swapped so that `check_left` always refers to the first argument
/// of the public API.
pub fn logup_geq_var_signed<C: Config, B: RootAPI<C>>(
    api: &mut B,
    logup_ctx: &mut LogupRangeCheckContext,
    bounds: SignedBounds,
    opts: CmpOptions,
) -> Result<(), CircuitError> {
    let SignedBounds {
        left,
        right,
        signed_bound_bits,
    } = bounds;

    logup_leq_var_signed::<C, B>(
        api,
        logup_ctx,
        SignedBounds {
            left: right,
            right: left,
            signed_bound_bits,
        },
        CmpOptions {
            check_left: opts.check_right,
            check_right: opts.check_left,
            debug_checks: opts.debug_checks,
        },
    )
}

// Legacy (may still be useful)

/// Enforces that each element of a little-endian bitstring is in {0, 1} and
/// reconstructs the corresponding integer as a circuit variable.
///
/// # Overview
///
/// Given a slice of variables:
///
/// ```text
/// [b_0, b_1, ..., b_{n-1}]
/// ```
///
/// representing a bitstring in little-endian order, this gadget:
///
/// 1. Enforces `b_i` in {0, 1} using `api.assert_is_bool(b_i)`.
/// 2. Computes the weighted sum:
///
/// ```text
/// recon = sum_{i=0}^{n-1} b_i * 2^i
/// ```
///
/// inside the circuit.
///
/// This function only validates booleanity of the bits and reconstructs the
/// implied integer. Callers typically assert `value == recon` separately to
/// enforce consistency.
///
/// # Arguments
///
/// - `api`: mutable reference to a circuit builder implementing `RootAPI<C>`.
/// - `least_significant_bits`: slice `[b_0, ..., b_{n-1}]`.
///
/// # Errors
///
/// - `UtilsError::ValueTooLarge` if index `i` cannot fit into a `u32`.
/// - `UtilsError::ValueTooLarge` if computing `2^i` overflows a `u32`.
///
/// # Returns
///
/// A `Variable` encoding `sum_i b_i * 2^i`.
///
/// # Examples
///
/// ```text
/// bits = [1, 1, 0, 1]
/// recon = 1*2^0 + 1*2^1 + 0*2^2 + 1*2^3 = 11
/// ```
pub fn constrained_reconstruct_from_bits<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    least_significant_bits: &[Variable],
) -> Result<Variable, CircuitError> {
    // Start with 0 and accumulate sum b_i * 2^i as we iterate.
    let mut reconstructed = api.constant(0u32);

    for (i, &bit) in least_significant_bits.iter().enumerate() {
        // Enforce b_i in {0, 1} via b(b - 1) = 0.
        api.assert_is_bool(bit);
        // Compute b_i * 2^i.

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

/// Enforces that `value` lies in the interval `[0, 2^{n_bits} - 1]`, using a
/// bit-decomposition and equality check.
///
/// # Overview
///
/// This gadget performs:
///
/// 1. `bits = unconstrained_to_bits(api, value, n_bits)`
/// 2. `recon = constrained_reconstruct_from_bits(api, bits)`
/// 3. `api.assert_is_equal(value, recon)`
///
/// If the field modulus is greater than `2^{n_bits}`, these constraints enforce
/// that `value` is exactly the integer represented by the bitstring.
///
/// The returned bits may be reused by the caller.
///
/// # Arguments
///
/// - `api`: circuit builder.
/// - `value`: field element to be range constrained.
/// - `n_bits`: number of unsigned bits.
///
/// # Errors
///
/// - Propagates errors from `unconstrained_to_bits`.
/// - Propagates errors from `constrained_reconstruct_from_bits`.
///
/// # Returns
///
/// Vector of bits `[b_0, ..., b_{n-1}]` in little-endian order.
///
/// # Examples
///
/// ```text
/// bits = range_check_pow2_unsigned(api, x, 8)
/// lsb = bits[0]
/// msb = bits[7]
/// ```
pub fn range_check_pow2_unsigned<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    value: Variable,
    n_bits: usize,
) -> Result<Vec<Variable>, CircuitError> {
    // 1) Bit-decompose value into n_bits bits.
    let bits = unconstrained_to_bits(api, value, n_bits)?;

    // 2) Enforce bits are {0, 1} and reconstruct.
    let recon = constrained_reconstruct_from_bits(api, &bits)?;

    // 3) Enforce equality value == recon.
    api.assert_is_equal(value, recon);

    Ok(bits)
}
