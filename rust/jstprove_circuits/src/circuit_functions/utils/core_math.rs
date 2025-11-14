use ethnum::U256;
/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
    utils::UtilsError,
};

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
    let recon = assert_is_bitstring_and_reconstruct(api, &bits)?;

    // 3) Enforce equality value == recon
    api.assert_is_equal(value, recon);

    Ok(bits)
}

// ─────────────────────────────────────────────────────────────────────────────
// CONTEXT: MaxMinAssertionContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for applying `constrained_max` / `constrained_min` with a fixed
/// shift exponent `s`, to avoid recomputing constants in repeated calls.
///
/// This is shared across:
///   - MaxPool (windowed max over tensors),
///   - MaxLayer (elementwise max),
///   - MinLayer (elementwise min).
pub struct MaxMinAssertionContext {
    /// The exponent `s` such that `S = 2^s`.
    pub shift_exponent: usize,

    /// The offset `S = 2^s`, lifted as a constant into the circuit.
    pub offset: Variable,
}

impl MaxMinAssertionContext {
    /// Creates a new context for asserting maximums/minimums, given a
    /// `shift_exponent = s`.
    ///
    /// # Arguments
    /// - `api`: Mutable reference to a builder for creating constants.
    /// - `shift_exponent`: Exponent `s` used to compute `2^s`.
    ///
    /// # Errors
    /// - [`LayerError::Other`] if `shift_exponent` is too large to fit in a `u32`.
    /// - [`LayerError::InvalidParameterValue`] if the computed offset overflows `u32`.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        shift_exponent: usize,
    ) -> Result<Self, LayerError> {
        let offset_: u32 = 1u32
            .checked_shl(
                u32::try_from(shift_exponent).map_err(|_| LayerError::Other {
                    layer: LayerKind::MaxPool,
                    msg: format!("Shift exponent {shift_exponent} is too large for type: u32"),
                })?,
            )
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxMinAssertionContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: shift_exponent.to_string(),
            })?;
        let offset = api.constant(offset_);
        Ok(Self {
            shift_exponent,
            offset,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_max
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the maximum value in a nonempty slice of field elements (interpreted as integers in `[0, p−1]`),
/// using only unconstrained witness operations and explicit selection logic.
///
/// Internally, this function performs pairwise comparisons using `unconstrained_greater` and `unconstrained_lesser_eq`,
/// and selects the maximum via weighted sums:
/// `current_max ← v·(v > current_max) + current_max·(v ≤ current_max)`
///
/// # Errors
/// - If `values` is empty.
///
/// # Arguments
/// - `api`: A mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `values`: A slice of `Variable`s, each assumed to lie in the range `[0, p−1]`.
///
/// # Returns
/// A `Variable` encoding `max_i values[i]`, the maximum value in the slice.
///
/// # Example
/// ```ignore
/// // For values = [7, 2, 9, 5], returns 9.
/// ```
pub fn unconstrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Max,
            msg: "unconstrained_max: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // Initialize with the first element
    let mut current_max = values[0];
    for &v in &values[1..] {
        // Compute indicators: is_greater = 1 if v > current_max, else 0
        let is_greater = api.unconstrained_greater(v, current_max);
        let is_not_greater = api.unconstrained_lesser_eq(v, current_max);

        // Select either v or current_max based on indicator bits
        let take_v = api.unconstrained_mul(v, is_greater);
        let keep_old = api.unconstrained_mul(current_max, is_not_greater);

        // Update current_max
        current_max = api.unconstrained_add(take_v, keep_old);
    }

    Ok(current_max)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_min
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the minimum value in a nonempty slice of field elements (interpreted as integers in `[0, p−1]`),
/// using only unconstrained witness operations and explicit selection logic.
///
/// Internally, this function performs pairwise comparisons using `unconstrained_greater` and `unconstrained_lesser_eq`,
/// and selects the minimum via weighted sums:
/// `current_min ← v·(current_min > v) + current_min·(current_min ≤ v)`
///
/// # Errors
/// - If `values` is empty.
///
/// # Arguments
/// - `api`: A mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `values`: A slice of `Variable`s, each assumed to lie in the range `[0, p−1]`.
///
/// # Returns
/// A `Variable` encoding `min_i values[i]`, the minimum value in the slice.
///
/// # Example
/// ```ignore
/// // For values = [7, 2, 9, 5], returns 2.
/// ```
pub fn unconstrained_min<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Min,
            msg: "unconstrained_min: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // Initialize with the first element
    let mut current_min = values[0];
    for &v in &values[1..] {
        // Compute indicators: is_lesser = 1 if v < current_min, else 0
        let is_lesser = api.unconstrained_lesser(v, current_min);
        let is_not_lesser = api.unconstrained_greater_eq(v, current_min);

        // Select either v or current_min based on indicator bits
        let take_v = api.unconstrained_mul(v, is_lesser);
        let keep_old = api.unconstrained_mul(current_min, is_not_lesser);

        // Update current_min
        current_min = api.unconstrained_add(take_v, keep_old);
    }

    Ok(current_min)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_max
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts that a given slice of `Variable`s contains a maximum value `M`,
/// by verifying that some `x_i` satisfies `M = max_i x_i`, using a combination of
/// unconstrained helper functions and explicit constraint assertions,
/// along with an offset-shifting technique to reduce comparisons to the
/// nonnegative range `[0, 2^(s + 1) − 1]`.
///
/// # Idea
/// Each `x_i` is a field element (i.e., a `Variable` representing the least nonnegative residue mod `p`)
/// that is **assumed** to encode a signed integer in the interval `[-S, T − S] = [−2^s, 2^s − 1]`,
/// where `S = 2^s` and `T = 2·S - 1 = 2^(s + 1) − 1]`.
///
/// Since all circuit operations take place in `𝔽_p` and each `x_i` is already reduced modulo `p`,
/// we shift each value by `S` on-circuit to ensure that the quantity `x_i + S` lands in `[0, T]`.
/// Under the assumption that `x_i ∈ [−S, T − S]`, this shift does **not** wrap around modulo `p`,
/// so `x_i + S` in `𝔽_p` reflects the true integer sum.
///
/// We then compute:
/// ```text
///     M^♯ = max_i (x_i + S)
///     M   = M^♯ − S mod p
/// ```
/// to recover the **least nonnegative residue** of the maximum value, `M`.
///
/// To verify that `M` is indeed the maximum:
/// - For each `x_i`, we compute `Δ_i = M − x_i`, and use bit decomposition to enforce
///   `Δ_i ∈ [0, T]`, using `s + 1` bits.
/// - Then we constrain the product `∏_i Δ_i` to be zero. This ensures that at least one
///   `Δ_i = 0`, i.e., that some `x_i = M`.
///
/// # Example
/// Suppose the input slice encodes the signed integers `[-2, 0, 3]`, and `s = 2`, so `S = 4`, `T = 7`.
///
/// - Shift:
///   `x_0 = -2` ⇒ `x_0 + S = 2`
///   `x_1 =  0` ⇒ `x_1 + S = 4`
///   `x_2 =  3` ⇒ `x_2 + S = 7`
///
/// - Compute:
///   `M^♯ = max{x_i + S} = 7`
///   `M   = M^♯ − S = 3`
///
/// - Verify:
///   For each `x_i`, compute `Δ_i = M − x_i ∈ [0, 7]`
///   The values are: `Δ = [5, 3, 0]`
///   Since one `Δ_i = 0`, we conclude that some `x_i = M`.
///
/// # Assumptions
/// - All values `x_i` are `Variable`s in `𝔽_p` that **encode signed integers** in `[-S, T − S]`.
/// - The prime `p` satisfies `p > T = 2^(s + 1) − 1`, so no wraparound occurs in `x_i + S`.
///
/// # Errors
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `context`: A `MaxMinAssertionContext` holding shift-related parameters.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
pub fn constrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &MaxMinAssertionContext, // S = 2^s = context.offset
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    // 0) Require nonempty input
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Max,
            msg: "constrained_max: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute max_i (x_i^♯), which equals M^♯ = M + S
    let max_offset = unconstrained_max(api, &values_offset)?;

    // 3) Recover M = M^♯ − S
    let max_raw = api.sub(max_offset, context.offset);

    // 4) For each x_i, range-check Δ_i = M − x_i ∈ [0, T] using s + 1 bits
    let n_bits =
        context
            .shift_exponent
            .checked_add(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::Max,
                layer_name: "MaxMinAssertionContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: context.shift_exponent.to_string(),
            })?;
    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        // Δ ∈ [0, T] ⇔ ∃ bitstring of length s + 1 summing to Δ
        let _delta_bits =
            range_check_pow2_unsigned(api, delta, n_bits).map_err(|e| LayerError::Other {
                layer: LayerKind::Max,
                msg: format!("range_check_pow2_unsigned failed: {e}"),
            })?;

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
    Ok(max_raw)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_min
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts that a given slice of `Variable`s contains a minimum value `M`,
/// by verifying that some `x_i` satisfies `M = min_i x_i`, using a combination of
/// unconstrained helper functions and explicit constraint assertions,
/// along with an offset-shifting technique to reduce comparisons to the
/// nonnegative range `[0, 2^(s + 1) − 1]`.
///
/// # Idea
/// Each `x_i` is a field element (i.e., a `Variable` representing the least nonnegative residue mod `p`)
/// that is **assumed** to encode a signed integer in the interval `[-S, T − S] = [−2^s, 2^s − 1]`,
/// where `S = 2^s` and `T = 2·S - 1 = 2^(s + 1) − 1]`.
///
/// Since all circuit operations take place in `𝔽_p` and each `x_i` is already reduced modulo `p`,
/// we shift each value by `S` on-circuit to ensure that the quantity `x_i + S` lands in `[0, T]`.
/// Under the assumption that `x_i ∈ [−S, T − S]`, this shift does **not** wrap around modulo `p`,
/// so `x_i + S` in `𝔽_p` reflects the true integer sum.
///
/// We then compute:
/// ```text
///     M^♯ = min_i (x_i + S)
///     M   = M^♯ − S mod p
/// ```
/// to recover the **least nonnegative residue** of the minimum value, `M`.
///
/// To verify that `M` is indeed the minimum:
/// - For each `x_i`, we compute `Δ_i = x_i - M`, and use bit decomposition to enforce
///   `Δ_i ∈ [0, T]`, using `s + 1` bits.
/// - Then we constrain the product `∏_i Δ_i` to be zero. This ensures that at least one
///   `Δ_i = 0`, i.e., that some `x_i = M`.
///
/// # Example
/// Suppose the input slice encodes the signed integers `[-2, 0, 3]`, and `s = 2`, so `S = 4`, `T = 7`.
///
/// - Shift:
///   `x_0 = -2` ⇒ `x_0 + S = 2`
///   `x_1 =  0` ⇒ `x_1 + S = 4`
///   `x_2 =  3` ⇒ `x_2 + S = 7`
///
/// - Compute:
///   `M^♯ = min{x_i + S} = 2`
///   `M   = M^♯ − S = -2`
///
/// - Verify:
///   For each `x_i`, compute `Δ_i = x_i - M ∈ [0, 7]`
///   The values are: `Δ = [0, 2, 5]`
///   Since one `Δ_i = 0`, we conclude that some `x_i = M`.
///
/// # Assumptions
/// - All values `x_i` are `Variable`s in `𝔽_p` that **encode signed integers** in `[-S, T − S]`.
/// - The prime `p` satisfies `p > T = 2^(s + 1) − 1`, so no wraparound occurs in `x_i + S`.
///
/// # Errors
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `context`: A `MaxMinAssertionContext` holding shift-related parameters.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
pub fn constrained_min<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &MaxMinAssertionContext, // S = 2^s = context.offset
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    // 0) Require nonempty input
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::Min,
            msg: "constrained_min: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute min_i (x_i^♯), which equals M^♯ = M + S
    let min_offset = unconstrained_min(api, &values_offset)?;

    // 3) Recover M = M^♯ − S
    let min_raw = api.sub(min_offset, context.offset);

    // 4) For each x_i, range-check Δ_i = x_i − M ∈ [0, T] using s + 1 bits
    let n_bits =
        context
            .shift_exponent
            .checked_add(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::Min,
                layer_name: "MaxMinAssertionContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: context.shift_exponent.to_string(),
            })?;
    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(x, min_raw);

        // Δ ∈ [0, T] ⇔ ∃ bitstring of length s + 1 summing to Δ
        let _delta_bits =
            range_check_pow2_unsigned(api, delta, n_bits).map_err(|e| LayerError::Other {
                layer: LayerKind::Min,
                msg: format!("range_check_pow2_unsigned failed: {e}"),
            })?;

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
    Ok(min_raw)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_clip
// ─────────────────────────────────────────────────────────────────────────────

/// Computes `clip(x; min, max)` at the witness level using only unconstrained
/// operations and the existing `unconstrained_max` / `unconstrained_min` gadgets.
///
/// Semantics (elementwise, assuming `min <= max` in the intended integer semantics):
/// - If both `lower` and `upper` are present:
///       y = min(max(x, lower), upper)
/// - If only `lower` is present:
///       y = max(x, lower)
/// - If only `upper` is present:
///       y = min(x, upper)
/// - If neither is present:
///       y = x
///
/// All variables are field elements (least nonnegative residues), interpreted
/// as signed integers in a fixed range consistent with the surrounding circuit.
pub fn unconstrained_clip<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    lower: Option<Variable>,
    upper: Option<Variable>,
) -> Result<Variable, CircuitError> {
    // Start from x and apply lower / upper bounds as needed.
    let mut cur = x;

    if let Some(a) = lower {
        // cur <- max(cur, a)
        cur = unconstrained_max(api, &[cur, a])?;
    }

    if let Some(b) = upper {
        // cur <- min(cur, b)
        cur = unconstrained_min(api, &[cur, b])?;
    }

    Ok(cur)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_clip
// ─────────────────────────────────────────────────────────────────────────────

/// Enforces `c = clip(x; min, max)` using the existing `constrained_max` and
/// `constrained_min` gadgets under a shared [`MaxMinAssertionContext`].
///
/// Assumptions:
/// - `x`, `min`, and `max` (if present) all encode signed integers in the range
///   `[-S, T - S] = [-2^s, 2^s - 1]`, where `S = 2^s` is given by
///   `context.shift_exponent`.
/// - The field modulus `p` satisfies `p > 2^(s + 1) - 1` to avoid wraparound.
///
/// Semantics match ONNX `Clip`:
/// - If both `lower` and `upper` are present:
///       c = min(max(x, lower), upper)
/// - If only `lower` is present:
///       c = max(x, lower)
/// - If only `upper` is present:
///       c = min(x, upper)
/// - If neither is present:
///       c = x
///
/// Each `constrained_max` / `constrained_min` call:
/// - Checks the signed range via bit-decomposition and reconstruction.
/// - Enforces that the result equals one of the inputs.
/// - Uses a product-of-differences check to guarantee at least one exact match.
pub fn constrained_clip<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &MaxMinAssertionContext,
    x: Variable,
    lower: Option<Variable>,
    upper: Option<Variable>,
) -> Result<Variable, CircuitError> {
    // Degenerate case: no bounds → identity
    if lower.is_none() && upper.is_none() {
        return Ok(x);
    }

    // Apply lower bound via constrained_max when present
    let mut cur = if let Some(a) = lower {
        constrained_max(api, context, &[x, a])?
    } else {
        x
    };

    // Apply upper bound via constrained_min when present
    if let Some(b) = upper {
        cur = constrained_min(api, context, &[cur, b])?;
    }

    Ok(cur)
}
