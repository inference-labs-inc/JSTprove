use expander_compiler::frontend::*;

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
/// # Example
/// ```ignore
/// // For input = 43 and n_bits = 4:
/// // Returns [1, 1, 0, 1], since 43 = 0b101011, and the 4 LSBs are 1011.
/// ```
pub fn unconstrained_to_bits<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input: Variable,
    n_bits: usize,
) -> Vec<Variable> {
    let mut least_significant_bits = Vec::with_capacity(n_bits);
    let mut current = input;

    for _ in 0..n_bits {
        // Extract bit 0 of `current`
        let bit = api.unconstrained_bit_and(current, 1u32);
        least_significant_bits.push(bit);
        // Shift right by one
        current = api.unconstrained_shift_r(current, 1u32);
    }

    least_significant_bits
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
) -> Variable {
    // Start with 0 and accumulate ∑ bᵢ·2ⁱ as we iterate
    let mut reconstructed = api.constant(0u32);

    for (i, &bit) in least_significant_bits.iter().enumerate() {
        // Enforce bᵢ ∈ {0,1} via b(b−1) = 0
        let one = api.constant(1u32);
        let bit_minus_one = api.sub(bit, one);
        let vanishing = api.mul(bit, bit_minus_one);
        api.assert_is_zero(vanishing);

        // Compute bᵢ · 2ⁱ
        let weight = 1u32
            .checked_shl(i as u32)
            .expect("bit index i must be < 32");
        let weight_const = api.constant(weight);
        let term = api.mul(weight_const, bit);
        reconstructed = api.add(reconstructed, term);
    }

    reconstructed
}

// TO DO: delete this is STRUCT + rescale (below) works.
// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: rescale_by_power_of_two
// ─────────────────────────────────────────────────────────────────────────────

/// Computes `q = floor((c + α·S)/α) − S`, optionally applying ReLU.
///
/// All intermediate values are computed using **unconstrained operations** (i.e.,  
/// witness-only helper functions such as division, modulo, and bit decomposition),  
/// and **correctness is enforced explicitly** via constraint assertions such as  
/// `assert_is_equal`, `assert_is_zero`, and bitstring range checks.
///
/// # Notation
/// - Let `κ = scaling_exponent`, and define `α = 2^κ`.
/// - Let `s = shift_exponent`, and define `S = 2^s`.
/// - Define `T = 2·S − 1 = 2^(s + 1) − 1`.
/// - `c` is the input `dividend`.
/// - `r` is the remainder.
/// - `q^♯` is the offset quotient: `q^♯ = q + S`.
///
/// # Process
/// 1. Form `shifted_dividend = α·S + c`.
/// 2. Unconstrained division: `shifted_dividend = α·q^♯ + r`.
/// 3. Enforce this equality with a constraint.
/// 4. Range-check `r ∈ [0, α − 1]`.
/// 5. Range-check `q^♯ ∈ [0, T] = [0, 2^(s + 1) − 1]`.
/// 6. Recover `q = q^♯ − S`.
/// 7. If `apply_relu`, output `max(q, 0)` using MSB of `q^♯`.
///
/// # Panics
/// - If `checked_shl` or `checked_mul` overflows a 32-bit integer.
///
/// # Arguments
/// - `api`: The circuit builder implementing `RootAPI<C>`.
/// - `dividend` (`c`): The field element to rescale, assumed in `[-α·S, α·(T − S)]`.
/// - `scaling_exponent` (`κ`): So that `α = 2^κ`.
/// - `shift_exponent` (`s`): So that `S = 2^s`.
/// - `apply_relu`: If `true`, returns `max(q, 0)` instead of `q`.
pub fn rescale_by_power_of_two<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    dividend: Variable,          // c
    scaling_exponent: usize,     // κ
    shift_exponent: usize,       // s
    apply_relu: bool,
) -> Variable {
    // Step 1: compute α = 2^κ and S = 2^s as native integers
    let scaling_factor_: u32 = 1u32
        .checked_shl(scaling_exponent as u32)
        .expect("scaling_exponent < 32");
    let shift_: u32 = 1u32
        .checked_shl(shift_exponent as u32)
        .expect("shift_exponent < 32");

    // Lift α and S to circuit constants
    let scaling_factor = api.constant(scaling_factor_);
    let shift = api.constant(shift_);

    // Step 2: compute α·S
    let scaled_shift_: u32 = scaling_factor_
        .checked_mul(shift_)
        .expect("2^κ · 2^s fits in u32");
    let scaled_shift = api.constant(scaled_shift_);

    // Step 3: compute shifted_dividend = α·S + c
    let shifted_dividend = api.add(scaled_shift, dividend);

    // Step 4: Unconstrained Euclidean division: α·S + c = α·q^♯ + r
    let shifted_q = api.unconstrained_int_div(shifted_dividend, scaling_factor_);
    let remainder = api.unconstrained_mod(shifted_dividend, scaling_factor_);

    // Step 4b: Enforce α·S + c = α·q^♯ + r 
    let rhs_first_term = api.mul(scaling_factor, shifted_q);
    let rhs = api.add(rhs_first_term, remainder);
    api.assert_is_equal(shifted_dividend, rhs);

    // Step 5: Range-check r ∈ [0, α − 1] using κ bits
    let rem_bits = unconstrained_to_bits(api, remainder, scaling_exponent);
    let rem_recon = assert_is_bitstring_and_reconstruct(api, &rem_bits);
    api.assert_is_equal(remainder, rem_recon);

    // Step 6: Range-check q^♯ ∈ [0, 2^(s + 1) − 1] using s + 1 bits
    let n_bits_q = shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 fits in usize");
    let q_bits = unconstrained_to_bits(api, shifted_q, n_bits_q);
    let q_recon = assert_is_bitstring_and_reconstruct(api, &q_bits);
    api.assert_is_equal(shifted_q, q_recon);

    // Step 7: Recover q = q^♯ − S
    let quotient = api.sub(shifted_q, shift);

    // Step 8: If ReLU is applied, zero out negatives using MSB of q^♯
    if apply_relu {
        // q ≥ 0 ⇔ q^♯ ≥ S ⇔ MSB d_s = 1
        let sign_bit = q_bits[shift_exponent]; // the (s + 1)-st bit d_s
        api.mul(quotient, sign_bit)
    } else {
        quotient
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: RescalingContext
// ─────────────────────────────────────────────────────────────────────────────

/// Holds integer and circuit-level constants for rescaling by `2^κ` and shifting by `2^s`.
pub struct RescalingContext {
    pub scaling_exponent: usize,   // κ
    pub shift_exponent: usize,     // s
    pub scaling_factor_: u32,      // α = 2^κ
    pub shift_: u32,               // S = 2^s
    pub scaled_shift_: u32,        // α·S = 2^{κ + s}

    pub scaling_factor: Variable,
    pub shift: Variable,
    pub scaled_shift: Variable,
}

impl RescalingContext {
    pub fn new<C:Config, Builder: RootAPI<C>>(api: &mut Builder, scaling_exponent: usize, shift_exponent: usize) -> Self {
        let scaling_factor_ = 1u32.checked_shl(scaling_exponent as u32).expect("scaling_exponent < 32");
        let shift_ = 1u32.checked_shl(shift_exponent as u32).expect("shift_exponent < 32");
        let scaled_shift_ = scaling_factor_.checked_mul(shift_).expect("2^κ · 2^s fits in u32");

        let scaling_factor = api.constant(scaling_factor_);
        let shift = api.constant(shift_);
        let scaled_shift = api.constant(scaled_shift_);

        Self {
            scaling_exponent,
            shift_exponent,
            scaling_factor_,
            shift_,
            scaled_shift_,
            scaling_factor,
            shift,
            scaled_shift,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: rescale
// ─────────────────────────────────────────────────────────────────────────────

/// Computes `q = floor((c + α·S)/α) − S`, optionally applying ReLU, using a
/// precomputed [`RescalingContext`] for efficiency and clarity.
///
/// All intermediate values are computed using **unconstrained operations** (i.e.,  
/// witness-only helper functions such as division, modulo, and bit decomposition),  
/// and **correctness is enforced explicitly** via constraint assertions such as  
/// `assert_is_equal`, `assert_is_zero`, and bitstring range checks.
///
/// # Notation
/// - Let `κ = context.scaling_exponent`, and define `α = 2^κ`.
/// - Let `s = context.shift_exponent`, and define `S = 2^s`.
/// - Define `T = 2·S − 1 = 2^(s + 1) − 1`.
/// - `c` is the input `dividend`.
/// - `r` is the remainder.
/// - `q^♯` is the offset quotient: `q^♯ = q + S`.
///
/// # Process
/// 1. Form `shifted_dividend = α·S + c` using precomputed constants from `context`.
/// 2. Unconstrained division: `shifted_dividend = α·q^♯ + r`.
/// 3. Enforce this equality with a constraint.
/// 4. Range-check `r ∈ [0, α − 1]`.
/// 5. Range-check `q^♯ ∈ [0, T] = [0, 2^(s + 1) − 1]`.
/// 6. Recover `q = q^♯ − S`.
/// 7. If `apply_relu`, output `max(q, 0)` using MSB of `q^♯`.
///
/// # Efficiency Note
/// The use of a [`RescalingContext`] avoids recomputing and re-lifting  
/// the constants `α`, `S`, and `α·S` on each call, which improves performance  
/// in matrix-wide applications or other scenarios involving repeated rescaling.
///
/// # Panics
/// - If the precomputed values in `context` were created using exponents
///   that caused `checked_shl` or `checked_mul` to overflow a 32-bit integer.
///
/// # Arguments
/// - `api`: The circuit builder implementing `RootAPI<C>`.
/// - `context`: A [`RescalingContext`] holding both native and circuit-lifted values
///   for `α`, `S`, and `α·S`.
/// - `dividend` (`c`): The field element to rescale, assumed in `[-α·S, α·(T − S)]`.
/// - `apply_relu`: If `true`, returns `max(q, 0)` instead of `q`.
pub fn rescale<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &RescalingContext, 
    dividend: Variable,
    apply_relu: bool,
) -> Variable {
    // Step 1: compute shifted_dividend = α·S + c
    let shifted_dividend = api.add(context.scaled_shift, dividend);

    // Step 2: Compute unchecked witness values q^♯, r via unconstrained Euclidean division: α·S + c = α·q^♯ + r
    let shifted_q = api.unconstrained_int_div(shifted_dividend, context.scaling_factor_); // q^♯
    let remainder = api.unconstrained_mod(shifted_dividend, context.scaling_factor_); // r

    // Step 3: Enforce α·S + c = α·q^♯ + r 
    let rhs_first_term = api.mul(context.scaling_factor, shifted_q);
    let rhs = api.add(rhs_first_term, remainder);
    api.assert_is_equal(shifted_dividend, rhs);

    // Step 4: Range-check r ∈ [0, α − 1] using κ bits
    let rem_bits = unconstrained_to_bits(api, remainder, context.scaling_exponent);
    let rem_recon = assert_is_bitstring_and_reconstruct(api, &rem_bits);
    api.assert_is_equal(remainder, rem_recon);

    // Step 5: Range-check q^♯ ∈ [0, 2^(s + 1) − 1] using s + 1 bits
    let n_bits_q = context.shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 fits in usize");
    let q_bits = unconstrained_to_bits(api, shifted_q, n_bits_q);
    let q_recon = assert_is_bitstring_and_reconstruct(api, &q_bits);
    api.assert_is_equal(shifted_q, q_recon);

    // Step 6: Recover quotient q = q^♯ − S
    let quotient = api.sub(shifted_q, context.shift); // q = q^♯ − S

    // Step 7: If ReLU is applied, zero out negatives using MSB of q^♯
    if apply_relu {
        // q ≥ 0 ⇔ q^♯ ≥ S ⇔ MSB (bit d_s) is 1, where q^♯ ≤ 2^(s + 1) - 1
        let sign_bit = q_bits[context.shift_exponent]; // the (s + 1)-st bit d_s
        api.mul(quotient, sign_bit)
    } else {
        quotient
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
/// # Panics
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
) -> Variable {
    assert!(
        !values.is_empty(),
        "unconstrained_max: input slice must be nonempty"
    );

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

    current_max
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: MaxAssertionContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for applying `assert_is_max` with a fixed shift exponent `s`,
/// to avoid recomputing constants in repeated calls (e.g., in max pooling).
pub struct MaxAssertionContext {
    /// The exponent `s` such that `S = 2^s`.
    pub shift_exponent: usize,

    /// The offset `S = 2^s`, lifted as a constant into the circuit.
    pub offset: Variable,
}

impl MaxAssertionContext {
    /// Creates a new context for asserting maximums, given a `shift_exponent = s`.
    ///
    /// Computes `S = 2^s` and lifts it to a constant for reuse.
    pub fn new<C: Config, Builder: RootAPI<C>>(api: &mut Builder, shift_exponent: usize) -> Self {
        let offset_: u32 = 1u32
            .checked_shl(shift_exponent as u32)
            .expect("shift_exponent must be less than 32");
        let offset = api.constant(offset_);
        Self {
            shift_exponent,
            offset,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: assert_is_max
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
/// # Panics
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `context`: A `MaxAssertionContext` holding shift-related parameters.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
pub fn assert_is_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &MaxAssertionContext, // S = 2^s = context.offset
    values: &[Variable],
) {
    // 0) Require nonempty input
    assert!(
        !values.is_empty(),
        "assert_is_max: input slice must be nonempty"
    );

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute max_i (x_i^♯), which equals M^♯ = M + S
    let max_offset = unconstrained_max(api, &values_offset);

    // 3) Recover M = M^♯ − S
    let max_raw = api.sub(max_offset, context.offset);

    // 4) For each x_i, range-check Δ_i = M − x_i ∈ [0, T] using s + 1 bits
    let n_bits = context
        .shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 must fit in usize");
    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        // Δ ∈ [0, T] ⇔ ∃ bitstring of length s + 1 summing to Δ
        let bits = unconstrained_to_bits(api, delta, n_bits);
        // TO DO: elaborate/make more explicit, e.g. "Range check enforcing Δ >= 0"
        let recon = assert_is_bitstring_and_reconstruct(api, &bits);
        api.assert_is_equal(delta, recon);

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
}

// TO DO: delete if assert_is_max works
// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: assert_is_max_old
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
/// where `S = 2^s` and `T = 2·S - 1 = 2^(s + 1) − 1`.
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
/// # Panics
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
/// - `shift_exponent`: The exponent `s`, so that `S = 2^s` and `T = 2^(s + 1) − 1`.
pub fn assert_is_max_old<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
    shift_exponent: usize,
) {
    // 0) Require nonempty input
    assert!(
        !values.is_empty(),
        "assert_is_max: input slice must be nonempty"
    );

    // 1) Compute offset = 2^s (S = 2^s), lifted to a circuit constant
    let offset_: u32 = 1u32
        .checked_shl(shift_exponent as u32)
        .expect("shift_exponent < 32");
    let offset = api.constant(offset_);

    // 2) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, offset));
    }

    // 3) Compute max_i (x_i^♯), which equals M^♯ = M + S
    let max_offset = unconstrained_max(api, &values_offset);

    // 4) Recover M = M^♯ − S
    let max_raw = api.sub(max_offset, offset);

    // 5) For each x_i, range-check Δ_i = M − x_i ∈ [0, T] using s + 1 bits
    let n_bits = shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 must fit in usize");
    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        // Δ ∈ [0, T] ⇔ ∃ bitstring of length s + 1 summing to Δ
        let bits = unconstrained_to_bits(api, delta, n_bits);
        let recon = assert_is_bitstring_and_reconstruct(api, &bits);
        api.assert_is_equal(delta, recon);

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 6) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
}