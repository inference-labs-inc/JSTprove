//! Utilities for quantization-related arithmetic in circuit construction.
//!
//! This module provides functionality for rescaling the product of two fixed-point
//! approximations, where each operand has been independently scaled and rounded.
//! That is, we assume integers `a ≈ α·x` and `b ≈ α·y`, where `α = 2^κ` is the fixed-point
//! scaling factor, and `a`, `b` are scaled versions of real numbers `x`, `y`.
//!
//! The product `c' = a·b` approximates `α²·x·y`. Since the circuit operates only on field
//! elements (nonnegative integers modulo `p`, the field modulus), we must simulate signed
//! integer arithmetic using representative values in the range `[0, p - 1]`.
//!
//! To correctly recover a fixed-point approximation of `x·y` at scale `α`, we compute:
//!
//! ```text
//! q = floor((c + α·S)/α) − S
//! ```
//!
//! where `c` is the least nonnegative residue of the signed integer `c'`,
//! and `S = 2^s` is a centering constant that ensures intermediate values
//! remain within the field during division and remainder operations.
//!
//! The computation enforces that:
//!
//! ```text
//! c + α·S = α·q^♯ + r,    with    0 ≤ r < α
//! ```
//!
//! Then recovers `q = q^♯ − S`, and optionally applies ReLU by zeroing out negative values
//! based on the most significant bit of `q^♯`.
//!
//! The core logic is implemented in [`rescale`] and its batched variants
//! [`rescale_2d_vector`] and [`rescale_4d_vector`], using a shared [`RescalingContext`]
//! to precompute constants and optimize performance.

// External crates
use ethnum::U256;
use ndarray::ArrayD;

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

// Internal modules
use super::core_math::{assert_is_bitstring_and_reconstruct, unconstrained_to_bits};

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: RescalingContext
// ─────────────────────────────────────────────────────────────────────────────

/// Holds integer and circuit-level constants for rescaling by `α = 2^κ` and shifting by `S = 2^s`.
pub struct RescalingContext {
    pub scaling_exponent: usize, // κ: exponent such that α = 2^κ
    pub shift_exponent: usize,   // s: exponent such that S = 2^s

    pub scaling_factor_: u32, // α = 2^κ, as a native u32
    pub shift_: u32,          // S = 2^s, as a native u32
    pub scaled_shift_: U256,  // α·S = 2^{κ + s}, as a U256 for overflow safety

    pub scaling_factor: Variable, // α = 2^κ, lifted to the circuit as a Variable
    pub shift: Variable,          // S = 2^s, lifted to the circuit as a Variable
    pub scaled_shift: Variable,   // α·S = 2^{κ + s}, lifted to the circuit as a Variable
}

// ─────────────────────────────────────────────────────────────────────────────
// IMPL: RescalingContext
// ─────────────────────────────────────────────────────────────────────────────

impl RescalingContext {
    /// Constructs a [`RescalingContext`] from the given scaling and shift exponents.
    ///
    /// Precomputes:
    /// - Native powers of two:  
    ///   - `scaling_factor_ = 2^κ` (`u32`)  
    ///   - `shift_ = 2^s` (`u32`)  
    ///   - `scaled_shift_ = 2^{κ + s}` (`U256`, to avoid overflow)
    ///
    /// - Circuit-lifted versions of these values:  
    ///   - `scaling_factor`, `shift`, and `scaled_shift` (as `Variable`)
    ///
    /// These are reused throughout rescaling to avoid redundant lifting and ensure consistent constraints.
    ///
    /// # Panics
    /// Panics if `scaling_exponent` or `shift_exponent` are too large for `u32` shifts or multiplication.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        scaling_exponent: usize,
        shift_exponent: usize,
    ) -> Self {
        let scaling_factor_ = 1u32
            .checked_shl(scaling_exponent as u32)
            .expect("scaling_exponent < 32"); // α = 2^κ
        let shift_ = 1u32
            .checked_shl(shift_exponent as u32)
            .expect("shift_exponent < 32"); // S = 2^s
        let scaled_shift_ = U256::from(scaling_factor_) * U256::from(shift_); // α·S = 2^{κ + s}

        let scaling_factor = api.constant(scaling_factor_); // α as Variable
        let shift = api.constant(shift_); // S as Variable
        let scaled_shift = api.constant(CircuitField::<C>::from_u256(scaled_shift_)); // α·S as Variable

        Self {
            scaling_exponent, // κ
            shift_exponent,   // s
            scaling_factor_,  // α = 2^κ
            shift_,           // S = 2^s
            scaled_shift_,    // α·S = 2^{κ + s}
            scaling_factor,   // α as Variable
            shift,            // S as Variable
            scaled_shift,     // α·S as Variable
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
    let n_bits_q = context
        .shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 fits in usize");
    let q_bits = unconstrained_to_bits(api, shifted_q, n_bits_q);
    let q_recon = assert_is_bitstring_and_reconstruct(api, &q_bits);
    api.assert_is_equal(shifted_q, q_recon);

    // Step 6: Recover quotient q = q^♯ − S
    // let quotient = api.sub(shifted_q, context.shift); // q = q^♯ − S
    let quotient = api.sub(
        shifted_q,
        CircuitField::<C>::from_u256(U256::from(context.shift_ as u64)),
    ); // q = q^♯ − S

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
// FUNCTION: rescale_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies `rescale` elementwise to an `ArrayD<Variable>`, using provided scaling and shift exponents.
///
/// Internally constructs a [`RescalingContext`] with the given exponents:
/// - `scaling_exponent` κ such that α = 2^κ
/// - `shift_exponent` s such that S = 2^s
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder.
/// - `array`: A tensor (of any shape) of `Variable`s to rescale.
/// - `scaling_exponent`: κ for scaling by 2^κ.
/// - `shift_exponent`: s for shifting by 2^s.
/// - `apply_relu`: Whether to apply ReLU after rescaling.
///
/// # Returns
/// An `ArrayD<Variable>` of the same shape with all values rescaled.
pub fn rescale_array<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    array: ArrayD<Variable>,
    scaling_exponent: usize,
    shift_exponent: usize,
    apply_relu: bool,
) -> ArrayD<Variable> {
    let context = RescalingContext::new(api, scaling_exponent, shift_exponent);
    array.map(|x| rescale(api, &context, *x, apply_relu))
}
