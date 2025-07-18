use expander_compiler::frontend::*;
use ethnum::U256;
use super::utils_core_math::{unconstrained_to_bits, assert_is_bitstring_and_reconstruct};

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: RescalingContext
// ─────────────────────────────────────────────────────────────────────────────

/// Holds integer and circuit-level constants for rescaling by `2^κ` and shifting by `2^s`.
pub struct RescalingContext {
    pub scaling_exponent: usize,   // κ
    pub shift_exponent: usize,     // s
    pub scaling_factor_: u32,      // α = 2^κ
    pub shift_: u32,               // S = 2^s
    pub scaled_shift_: U256,       // α·S = 2^{κ + s} (could overflow u32)

    pub scaling_factor: Variable,
    pub shift: Variable,
    pub scaled_shift: Variable,
}

impl RescalingContext {
    pub fn new<C:Config, Builder: RootAPI<C>>(api: &mut Builder, scaling_exponent: usize, shift_exponent: usize) -> Self {
        let scaling_factor_ = 1u32.checked_shl(scaling_exponent as u32).expect("scaling_exponent < 32");
        let shift_ = 1u32.checked_shl(shift_exponent as u32).expect("shift_exponent < 32");
        let scaled_shift_ = U256::from(scaling_factor_)*U256::from(shift_);

        let scaling_factor = api.constant(scaling_factor_);
        let shift = api.constant(shift_);
        let scaled_shift = api.constant(
            CircuitField::<C>::from_u256(scaled_shift_)
        );

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

pub fn rescale_2d_vector<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_exponent: usize,
    shift_exponent: usize,
    apply_relu: bool,
) -> Vec<Vec<Variable>> {
    let context = RescalingContext::new(api, scaling_exponent, shift_exponent);
    let mut output_matrix = Vec::with_capacity(input_matrix.len());

    for row in input_matrix {
        let mut output_row = Vec::with_capacity(row.len());
        for &value in &row {
            output_row.push(rescale(api, &context, value, apply_relu));
        }
        output_matrix.push(output_row);
    }

    output_matrix
}

pub fn rescale_4d_vector<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_tensor: Vec<Vec<Vec<Vec<Variable>>>>,
    scaling_exponent: usize,
    shift_exponent: usize,
    apply_relu: bool,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let context = RescalingContext::new(api, scaling_exponent, shift_exponent);
    let mut output_tensor = Vec::with_capacity(input_tensor.len());

    for dim1 in input_tensor {
        let mut dim1_out = Vec::with_capacity(dim1.len());
        for dim2 in dim1 {
            let mut dim2_out = Vec::with_capacity(dim2.len());
            for dim3 in dim2 {
                let mut dim3_out = Vec::with_capacity(dim3.len());
                for &element in &dim3 {
                    dim3_out.push(rescale(api, &context, element, apply_relu));
                }
                dim2_out.push(dim3_out);
            }
            dim1_out.push(dim2_out);
        }
        output_tensor.push(dim1_out);
    }

    output_tensor
}
