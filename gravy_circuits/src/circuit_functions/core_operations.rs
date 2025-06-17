use expander_compiler::frontend::*;

/// Extract the least significant `n_bits` bits of a given field element or integer represented as a `Variable`,
/// using the Expander Compiler Collection's unconstrained operations.
///
/// This function returns a vector of `Variable`s corresponding to the bits in little-endian order,
/// i.e., `bits[0]` is the least significant bit.
///
/// # Type Parameters
/// - `C`: The circuit configuration (implements `Config`).
/// - `Builder`: The prover API (implements `RootAPI<C>`), providing unconstrained bitwise operations.
///
/// # Parameters
/// - `api`: A mutable reference to the circuit builder.
/// - `input`: The `Variable` whose lower bits are being extracted.
/// - `n_bits`: The number of least significant bits to extract.
///
/// # Returns
/// A `Vec<Variable>` of length `n_bits`, containing the extracted bits in little-endian order.
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


/// Enforce that a little-endian bitstring really is bits ∈ {0,1}, and
/// then reconstruct the integer ∑_{i=0}^{bits.len()−1} bits[i]·2^i.
/// Panics if there’s any overflow in 2^i for i ≥ 32.
/// 
/// # Arguments
/// * `api`                  – circuit builder implementing `RootAPI<C>`.
/// * `least_significant_bits` – slice of Variables to check & combine
/// 
/// # Returns
/// The field element corresponding to the reconstructed integer.
pub fn assert_is_bitstring_and_reconstruct<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    least_significant_bits: &[Variable],
) -> Variable {
    // Start accumulation at zero
    let mut reconstructed = api.constant(0u32);

    for (i, &bit) in least_significant_bits.iter().enumerate() {
        // 1) Enforce bit ∈ {0,1} via vanishing polynomial b(b−1)=0
        let one = api.constant(1u32);
        let bit_minus_one = api.sub(bit, one);
        let vanishing = api.mul(bit, bit_minus_one);
        api.assert_is_zero(vanishing);

        // 2) Add bit · 2^i to the running total
        let weight = 1u32
            .checked_shl(i as u32)
            .expect("bit index i must be < 32");
        let term = api.mul(api.constant(weight), bit);
        reconstructed = api.add(reconstructed, term);
    }

    reconstructed
}


/// # Notation
/// - Let `κ = scaling_exponent`, and define `α = 2^κ`  
/// - Let `s = shift_exponent`, and define `S = 2^s`  
/// - Define `T = 2·S − 1 = 2^(s + 1) − 1`  
/// - `c = dividend`  
/// - `r = remainder`  
/// - `q^♯ = shifted_q`
///
/// Divide out a α = 2^κ “scaling_factor” then subtract a S = 2^s “shift” (all via unconstrained div + mod
/// plus range‐checks), and optionally ReLU the result:
/// 1) form `shifted_dividend = α·S + c`
/// 2) `shifted_dividend = α·q^♯ + r` by unconstrained div+mod  
/// 3) assert exactness of that decomposition  
/// 4) range‐check `r ∈ [0, α - 1] = [0, 2^κ - 1]`  
/// 5) range‐check `q^♯ ∈ [0, T] = [0, 2·S - 1] = [0, 2^(s + 1) - 1]`  
/// 6) recover `q = q^♯ − S`  
/// 7) if `apply_relu`, zero‐out negatives via the MSB of `q^♯`  
///
/// # Panics
/// - if any `checked_shl` or `checked_add` overflows a 32-bit  
///
/// # Arguments
/// * `api`               – circuit builder implementing `RootAPI<C>`.  
/// * `dividend` (`c`)    – a `Variable ≡ original_integer mod p`, assumed in  
///                        `[−α·S, α·(T - S)] = [−α·S, α·(S - 1)] = [-2^(κ + s),2^κ·(2^s - 1)]`  
/// * `scaling_exponent` (`κ`)  – so that `α = 2^κ`  
/// * `shift_exponent`   (`s`)  – so that `S = 2^s`  
/// * `apply_relu`        – if `true`, output `max(q,0)` instead of `q`  
pub fn rescale_by_power_of_two<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    dividend: Variable, // c
    scaling_exponent: usize, // κ
    shift_exponent: usize, // s
    apply_relu: bool,
) -> Variable {
 // 1) compute scaling_factor_ = 2^scaling_exponent (α = 2^κ) and shift_ = 2^shift_exponent (S = 2^s) as u32
let scaling_factor_: u32 = 1u32
    .checked_shl(scaling_exponent as u32)
    .expect("scaling_exponent < 32");
// lift scaling_factor_ (α = 2^κ) to a circuit constant
let scaling_factor = api.constant(scaling_factor_); 
let shift_: u32 = 1u32
    .checked_shl(shift_exponent as u32)
    .expect("shift_exponent < 32");
// lift shift_ (S = 2^s) to a circuit constant
let shift = api.constant(shift_); 

// 2) compute the scaled_shift = scaling_factor_*shift_ (α·S = 2^κ·2^s)
let scaled_shift_: u32 = scaling_factor_
    .checked_mul(shift_)
    .expect("2^scaling_exponent · 2^shift_exponent fits in u32");
// lift scaled_shift_ (α·S = 2^κ·2^s) to a circuit constant
let scaled_shift = api.constant(scaled_shift_); 

// 3) form shifted_dividend = scaled_shift + dividend (α·S + c)
let shifted_dividend = api.add(scaled_shift, dividend);

// 4) unconstrained Euclidean division: shifted_dividend = scaling_factor_*shifted_q + remainder (α·S + c = α·q^♯ + r)
// q^♯ = floor((α·S + c)/α)
let shifted_q = api.unconstrained_int_div(shifted_dividend, scaling_factor_); 
// r = α·S + c - α·q^♯
let remainder = api.unconstrained_mod(shifted_dividend, scaling_factor_); 

// 4b) constrain Euclidean division: shifted_dividend = scaling_factor_*shifted_q + remainder (α·S + c = α·q^♯ + r)
// α·q^♯
let scaling_factor_times_shifted_q = api.mul(scaling_factor, shifted_q); 
// α·q^♯ + r
let recomposed_shifted_dividend = api.add(scaling_factor_times_shifted_q, remainder); 
// α·S + c = α·q^♯ + r
api.assert_is_equal(shifted_dividend, recomposed_shifted_dividend); 

// 5) range‐check remainder r ∈ [0, α - 1] = [0, 2^κ - 1]
// r = d_0 + d_1·2 + ... + d_{κ - 1}·2^(κ - 1) + D_κ·2^κ
let rem_bits = unconstrained_to_bits(api, remainder, scaling_exponent); 
// d_i·(d_i - 1) = 0; rem_recon = d_0 + d_1·2 + ... + d_{κ - 1}·2^(κ - 1)
let rem_recon = assert_is_bitstring_and_reconstruct(api, &rem_bits); 
// r = d_0 + d_1·2 + ... + d_{κ - 1}·2^(κ - 1)
api.assert_is_equal(remainder, rem_recon); 

// 6) range‐check shifted_q q^♯ ∈ [0, T] = [0, 2·S - 1] = [0, 2^(s + 1) - 1]
let n_bits_q = shift_exponent
    .checked_add(1)
    .expect("shift_exponent + 1 fits in usize");
// q^♯ = d_0 + d_1·2 + ... + d_s·2^s + D_{s + 1}·2^(s + 1)
let q_bits = unconstrained_to_bits(api, shifted_q, n_bits_q); 
// d_i·(d_i - 1) = 0; q_recon = d_0 + d_1·2 + ... + d_s·2^s
let q_recon = assert_is_bitstring_and_reconstruct(api, &q_bits); 
// q^♯ = d_0 + d_1·2 + ... + d_{s - 1}·2^s
api.assert_is_equal(shifted_q, q_recon); 

// 7) recover the quotient: shifted_q − 2^s (q = q^♯ - S)
let quotient = api.sub(shifted_q, shift);

// 8) optionally zero out negatives via MSB of shifted_q 
if apply_relu {
    // q ≥ 0 ⇔ q + S ≥ S <=> q^♯ ≥ S = 2^s ⇔ d_s = 1 (since q^♯ ≤ 2^(s + 1) - 1)
    let sign_bit = q_bits[shift_exponent]; // the s-th bit d_s
    // ReLU(q) = d_s·q
    api.mul(quotient, sign_bit) 
} else {
    quotient
}
}

/// Find the maximum element of a nonempty slice of field elements (viewed as integers in [0,p−1]),
/// using only unconstrained comparisons and multiplications.
/// 
/// # Panics
/// - if `values` is empty.
/// 
/// # Arguments
/// * `api`    – circuit builder implementing `RootAPI<C>` 
/// * `values` – slice of `Variable` (each in [0,p−1])  
/// 
/// # Returns
/// A `Variable` holding `max_i values[i]`.
pub fn unconstrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Variable {
    assert!(!values.is_empty(), "unconstrained_max: values must be nonempty");

    // Start with the first element as the current max
    let mut current_max = values[0];

    // Iterate through the rest, updating via unconstrained comparisons
    for &v in &values[1..] {
        // is_greater = 1 if v > current_max, else 0
        let is_greater = api.unconstrained_greater(v, current_max);
        // is_not_greater = 1 if v <= current_max, else 0
        let is_not_greater = api.unconstrained_lesser_eq(v, current_max);

        // pick v when it's larger, otherwise keep current_max
        let take_v = api.unconstrained_mul(v, is_greater);
        let keep_old = api.unconstrained_mul(current_max, is_not_greater);

        // new current_max = take_v + keep_old
        current_max = api.unconstrained_add(take_v, keep_old);
    }

    current_max
}

/// Verifies that a nonempty slice of `Variable`s `values`, each encoding an integer in
/// [-S, T - S] = [−2^s, 2^s − 1], has maximum `M`, by performing the offset‐shift on‐circuit:
///
/// 1. Compute S = 2^s, lifted to a circuit constant.
/// 2. For each x ∈ values, form x + S ∈ [0, T] = [0, 2·S - 1] = [0, 2^(s + 1) - 1].
/// 3. Call `unconstrained_max` on `&[x_offset]` to get M^♯ = max{x + S : x ∈ values} = max{x : x ∈ values} + S = M + S.
/// 4. Recover the (least nonnegative residue of the) true max: M = M^♯ − S.
/// 5. For each original x, compute Δ = M − x, bit‐decompose Δ ∈ [0, T] = [0, 2·S - 1] = [0, 2^(s + 1) - 1].
///    (using s + 1 bits), and recompose to assert exactness.
/// 6. Multiply all Δ’s together and assert the product is zero — ensures at least one
///    Δ = 0, i.e. some x = M.
///
/// # Panics
/// - if `values` is empty.
/// - if any 2^s or s + 1 shift overflows a `u32`.
///
/// # Type Parameters
/// - `C`: the circuit‐field config.
/// - `Builder`: must implement `RootAPI<C>`.
///
/// # Arguments
/// - `api`             – your circuit builder.
/// - `values`          – nonempty slice of `Variable`, each in [-S, T - S] = [−2^s, 2^s − 1].
/// - `shift_exponent` – `s`, so that `offset = 2^s`.
pub fn assert_is_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable], // x_0,...,x_{n - 1}
    shift_exponent: usize, // s
) {
    // 0) require nonempty input
    assert!(
        !values.is_empty(),
        "assert_is_max: `values` slice must be nonempty"
    );

    // 1) compute offset = 2^shift_exponent (S = 2^s) as a u32 and lift into the circuit
    let offset_: u32 = 1u32
        .checked_shl(shift_exponent as u32)
        .expect("shift_exponent < 32");
    let offset = api.constant(offset_);

    // 2) on‐circuit shift: x_offset = x + offset (x^♯ = x + S)
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, offset));
    }

    // 3) unconstrained max over the shifted values (M^♯ = max{x + S : x ∈ values} = max{x : x ∈ values} + S = M + S)
    let max_offset = unconstrained_max(api, &values_offset);

    // 4) recover the (least nonnegative residue of the) true maximum: max_raw = max_offset - offset (M = M^♯ − S)
    let max_raw = api.sub(max_offset, offset);

    // 5) range‐check each Δ = M − x ∈ [0, T] = [0, 2·S − 1] = [0, 2^(s + 1) - 1] via bit‐decomp (need s + 1 bits)
    let n_bits = shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 fits in usize");
    let mut prod = api.constant(1);

    for &x in values {
        // Δ = M − x
        let delta = api.sub(max_raw, x);
        // If M - x ∈ [0, T], then M ≥ x.
        // Conversely, if M ≥ x AND M, x ∈ [-S, T - S], then M - x ∈ [0, T] = [0, 2·S - 1] = [0, 2^(s + 1) - 1].
        // bit‐decompose and recompose Δ = d_0 + d_1·2 + ... + d_s·2^s + D_s·2^(s + 1)
        let bits = unconstrained_to_bits(api, delta, n_bits);
        // recon = d_0 + d_1·2 + ... + d_s·2^s
        let recon = assert_is_bitstring_and_reconstruct(api, &bits);
        // Δ = d_0 + d_1·2 + ... + d_s·2^s
        api.assert_is_equal(delta, recon);

        // accumulate product of all deltas
        prod = api.mul(prod, delta);
    }

    // 6) prod = 0 if and only if Δ = 0 for some x ∈ values, i.e. x = M for some x ∈ values.
    api.assert_is_zero(prod);
}
