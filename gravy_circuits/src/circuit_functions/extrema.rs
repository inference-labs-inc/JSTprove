//! Constraint helpers for extrema-related logic.
//!
//! This module provides reusable constraint functions over finite fields for:
//! - Asserting that a value is the maximum or minimum of a list
//! - Enforcing ReLU constraints as a special case of max
//! - Performing base-`b` digit decompositions and range checks
//! - Constructing multiplicative zero-test constraints to enforce equality to some candidate
//!
//! These functions are intended for use when building arithmetic circuits using the
//! Expander Compiler Collection, and assume that all values live in a prime field.
//!
//! All functions operate on `Variable`s and are compatible with the `RootAPI` trait.
//!
//! This module does **not** define circuits itself — it provides constraint logic for use
//! in circuit construction elsewhere.

// Import core types and traits for constraint construction.
// Includes `Variable`, `Config`, `RootAPI`, and standard circuit APIs.
use expander_compiler::frontend::*;

// Import 256-bit unsigned integer type used for handling large field elements or I/O serialization.
// Useful when converting to/from field representations or interfacing with external formats.
use ethnum::U256;

// Import lookup-based range proof structure.
// Used to enforce that a variable lies in a bounded range via lookup table instead of bit decomposition.
use circuit_std_rs::logup::LogUpRangeProofTable;

/// Converts a signed 32-bit integer to its field element representation,
/// using the least-residue mapping. If the value is negative, its absolute
/// value is converted to a field element and then negated.
///
/// # Arguments
/// * `api` - A mutable reference to an object implementing the `RootAPI` trait.
/// * `value` - A signed 32-bit integer (i32) to be converted.
///
/// # Returns
/// A `Variable` representing the field element corresponding to `value`.
pub fn signed_to_field_small<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    value: i32,
) -> Variable {
    if value < 0 {
        let abs_val = C::CircuitField::from(value.unsigned_abs());
        let var = api.constant(abs_val);
        api.neg(var)
    } else {
        api.constant(C::CircuitField::from(value as u32))
    }
}

/// Converts a signed 64-bit integer to its field element representation,
/// using a 256-bit unsigned integer (U256) for conversion. This version is
/// safe for values that might exceed 32-bit range.
/// 
/// # Arguments
/// * `api` - A mutable reference to an object implementing the `RootAPI` trait.
/// * `value` - A signed 64-bit integer (i64) to be converted.
///
/// # Returns
/// A `Variable` representing the field element corresponding to `value`.
pub fn signed_to_field_any<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    value: i64,
) -> Variable {
    if value < 0 {
        let abs_val = C::CircuitField::from_u256(U256::from(value.unsigned_abs()));
        let var = api.constant(abs_val);
        api.neg(var)
    } else {
        api.constant(C::CircuitField::from_u256(U256::from(value as u64)))
    }
}

/// Converts a 2D matrix (vector of vectors) of signed integers into a matrix of field elements,
/// assuming that all values fit within a 32-bit signed integer range. This version uses 
/// `signed_to_field_small` and casts each element from i64 to i32.
///
/// # Arguments
/// * `api` - A mutable reference to an object implementing the `RootAPI` trait.
/// * `matrix` - A 2D vector (`Vec<Vec<i64>>`) of signed integers.
///
/// # Returns
/// A 2D vector of `Variable` representing the field elements.
/// 
/// # Note
/// This helper assumes that all values in the matrix are small enough to be represented as i32.
pub fn signed_matrix_to_field_small<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix: Vec<Vec<i64>>,
) -> Vec<Vec<Variable>> {
    matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|x| signed_to_field_small::<C, Builder>(api, x as i32))
                .collect()
        })
        .collect()
}

/// Converts a 2D matrix (vector of vectors) of signed 64-bit integers into a matrix of field elements,
/// using the `signed_to_field_any` helper which supports the full 64-bit range via U256 conversion.
///
/// # Arguments
/// * `api` - A mutable reference to an object implementing the `RootAPI` trait.
/// * `matrix` - A 2D vector (`Vec<Vec<i64>>`) of signed 64-bit integers.
///
/// # Returns
/// A 2D vector of `Variable` representing the field elements.
pub fn signed_matrix_to_field_any<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix: Vec<Vec<i64>>,
) -> Vec<Vec<Variable>> {
    matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|x| signed_to_field_any::<C, Builder>(api, x))
                .collect()
        })
        .collect()
}

/// Asserts that `target` is equal to at least one element in `candidates`.
/// 
/// This is enforced by constructing the multiplicative zero-testing polynomial:
/// 
///     ∏_{j=0}^{ℓ-1} (target - candidates[j]) = 0
/// 
/// If this product is zero, then at least one factor must be zero, meaning that
/// `target == candidates[j]` for some j.
/// 
/// # Parameters
/// - `api`: mutable reference to a Builder implementing `RootAPI`
/// - `target`: the Variable that is claimed to be equal to one of the candidates
/// - `candidates`: slice of Variables among which `target` must equal at least one
pub fn assert_equal_to_some<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    target: Variable,
    candidates: &[Variable],
) {
    let mut product = api.constant(1);
    for &a in candidates {
        // Compute (target - a) for each candidate.
        let delta = api.sub(target, a);
        // Multiply into the cumulative product.
        product = api.mul(product, delta);
    }
    // Enforce that the product is exactly zero.
    api.assert_is_zero(product);
}

/// Create a LogUpRangeProofTable for the given base.
/// Computes the number of bits required to represent values in `{0, ..., base - 1}`
/// and initializes a lookup table suitable for range checks.
pub fn make_logup_table_for_base(base: u32) -> LogUpRangeProofTable {
    let nb_bits = (32 - base.leading_zeros()) as usize;
    LogUpRangeProofTable::new(nb_bits)
}

// --- new helper -------------------------------------------------------------
// use expander_compiler::hints::{HintRegistry, StubHintCaller};

/// Registers the single HintFn that LogUp tables need.
// pub fn register_logup_hints<F: Field>(reg: &mut HintRegistry<F>) {
//     circuit_std_rs::logup::LogUpRangeProofTable::register_builtins(reg);
// }


/// Asserts that a given `digit` lies in the range `{0, 1, ..., base - 1}`.
///
/// This function provides two ways to enforce range membership:
///
/// - If `use_lookup` is `false`, it constructs the set `{0, ..., base - 1}` as constants
///   and asserts that `digit` is equal to one of them using `assert_equal_to_some`.
///
/// - If `use_lookup` is true, uses a LogUp-based range proof table.
///   The `table` parameter is taken as a mutable reference to an Option, so that it
///   can be reborrowed without moving it. Otherwise, a fresh table is constructed on the fly
///   using the bitlength of `base`.
///
/// # Parameters
/// - `api`: mutable reference to a builder implementing `RootAPI`
/// - `digit`: the variable to be range-checked
/// - `base`: the upper bound of the digit range (exclusive); valid digits are `< base`
/// - `use_lookup`: whether to use a lookup-based range check or a polynomial constraint
/// - `table`: optional mutable reference to a `LogUpRangeProofTable` (required if reusing tables)
pub fn assert_is_base_b_digit<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    digit: Variable,
    base: u32,
    use_lookup: bool,
    table: &mut Option<&mut LogUpRangeProofTable>,
) {
    if use_lookup {
        if let Some(t) = table.as_mut() {
            // Use the provided table.
            t.rangeproof(api, digit, base as usize);
        } else {
            // Construct a temporary table on the fly.
            let nb_bits = (32 - base.leading_zeros()) as usize;
            let mut temp_table = LogUpRangeProofTable::new(nb_bits);
            temp_table.rangeproof(api, digit, base as usize);
        }
    } else {
        // Fallback using the naïve polynomial constraint.
        let allowed: Vec<Variable> = (0..base).map(|v| api.constant(v)).collect();
        assert_equal_to_some(api, digit, &allowed);
    }
}

/// Decomposes a nonnegative integer (provided as a circuit Variable) into its
/// `max_digits` least-significant digits in the given `base`.
///
/// This function uses unconstrained integer division and modulo operations:
/// 
/// - It first converts the provided base (a small number, e.g. 10 or 16) to a field constant.
/// - Then, in each iteration, it extracts the least-significant digit of `nonnegative_integer`
///   (using modulo), and updates the number using integer division.
///
/// The digits are returned in little-endian order (least-significant digit first).
///
/// # Arguments
///
/// * `api` - A mutable reference to an object implementing `RootAPI`. This object provides
///   methods (e.g., `constant`, `unconstrained_mod`, `unconstrained_int_div`) to build circuit constraints.
/// * `nonnegative_integer` - A Variable representing the nonnegative integer input to be decomposed.
/// * `base` - The base as a `u32`. Typically, this is a small number like 10 or 16.
/// * `max_digits` - The number of digits (κ) to extract from the input number.
/// 
/// # Returns
///
/// A vector of Variables representing the extracted digits. The vector is in little-endian order:
/// the first element is the least-significant digit.
pub fn to_base_b_digits<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    mut nonnegative_integer: Variable,
    base: u32,
    max_digits: usize,
) -> Vec<Variable> {
    // Convert the base to a field constant; this allows us to perform operations
    // on the base within the circuit.
    let base_var = api.constant(base);
    let mut digits = Vec::with_capacity(max_digits);

    for _ in 0..max_digits {
        // Extract the least-significant digit using unconstrained modulo with base_var.
        let digit = api.unconstrained_mod(nonnegative_integer, base_var);
        // Compute the quotient by integer division using base_var.
        let quotient = api.unconstrained_int_div(nonnegative_integer, base_var);
        digits.push(digit);
        nonnegative_integer = quotient;
    }

    digits
}

/// Reconstructs an integer from its base-`base` digits provided in little-endian order.
/// 
/// This function multiplies each digit by the corresponding power of `base` and sums the results.
/// 
/// # Parameters
/// - `api`: mutable reference to a Builder implementing `RootAPI`
/// - `digits`: slice of Variables representing the digits (least significant first)
/// - `base`: the base used for the digit representation
/// 
/// # Returns
/// The reconstructed integer as a `Variable`.
pub fn from_base_b_digits<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    digits: &[Variable],
    base: u32,
) -> Variable {
    let mut reconstructed = api.constant(0);
    let mut power = api.constant(1);
    let base_var = api.constant(base);

    // Sum up digit * (base^index) for each digit.
    for &digit in digits {
        let term = api.mul(power, digit);
        reconstructed = api.add(reconstructed, term);
        power = api.mul(power, base_var);
    }

    reconstructed
}

/// Performs a base-`b` range check on `nonnegative_integer` by decomposing it into digits,
/// reconstructing it, and enforcing that each digit lies in `{0, ..., base - 1}`.
///
/// This function verifies that `nonnegative_integer ∈ [0, base^max_digits)` by:
/// 1. Decomposing it into at most `max_digits` digits in base `base` (little-endian),
/// 2. Reconstructing the value from its digits and asserting equality with the original,
/// 3. Constraining each digit to lie in the valid range.
///
/// The digit constraint can be enforced using either:
/// - A naïve polynomial constraint (default if `use_lookup` is `false`),
/// - A lookup-based range proof using `LogUpRangeProofTable` (if `use_lookup` is `true`).
///
/// An optional `table` parameter can be supplied to reuse a preconstructed lookup table
/// for efficiency. The `table` parameter is a mutable reference to an Option: if Some,
/// a preconstructed LogUp table is reused; if None, table is constructed on the fly.
///
/// # Parameters
/// - `api`: mutable reference to a builder implementing `RootAPI`
/// - `nonnegative_integer`: the variable to range-check
/// - `base`: the base used for digit decomposition
/// - `max_digits`: the maximum number of base-`base` digits
/// - `use_lookup`: whether to use a lookup table or polynomial constraint for digit range checking
/// - `table`: optional mutable reference to a reusable `LogUpRangeProofTable`
///
/// # Returns
/// A vector of base-`base` digits (in little-endian order) representing `nonnegative_integer`.
pub fn range_check_base_b<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    nonnegative_integer: Variable,
    base: u32,
    max_digits: usize,
    use_lookup: bool,
    table: &mut Option<&mut LogUpRangeProofTable>,
) -> Vec<Variable> {
    let digits = to_base_b_digits(api, nonnegative_integer, base, max_digits);
    let reconstructed = from_base_b_digits(api, &digits, base);

    // Enforce equality between the original integer and its reconstruction.
    api.assert_is_equal(nonnegative_integer, reconstructed);

    // For each digit, ensure it is in {0, …, base - 1} using our range check.
    for &digit in &digits {
        assert_is_base_b_digit(api, digit, base, use_lookup, table);
    }

    digits
}

/// Range-checks the gap between the claimed extremum and each candidate.
/// 
/// For each candidate value `a`, if checking for maximum (`is_max = true`), the gap is computed as (extremum - a);
/// if checking for minimum (`is_max = false`), as (a - extremum). Then each gap is range-checked to ensure it is less than base^(num_digits).
/// 
/// # Parameters
/// - `api`: mutable reference to a Builder implementing `RootAPI`
/// - `extremum`: the Variable representing the claimed extremum (either maximum or minimum)
/// - `candidates`: slice of Variables among which the extremum is claimed
/// - `base`: the base used for range checking
/// - `num_digits`: the number of digits (κ) used to represent the allowable gap (i.e., the gap is in [0, base^(num_digits)))
/// - `is_max`: flag indicating whether the check is for a maximum (true) or a minimum (false)
pub fn range_check_extrema_gap<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    extremum: Variable,
    candidates: &[Variable],
    base: u32,
    num_digits: usize,
    is_max: bool,
    use_lookup: bool,
    table: &mut Option<&mut LogUpRangeProofTable>,
) {
    for &a in candidates {
        let gap = if is_max {
            api.sub(extremum, a)
        } else {
            api.sub(a, extremum)
        };

        let _ = range_check_base_b(api, gap, base, num_digits, use_lookup, table);
    }
}

// pub fn range_check_extrema_gap<C: Config, Builder: RootAPI<C>>(
//     api: &mut Builder,
//     extremum: Variable,
//     candidates: &[Variable],
//     base: u32,
//     num_digits: usize,
//     is_max: bool,
// ) {
//     // Even if not using lookup, we still provide a mutable Option.
//     let mut table: Option<&mut LogUpRangeProofTable> = None;
    
//     for &a in candidates {
//         // Compute the gap: for maximum, gap = extremum - a_i; for minimum, gap = a_i - extremum.
//         let gap = if is_max {
//             api.sub(extremum, a)
//         } else {
//             api.sub(a, extremum)
//         };
//         // Call our range check function; since use_lookup is false here, our table parameter is ignored.
//         let _ = range_check_base_b(api, gap, base, num_digits, false, &mut table);
//     }
// }

/// Verifies that `extremum` is indeed the extreme (maximum or minimum) of the provided `candidates`.
/// 
/// This function enforces two properties:
/// 
/// 1. **Attainment:** It asserts that `extremum` is equal to at least one of the candidate values.
/// 2. **Extremal Gap:** For every candidate `a_i`, it range-checks the gap between `extremum` and `a_i`.
///    - If checking for a maximum, it checks that `extremum - a_i` fits in the range [0, base^(num_digits)).
///    - If checking for a minimum, it checks that `a_i - extremum` fits in the range [0, base^(num_digits)).
/// 
/// # Parameters
/// - `api`: mutable reference to a Builder implementing `RootAPI`
/// - `extremum`: the Variable representing the claimed maximum or minimum
/// - `candidates`: slice of Variables among which the extremum is claimed
/// - `base`: the base for the digit decomposition and range checking
/// - `num_digits`: the number of digits used for the gap range constraint
/// - `is_max`: set to true if verifying a maximum; false if verifying a minimum.
pub fn assert_extremum<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    extremum: Variable,
    candidates: &[Variable],
    base: u32,
    num_digits: usize,
    is_max: bool,
    use_lookup: bool,
    table: &mut Option<&mut LogUpRangeProofTable>,
) {
    // Enforce that the claimed extremum equals at least one candidate.
    assert_equal_to_some(api, extremum, candidates);
    
    // For each candidate, range-check the gap between the extremum and the candidate.
    range_check_extrema_gap(api, extremum, candidates, base, num_digits, is_max, use_lookup, table);
}

/// Enforces that `relu_output` is the ReLU of `relu_input`, i.e., relu_output = max(relu_input, 0).
///
/// This is done by asserting that `relu_output` is the maximum of the set {relu_input, 0},
/// and verifying that it satisfies the required range constraints.
///
/// # Parameters
/// - `api`: mutable reference to a Builder implementing `RootAPI`
/// - `relu_input`: the input value to ReLU (can be negative)
/// - `relu_output`: the claimed result of ReLU
/// - `base`: the numeric base used for range checks
/// - `num_digits`: the number of digits (κ) used to constrain the range of differences
pub fn assert_relu<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    relu_input: Variable,
    relu_output: Variable,
    base: u32,
    num_digits: usize,
    use_lookup: bool,
    table: &mut Option<&mut LogUpRangeProofTable>, 
) {
    let zero = api.constant(0);
    let candidates = [relu_input, zero];
    assert_extremum(api, relu_output, &candidates, base, num_digits, true, use_lookup, table);
}