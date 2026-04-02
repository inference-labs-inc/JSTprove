// Copyright 2025 Irreducible Inc.
use std::iter;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

/// Represents an arbitrarily large unsigned integer using a vector of `Wire`s
///
/// - Each `Wire` holds a 64-bit unsigned integer value (a "limb")
/// - Limbs are stored in little-endian order (index 0 = least significant)
/// - The total bit width is always a multiple of 64 bits (number of limbs × 64)
#[derive(Clone)]
pub struct BigUint {
    pub limbs: Vec<Wire>,
}

impl BigUint {
    /// Creates a new BigUint with the given number of limbs as inout wires.
    pub fn new_inout(b: &CircuitBuilder, num_limbs: usize) -> Self {
        let limbs = (0..num_limbs).map(|_| b.add_inout()).collect();
        BigUint { limbs }
    }

    /// Creates a new BigUint with the given number of limbs as witness wires.
    pub fn new_witness(b: &CircuitBuilder, num_limbs: usize) -> Self {
        let limbs = (0..num_limbs).map(|_| b.add_witness()).collect();
        BigUint { limbs }
    }

    /// Creates a constant BigUint from num_bigint::BigUint.
    pub fn new_constant(b: &CircuitBuilder, num_biguint: &num_bigint::BigUint) -> Self {
        let limbs = num_biguint
            .iter_u64_digits()
            .map(|limb| b.add_constant_64(limb))
            .collect();
        BigUint { limbs }
    }

    /// Returns zero unless the MSB-boolean `cond` is true, then passes the value unchanged.
    pub fn zero_unless(&self, b: &CircuitBuilder, cond: Wire) -> Self {
        let zero = b.add_constant(Word::ZERO);
        let limbs = self
            .limbs
            .iter()
            .map(|&limb| b.select(cond, limb, zero))
            .collect();
        Self { limbs }
    }

    /// Checks whether BigUint is zero and returns the check result as a boolean wire.
    pub fn is_zero(&self, b: &CircuitBuilder) -> Wire {
        // TODO: it's more efficient to add all-1 BigUint and check carry out (jimpo)
        //       do this once BigUint addition returning a Wire carry out exists
        let zero = b.add_constant(Word::ZERO);
        let msb_one = b.add_constant(Word::MSB_ONE);
        self.limbs
            .iter()
            .map(|&limb| b.icmp_eq(limb, zero))
            .fold(msb_one, |lhs, rhs| b.band(lhs, rhs))
    }

    /// Pads to given limb length with zeros.
    ///
    /// No-op if `new_limbs_len` is shorter then the current one.
    pub fn zero_extend(&self, b: &CircuitBuilder, new_limbs_len: usize) -> Self {
        let zero = b.add_constant(Word::ZERO);
        self.pad_limbs_to(new_limbs_len, zero)
    }

    /// Pads to given limb length with a wire value.
    ///
    /// No-op if `new_limbs_len` is shorter then the current one.
    pub fn pad_limbs_to(&self, new_limbs_len: usize, padding_value: Wire) -> Self {
        let mut padded_limbs = self.limbs.clone();
        if new_limbs_len > padded_limbs.len() {
            padded_limbs.resize(new_limbs_len, padding_value);
        }
        Self {
            limbs: padded_limbs,
        }
    }

    /// Splits the `BigUint` at a given limb position into `(lo, hi)`. The result
    /// satisfies `lo + 2^(WORD_SIZE_BITS * lo.limbs.len()) * hi`.
    pub fn split_at_limbs(mut self, at_limbs: usize) -> (Self, Self) {
        let hi_limbs = self.limbs.split_off(at_limbs);
        (self, Self { limbs: hi_limbs })
    }

    /// Concatenate the limbs of another `BigUint` on top. The resulting value
    /// equals `self + 2^(WORD_SIZE_BITS * self.limbs.len()) * hi`.
    pub fn concat_limbs(&self, hi: &Self) -> Self {
        let mut limbs = self.limbs.clone();
        limbs.extend(&hi.limbs);
        Self { limbs }
    }

    /// Populate the BigUint with the expected limb_values
    ///
    /// Panics if limb_values.len() != self.limbs.len()
    pub fn populate_limbs(&self, w: &mut WitnessFiller, limb_values: &[u64]) {
        assert!(limb_values.len() == self.limbs.len());
        for (&wire, &v) in iter::zip(&self.limbs, limb_values) {
            w[wire] = Word::from_u64(v);
        }
    }
}

/// Asserts that that two `BigUint`s are equal.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand `BigUint`
/// * `b` - Second operand `BigUint` (must have same number of limbs as `a`)
///
/// # Panics
/// Panics if `a` and `b` have different number of limbs.
pub fn assert_eq(builder: &CircuitBuilder, name: impl Into<String>, a: &BigUint, b: &BigUint) {
    assert_eq!(
        a.limbs.len(),
        b.limbs.len(),
        "biguint assert_eq: inputs must have the same number of limbs"
    );
    let base_name = name.into();
    for (i, (&a_l, &b_l)) in iter::zip(&a.limbs, &b.limbs).enumerate() {
        builder.assert_eq(format!("{base_name}[{i}]"), a_l, b_l);
    }
}

/// Conditionally asserts that that two `BigUint`s are equal.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand `BigUint`
/// * `b` - Second operand `BigUint` (must have same number of limbs as `a`)
/// * `cond` - a must equal b if cond is msb-true; constraint is ignored if cond is msb-false
///
/// # Panics
/// Panics if `a` and `b` have different number of limbs.
pub fn assert_eq_cond(
    builder: &CircuitBuilder,
    name: impl Into<String>,
    a: &BigUint,
    b: &BigUint,
    cond: Wire,
) {
    assert_eq!(
        a.limbs.len(),
        b.limbs.len(),
        "biguint assert_eq_cond: inputs must have the same number of limbs"
    );
    let base_name = name.into();
    for (i, (&a_l, &b_l)) in iter::zip(&a.limbs, &b.limbs).enumerate() {
        builder.assert_eq_cond(format!("{base_name}[{i}]"), a_l, b_l, cond);
    }
}

/// Conditionally selects between two equal-sized `BigUint`s.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `cond` - an MSB-boolean
/// * `t` - Value to select when cond is true (MSB=1)
/// * `f` - Value to select when cond is false (MSB=0)
///
/// # Return value
/// Selects `t` if `cond` is true, otherwise selects `f`.
///
/// # Panics
/// Panics if `t` and `f` have different number of limbs.
pub fn select(builder: &CircuitBuilder, cond: Wire, t: &BigUint, f: &BigUint) -> BigUint {
    assert_eq!(
        t.limbs.len(),
        f.limbs.len(),
        "biguint select: inputs must have the same number of limbs"
    );

    let limbs = iter::zip(&t.limbs, &f.limbs)
        .map(|(&l1, &l2)| builder.select(cond, l1, l2))
        .collect();
    BigUint { limbs }
}
