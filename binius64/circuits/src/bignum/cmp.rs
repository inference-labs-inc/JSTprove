// Copyright 2025 Irreducible Inc.
use std::iter;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::biguint::BigUint;

/// Less-than comparison between equally-sized `BigUint`s.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand
/// * `b` - Second operand (must have same number of limbs as `a`)
///
/// # Returns
/// Boolean wire that is true when `a < b`.
///
/// # Panics
/// - Panics if `a` and `b` have different number of limbs
pub fn biguint_lt(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> Wire {
    assert_eq!(
        a.limbs.len(),
        b.limbs.len(),
        "biguint_lt: inputs must have the same number of limbs"
    );

    let mut result = builder.add_constant(Word::ZERO);

    for (&a_limb, &b_limb) in iter::zip(&a.limbs, &b.limbs) {
        let lt_flag = builder.icmp_ult(a_limb, b_limb);
        let eq_flag = builder.icmp_eq(a_limb, b_limb);
        result = builder.bor(lt_flag, builder.band(eq_flag, result));
    }

    result
}

/// Equality check between equally-sized `BigUint`s.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand
/// * `b` - Second operand (must have same number of limbs as `a`)
///
/// # Returns
/// Boolean wire that is true when `a == b`.
///
/// # Panics
/// - Panics if `a` and `b` have different number of limbs
pub fn biguint_eq(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> Wire {
    assert_eq!(
        a.limbs.len(),
        b.limbs.len(),
        "biguint_eq: inputs must have the same number of limbs"
    );

    let mut result = builder.add_constant(Word::ALL_ONE);

    for (&a_limb, &b_limb) in iter::zip(&a.limbs, &b.limbs) {
        result = builder.band(builder.icmp_eq(a_limb, b_limb), result);
    }

    result
}
