// Copyright 2025 Irreducible Inc.
use binius_core::{consts::WORD_SIZE_BITS, word::Word};
use binius_frontend::{CircuitBuilder, Wire, util::num_biguint_from_u64_limbs};

use super::{
    BigUint, PseudoMersenneModReduce, add, biguint_lt, sub, textbook_mul, textbook_square,
};

/// A struct that implements prime field arithmetic over pseudo-Mersenne modulus.
///
/// Field elements are `BigUint`s consisting of `PseudoMersennePrimeField::limbs_len` limbs.
/// It is assumed that all field elements are correctly represented (less than modulus).
pub struct PseudoMersennePrimeField {
    modulus: BigUint,
    modulus_po2: usize,
    modulus_subtrahend: BigUint,
}

impl PseudoMersennePrimeField {
    /// Create a new pseudo-Mersenne prime field.
    ///
    /// See [`PseudoMersenneModReduce`] for description of the parameters.
    pub fn new(b: &CircuitBuilder, modulus_po2: usize, modulus_subtrahend: &[u64]) -> Self {
        let modulus_subtrahend = num_biguint_from_u64_limbs(modulus_subtrahend);
        let po2 = num_bigint::BigUint::from(2usize).pow(modulus_po2 as u32);
        assert!(
            po2 > modulus_subtrahend,
            "2^modulus_po2 - modulus_subtrahend > 0"
        );

        let modulus = BigUint::new_constant(b, &(po2 - &modulus_subtrahend));
        let modulus_subtrahend = BigUint::new_constant(b, &modulus_subtrahend);

        Self {
            modulus,
            modulus_po2,
            modulus_subtrahend,
        }
    }

    /// Number of limbs in `BigUint`s representing field elements.
    pub fn limbs_len(&self) -> usize {
        self.modulus.limbs.len()
    }

    /// Field modulus.
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Field addition.
    ///
    /// Equivalent formula: `(fe1 + fe2) % modulus`
    pub fn add(&self, b: &CircuitBuilder, fe1: &BigUint, fe2: &BigUint) -> BigUint {
        let l = self.limbs_len();
        assert!(fe1.limbs.len() == l && fe2.limbs.len() == l);

        let zero = b.add_constant(Word::ZERO);

        // May need an extra limb to accommodate overflow.
        let extra_limbs = if self.modulus_po2 + 1 > l * WORD_SIZE_BITS {
            1
        } else {
            0
        };

        let fe1 = fe1.pad_limbs_to(l + extra_limbs, zero);
        let fe2 = fe2.pad_limbs_to(l + extra_limbs, zero);
        let modulus = self.modulus.pad_limbs_to(l + extra_limbs, zero);

        let unreduced_sum = add(b, &fe1, &fe2);
        // TODO: consider nondeterminism
        let need_reduction = b.bnot(biguint_lt(b, &unreduced_sum, &modulus));
        let reduced = sub(b, &unreduced_sum, &modulus.zero_unless(b, need_reduction));

        // Higher limb is zero if max(fe1, fe2) < modulus (both are correctly represented)
        let (result, _) = reduced.split_at_limbs(l);
        result
    }

    /// Field subtraction.
    ///
    /// Equivalent formula: `(fe1 - fe2) % modulus`
    pub fn sub(&self, b: &CircuitBuilder, fe1: &BigUint, fe2: &BigUint) -> BigUint {
        assert!(fe1.limbs.len() == self.limbs_len() && fe2.limbs.len() == self.limbs_len());
        // NB: fe1 - fe2 = fe1 + modulus - fe2 <= 2*modulus - 1 (one subtraction still normalizes)
        let fe2_add_inv = sub(b, &self.modulus, fe2);
        self.add(b, fe1, &fe2_add_inv)
    }

    /// Field squaring.
    ///
    /// Equivalent formula: `(fe ** 2) % modulus`
    pub fn square(&self, b: &CircuitBuilder, fe: &BigUint) -> BigUint {
        assert!(fe.limbs.len() == self.limbs_len());
        self.reduce_product(b, textbook_square(b, fe))
    }

    /// Field multiplication.
    ///
    /// Equivalent formula: `(fe1 * fe2) % modulus`
    /// Note: Both fe1 and fe2 may be greater or equal to modulus.
    pub fn mul(&self, b: &CircuitBuilder, fe1: &BigUint, fe2: &BigUint) -> BigUint {
        assert!(fe1.limbs.len() == self.limbs_len() && fe2.limbs.len() == self.limbs_len());
        self.reduce_product(b, textbook_mul(b, fe1, fe2))
    }

    fn reduce_product(&self, b: &CircuitBuilder, product: BigUint) -> BigUint {
        let (quotient, remainder) = b.biguint_divide_hint(&product.limbs, &self.modulus.limbs);

        let zero = b.add_constant(Word::ZERO);

        let quotient = BigUint { limbs: quotient };
        let remainder = BigUint { limbs: remainder }.pad_limbs_to(self.limbs_len(), zero);

        b.assert_true(
            "remainder < modulus",
            biguint_lt(b, &remainder, &self.modulus),
        );

        // constraint: product == remainder + quotient * modulus
        PseudoMersenneModReduce::new(
            b,
            &product,
            self.modulus_po2,
            &self.modulus_subtrahend,
            &quotient,
            &remainder,
        )
        .constrain(b);

        remainder
    }

    /// Field inverse.
    ///
    /// Equivalent formula (for prime modulus): `(fe1 ** (modulus - 2)) % modulus`
    /// The wire parameter `exists` is a boolean-wire signifying the existence of the inverse;
    /// if `exists` is false, the modular reduction constraint is not applied. This is useful
    /// for avoiding overconstraining in skipped parts of larger circuits.
    pub fn inverse(&self, b: &CircuitBuilder, fe: &BigUint, exists: Wire) -> BigUint {
        assert!(fe.limbs.len() == self.limbs_len());
        let (quotient, inverse) = b.mod_inverse_hint(&fe.limbs, &self.modulus.limbs);

        let zero = b.add_constant(Word::ZERO);

        let quotient = BigUint { limbs: quotient };
        let inverse = BigUint { limbs: inverse }.pad_limbs_to(self.limbs_len(), zero);
        let one = BigUint::new_constant(b, &num_bigint::BigUint::from(1usize))
            .pad_limbs_to(self.limbs_len(), zero);

        let product = textbook_mul(b, &inverse, fe);

        b.assert_true("inverse < modulus", biguint_lt(b, &inverse, &self.modulus));

        // constraint: base * inverse = 1 + quotient * modulus
        PseudoMersenneModReduce::new(
            b,
            &product,
            self.modulus_po2,
            &self.modulus_subtrahend,
            &quotient,
            &one,
        )
        .constrain_cond(b, exists);

        inverse
    }
}
