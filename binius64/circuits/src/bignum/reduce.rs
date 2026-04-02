// Copyright 2025 Irreducible Inc.
use binius_core::{consts::WORD_SIZE_BITS, word::Word};
use binius_frontend::{CircuitBuilder, Wire};

use super::{
    addsub::{add, sub},
    biguint::{BigUint, assert_eq, assert_eq_cond},
    mul::{optimal_mul, textbook_mul},
};

/// Modular reduction verification for BigUint.
///
/// This circuit verifies that:
///
/// a = quotient * modulus + remainder
pub struct ModReduce {
    pub a: BigUint,
    pub modulus: BigUint,
    pub quotient: BigUint,
    pub remainder: BigUint,
}

impl ModReduce {
    /// Creates a new modular reduction verifier circuit.
    ///
    /// # Arguments
    /// * `builder` - Circuit builder for constraint generation
    /// * `a` - The dividend
    /// * `modulus` - The divisor
    /// * `quotient` - The quotient
    /// * `remainder` - The remainder
    ///
    /// # Constraints
    /// The circuit enforces that `a = quotient * modulus + remainder`
    pub fn new(
        builder: &CircuitBuilder,
        a: BigUint,
        modulus: BigUint,
        quotient: BigUint,
        remainder: BigUint,
    ) -> Self {
        let zero = builder.add_constant(Word::ZERO);

        let product = optimal_mul(builder, &quotient, &modulus);

        let remainder_padded = remainder.pad_limbs_to(product.limbs.len(), zero);
        let reconstructed = add(builder, &product, &remainder_padded);

        let n_limbs = reconstructed.limbs.len().max(a.limbs.len());
        assert_eq(
            builder,
            "modreduce_a_eq_reconstructed",
            &reconstructed.pad_limbs_to(n_limbs, zero),
            &a.pad_limbs_to(n_limbs, zero),
        );

        ModReduce {
            a,
            modulus,
            quotient,
            remainder,
        }
    }
}

/// Modular reduction verification for BigUint for pseudo Mersenne moduli.
///
/// This circuit verifies that:
///
/// a = quotient * (2^modulus_po2 - modulus_subtrahend) + remainder
///
/// where modulus_po2 is additionally restricted to be a multiple of limb size to only
/// split BigUint at limb boundaries.
///
/// This algorithm is more efficient than `ModReduce` when `modulus_subtrahend` is a short
/// compared to `modulus_po2`. This is the case for many practically interesting prime field.
pub struct PseudoMersenneModReduce {
    lhs: BigUint,
    rhs: BigUint,
}

impl PseudoMersenneModReduce {
    /// Creates a new pseudo Mersenne modular reduction verifier circuit.
    ///
    /// # Arguments
    /// * `builder` - Circuit builder for constraint generation
    /// * `a` - The dividend
    /// * `modulus_po2` - the power of two modulus minuend (has to be a multiple of
    ///   `WORD_SIZE_BITS`)
    /// * `modulus_subtrahend` - the value subtracted form `2^modulus_po2` to obtain modulus
    /// * `quotient` - The quotient
    /// * `remainder` - The remainder
    ///
    /// # Constraints
    /// The circuit enforces that `a = quotient * (2^modulus_po2 - modulus_subtrahend) + remainder`.
    /// Remainder range check (`0 <= remainder < 2^modulus_po2 - modulus_subtrahend`) is _not_
    /// enforced.
    ///
    /// Note: This adds arithmetic constraints for computing intermediate values
    /// (multiplication, addition, subtraction), but does NOT add the final equality
    /// constraint. You must call `.constrain()` or `.constrain_cond()` to enforce
    /// that the equation actually holds.
    #[must_use]
    pub fn new(
        builder: &CircuitBuilder,
        a: &BigUint,
        modulus_po2: usize,
        modulus_subtrahend: &BigUint,
        quotient: &BigUint,
        remainder: &BigUint,
    ) -> Self {
        // a = quotient * (2^modulus_po2 - modulus_subtrahend) + remainder
        // hi * 2^modulus_po2 + lo = quotient * (2^modulus_po2 - modulus_subtrahend) + remainder
        // lo + quotient * modulus_subtrahend = remainder + 2^modulus_po2 * (quotient - hi)
        // max(lo, remainder) < 2^modulus_po2
        // quotient < |a/(2^modulus_po2 - modulus_subtrahend)|
        // quotient >= hi
        assert!(modulus_po2.is_multiple_of(WORD_SIZE_BITS));
        assert!(modulus_subtrahend.limbs.len() * WORD_SIZE_BITS <= modulus_po2);
        assert!(remainder.limbs.len() * WORD_SIZE_BITS <= modulus_po2);

        let zero = builder.add_constant(Word::ZERO);

        let n_lo_limbs = modulus_po2 / WORD_SIZE_BITS;

        let (a_lo, a_hi) = a.pad_limbs_to(n_lo_limbs, zero).split_at_limbs(n_lo_limbs);

        let rhs_hi = sub(
            builder,
            quotient,
            &a_hi.pad_limbs_to(quotient.limbs.len(), zero),
        );
        let rhs = remainder
            .pad_limbs_to(n_lo_limbs, zero)
            .concat_limbs(&rhs_hi);

        let quotient_modulus_subtrahend = textbook_mul(builder, quotient, modulus_subtrahend);
        let lhs_rhs_len = [
            rhs.limbs.len(),
            quotient_modulus_subtrahend.limbs.len() + 1,
            a_lo.limbs.len() + 1,
        ]
        .into_iter()
        .max()
        .expect("exactly 3 elements");

        let lhs = add(
            builder,
            &a_lo.pad_limbs_to(lhs_rhs_len, zero),
            &quotient_modulus_subtrahend.pad_limbs_to(lhs_rhs_len, zero),
        );

        let rhs = rhs.pad_limbs_to(lhs_rhs_len, zero);

        Self { lhs, rhs }
    }

    /// Apply the reduction constraint unconditionally.
    pub fn constrain(self, builder: &CircuitBuilder) {
        assert_eq(builder, "modreduce_pseudo_mersenne", &self.lhs, &self.rhs);
    }

    /// Apply the reduction constraint conditionally based on the value of boolean `mask` wire.
    pub fn constrain_cond(self, builder: &CircuitBuilder, cond: Wire) {
        assert_eq_cond(
            builder,
            "modred_pseudo_mersenne",
            &self.lhs,
            &self.rhs,
            cond,
        )
    }
}
