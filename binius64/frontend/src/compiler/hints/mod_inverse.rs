// Copyright 2025 Irreducible Inc.
//! Modular inverse hint implementation

use binius_core::Word;

use super::Hint;
use crate::util::num_biguint_from_u64_limbs;

/// ModInverse hint implementation
pub struct ModInverseHint;

impl ModInverseHint {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ModInverseHint {
    fn default() -> Self {
        Self::new()
    }
}

impl Hint for ModInverseHint {
    fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
        let [base_limbs, mod_limbs] = dimensions else {
            panic!("ModInverse requires 2 dimensions");
        };
        (*base_limbs + *mod_limbs, 2 * *mod_limbs)
    }

    fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
        let [n_base, n_mod] = dimensions else {
            panic!("ModInverse requires 2 dimensions");
        };

        let base_limbs = &inputs[0..*n_base];
        let mod_limbs = &inputs[*n_base..];

        let base = num_biguint_from_u64_limbs(base_limbs.iter().map(|w| w.as_u64()));
        let modulus = num_biguint_from_u64_limbs(mod_limbs.iter().map(|w| w.as_u64()));

        let zero = num_bigint::BigUint::ZERO;
        let (quotient, inverse) = if let Some(inverse) = base.modinv(&modulus) {
            let quotient = (base * &inverse - num_bigint::BigUint::from(1usize)) / &modulus;
            (quotient, inverse)
        } else {
            (zero.clone(), zero)
        };

        assert_eq!(outputs.len(), 2 * *n_mod);
        let (quotient_words, inverse_words) = outputs.split_at_mut(*n_mod);

        // Fill output quotient limbs
        for (i, limb) in quotient.iter_u64_digits().enumerate() {
            quotient_words[i] = Word::from_u64(limb);
        }

        // Zero remaining outputs if quotient has fewer limbs
        for i in quotient.iter_u64_digits().len()..*n_mod {
            quotient_words[i] = Word::ZERO;
        }

        // Fill output inverse limbs
        for (i, limb) in inverse.iter_u64_digits().enumerate() {
            inverse_words[i] = Word::from_u64(limb);
        }
        // Zero remaining outputs if inverse has fewer limbs
        for i in inverse.iter_u64_digits().len()..*n_mod {
            inverse_words[i] = Word::ZERO;
        }
    }
}
