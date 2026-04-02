// Copyright 2025 Irreducible Inc.
//! BigUint mod pow hint implementation

use binius_core::Word;

use super::Hint;
use crate::util::num_biguint_from_u64_limbs;

pub struct BigUintModPowHint;

impl BigUintModPowHint {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BigUintModPowHint {
    fn default() -> Self {
        Self::new()
    }
}

impl Hint for BigUintModPowHint {
    fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
        let [n_base_limbs, n_exp_limbs, n_modulus_limbs] = dimensions else {
            panic!("BigUintModPowHint requires 3 dimensions");
        };
        (
            *n_base_limbs + *n_exp_limbs + *n_modulus_limbs,
            *n_modulus_limbs,
        )
    }

    fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
        let [n_base_limbs, n_exp_limbs, n_modulus_limbs] = dimensions else {
            panic!("BigUintModPowHint requires 3 dimensions");
        };

        assert_eq!(
            inputs.len(),
            *n_base_limbs + *n_exp_limbs + *n_modulus_limbs
        );
        assert_eq!(outputs.len(), *n_modulus_limbs);

        let (base_limbs, inputs) = inputs.split_at(*n_base_limbs);
        let (exp_limbs, modulus_limbs) = inputs.split_at(*n_exp_limbs);

        let base = num_biguint_from_u64_limbs(base_limbs.iter().map(|w| w.as_u64()));
        let exp = num_biguint_from_u64_limbs(exp_limbs.iter().map(|w| w.as_u64()));
        let modulus = num_biguint_from_u64_limbs(modulus_limbs.iter().map(|w| w.as_u64()));

        let modpow = base.modpow(&exp, &modulus);

        // Fill modpow limbs (first part of the output)
        for (i, limb) in modpow.iter_u64_digits().enumerate() {
            outputs[i] = Word::from_u64(limb);
        }

        for i in modpow.iter_u64_digits().len()..*n_modulus_limbs {
            outputs[i] = Word::ZERO;
        }
    }
}
