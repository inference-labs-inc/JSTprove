// Copyright 2025 Irreducible Inc.
//! BigUint division hint implementation

use binius_core::Word;

use super::Hint;
use crate::util::num_biguint_from_u64_limbs;

pub struct BigUintDivideHint;

impl BigUintDivideHint {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BigUintDivideHint {
    fn default() -> Self {
        Self::new()
    }
}

impl Hint for BigUintDivideHint {
    fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
        let [dividend_limbs, divisor_limbs] = dimensions else {
            panic!("BigUintDivide requires 2 dimensions");
        };
        (
            *dividend_limbs + *divisor_limbs,
            *dividend_limbs + *divisor_limbs,
        )
    }

    fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
        let [n_dividend, n_divisor] = dimensions else {
            panic!("BigUintDivide requires 2 dimensions");
        };
        let n_quotient = *n_dividend;
        let n_remainder = *n_divisor;

        let dividend_limbs = &inputs[0..*n_dividend];
        let divisor_limbs = &inputs[*n_dividend..];

        let dividend = num_biguint_from_u64_limbs(dividend_limbs.iter().map(|w| w.as_u64()));
        let divisor = num_biguint_from_u64_limbs(divisor_limbs.iter().map(|w| w.as_u64()));

        let zero = num_bigint::BigUint::ZERO;
        let (quotient, remainder) = if divisor != zero {
            (dividend.clone() / divisor.clone(), dividend % divisor)
        } else {
            (zero.clone(), zero.clone())
        };

        // Fill quotient limbs (first part of output)
        for (i, limb) in quotient.iter_u64_digits().enumerate() {
            if i < n_quotient {
                outputs[i] = Word::from_u64(limb);
            }
        }
        // Zero remaining quotient outputs
        for i in quotient.iter_u64_digits().len()..n_quotient {
            outputs[i] = Word::ZERO;
        }

        // Fill remainder limbs (second part of output)
        for (i, limb) in remainder.iter_u64_digits().enumerate() {
            if i < n_remainder {
                outputs[n_quotient + i] = Word::from_u64(limb);
            }
        }
        // Zero remaining remainder outputs
        for i in remainder.iter_u64_digits().len()..n_remainder {
            outputs[n_quotient + i] = Word::ZERO;
        }
    }
}
