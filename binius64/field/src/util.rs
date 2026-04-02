// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use crate::field::FieldOps;

/// Iterate the powers of a given value, beginning with 1 (the 0'th power).
pub fn powers<F: FieldOps>(val: F) -> impl Iterator<Item = F> {
    iter::successors(Some(F::one()), move |power| {
        Some(power.clone() * val.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryField128bGhash, Field};

    #[test]
    fn test_powers_against_pow() {
        let generator = BinaryField128bGhash::MULTIPLICATIVE_GENERATOR;
        let power_values: Vec<_> = powers(generator).take(10).collect();

        for i in 0..10 {
            assert_eq!(power_values[i], generator.pow([i as u64]));
        }
    }
}
