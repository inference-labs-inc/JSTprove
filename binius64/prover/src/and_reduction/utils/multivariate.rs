// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_field::Field;
use binius_math::FieldBuffer;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::LOG_WORD_SIZE_BITS;

use crate::and_reduction::fold_lookup::FoldLookup;

// ALL FOLDING IS LOW TO HIGH
/// Represents a OblongMultilinear polynomial with binary (0/1) coefficients.
///
/// This struct stores the evaluations of a OblongMultilinear polynomial over the binary hypercube
/// in a packed format for efficient processing. The polynomial can be partially evaluated
/// (folded) on its first variable using either a naive approach or an optimized lookup table.
#[derive(Debug, Clone)]
pub struct OneBitOblongMultilinear {
    /// Logarithm base 2 of the number of rows (evaluations) in the polynomial.
    /// The total number of evaluations is 2^log_num_rows.
    pub log_num_rows: usize,
    /// Packed binary field elements storing the polynomial evaluations.
    /// Each element contains 64 binary values packed together to create 64-bit words.
    pub packed_evals: Vec<Word>,
}

impl OneBitOblongMultilinear {
    /// Performs partial evaluation of the OblongMultilinear polynomial on its first variable using
    /// a lookup table.
    ///
    /// This method is the optimized version of `fold_naive`, using a precomputed lookup table
    /// for efficient evaluation. It evaluates the polynomial at the challenge point that was
    /// used to construct the lookup table.
    ///
    /// # Type Parameters
    /// * `F` - The field type for the evaluation result
    ///
    /// # Arguments
    /// * `lookup` - The precomputed lookup table for the evaluation point
    ///
    /// # Returns
    /// A `FieldBuffer` containing the evaluations of the partially evaluated polynomial
    /// over the remaining variables.
    #[allow(clippy::modulo_one)]
    pub fn fold<F: Field>(&self, lookup: &FoldLookup<F, LOG_WORD_SIZE_BITS>) -> FieldBuffer<F> {
        let new_n_vars = self.log_num_rows - LOG_WORD_SIZE_BITS;
        let multilin_vals = self
            .packed_evals
            .par_iter()
            .map(|word| {
                let word_bytes = word.as_u64().to_le_bytes();
                lookup.fold_one_bit_univariate(word_bytes.into_iter())
            })
            .collect();
        FieldBuffer::new(new_n_vars, multilin_vals)
    }
}

#[cfg(test)]
mod test {
    use std::{iter, iter::repeat_with};

    use binius_core::word::Word;
    use binius_field::{BinaryField, FieldOps, Random};
    use binius_math::{BinarySubspace, FieldBuffer, univariate::lagrange_evals_scalars};
    use binius_verifier::{
        and_reduction::utils::constants::SKIPPED_VARS,
        config::{B128, LOG_WORD_SIZE_BITS, WORD_SIZE_BITS},
    };
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::OneBitOblongMultilinear;
    use crate::and_reduction::fold_lookup::FoldLookup;

    // Performs partial evaluation of the OblongMultilinear polynomial on its first variable using
    /// a naive approach.
    ///
    /// This method evaluates the polynomial at a given challenge point for the first variable,
    /// effectively reducing the number of variables by `LOG_FIRST_VARIABLE_DEGREE_BOUND`.
    /// It uses direct Lagrange interpolation without precomputation.
    ///
    /// # Type Parameters
    /// * `FDomain` - The field type for the univariate domain
    /// * `F` - The field type for the evaluation result
    /// * `LOG_FIRST_VARIABLE_DEGREE_BOUND` - The logarithm base 2 of the degree bound for the first
    ///   variable
    ///
    /// # Arguments
    /// * `univariate_domain` - The domain over which the first variable is defined
    /// * `challenge` - The point at which to evaluate the first variable
    ///
    /// # Returns
    /// A `FieldBuffer` containing the evaluations of the partially evaluated polynomial
    /// over the remaining variables.
    #[allow(clippy::modulo_one)]
    pub fn fold_naive<F: BinaryField>(
        one_bit_oblong: &OneBitOblongMultilinear,
        univariate_domain: &BinarySubspace<F>,
        challenge: F,
    ) -> FieldBuffer<F> {
        let new_n_vars = one_bit_oblong.log_num_rows - LOG_WORD_SIZE_BITS;

        let lagrange_basis_vectors = lagrange_evals_scalars(univariate_domain, challenge);

        let result_vals = one_bit_oblong
            .packed_evals
            .iter()
            .map(|word| {
                let word_bits = (0..WORD_SIZE_BITS).map(|i| (word.as_u64() >> i) & 1 == 1);
                iter::zip(&lagrange_basis_vectors, word_bits)
                    .map(|(&elem, bit)| if bit { elem } else { F::zero() })
                    .sum()
            })
            .collect();
        FieldBuffer::new(new_n_vars, result_vals)
    }

    fn random_one_bit_multivariate(
        log_num_rows: usize,
        mut rng: impl Rng,
    ) -> OneBitOblongMultilinear {
        OneBitOblongMultilinear {
            log_num_rows,
            packed_evals: repeat_with(|| Word(rng.random()))
                .take(1 << (log_num_rows - LOG_WORD_SIZE_BITS))
                .collect(),
        }
    }

    #[test]
    fn test_lookup_fold() {
        let log_num_rows = 10;
        let mut rng = StdRng::from_seed([0; 32]);
        let mlv = random_one_bit_multivariate(log_num_rows, &mut rng);

        let challenge = B128::random(&mut rng);

        let univariate_domain = BinarySubspace::with_dim(SKIPPED_VARS);

        let lookup = FoldLookup::<_, SKIPPED_VARS>::new(&univariate_domain, challenge);

        let folded_naive = fold_naive(&mlv, &univariate_domain, challenge);

        let folded_smart = mlv.fold(&lookup);

        for i in 0..1 << (log_num_rows - SKIPPED_VARS) {
            assert_eq!(folded_naive.as_ref()[i], folded_smart.as_ref()[i]);
        }
    }
}
