// Copyright 2025 Irreducible Inc.
use binius_field::{BinaryField, Field};
use binius_math::{BinarySubspace, univariate::lagrange_evals_scalars};

/// A lookup table for efficiently evaluating univariate polynomials at a point.
///
/// This struct precomputes Lagrange interpolation coefficients for a given evaluation point,
/// enabling fast evaluation of univariate polynomials represented as packed binary coefficients.
/// The lookup table is organized by byte chunks to enable efficient batch processing.
///
/// Uses a boxed 2D array for better memory layout and cache performance than Vec<Vec<_>>.
///
/// # Type Parameters
/// * `F` - The field type for the evaluation result
/// * `LOG_UNIVARIATE_POLY_COEFFS` - The logarithm base 2 of the number of coefficients in the
///   univariate polynomial
pub struct FoldLookup<F, const LOG_UNIVARIATE_POLY_COEFFS: usize> {
    lookup_table: Box<[[F; 256]]>,
}

impl<F, const LOG_UNIVARIATE_POLY_COEFFS: usize> FoldLookup<F, LOG_UNIVARIATE_POLY_COEFFS>
where
    F: Field,
{
    /// Constructs a new lookup table for evaluating univariate polynomials at a given point.
    ///
    /// This precomputes all possible evaluations for 8-bit chunks of polynomial coefficients
    /// using Lagrange interpolation. The resulting lookup table enables O(n/8) evaluation
    /// of n-coefficient polynomials.
    ///
    /// # Arguments
    /// * `univariate_domain` - The domain over which the univariate polynomial is defined
    /// * `challenge` - The point at which to evaluate the polynomial
    ///
    /// # Returns
    /// A `FoldLookup` instance containing precomputed evaluation results for all possible
    /// 8-bit coefficient patterns.
    pub fn new(univariate_domain: &BinarySubspace<F>, challenge: F) -> Self
    where
        F: Field + BinaryField,
    {
        let _span = tracing::debug_span!("precompute_fold_lookup").entered();

        let univariate_poly_coeffs = 1 << LOG_UNIVARIATE_POLY_COEFFS;
        let outer_size = univariate_poly_coeffs / 8;

        let mut lookup_table = vec![[F::ZERO; 256]; outer_size].into_boxed_slice();

        let lagrange_coeffs = lagrange_evals_scalars(univariate_domain, challenge);

        for (chunk_idx, this_byte_lookup) in lookup_table.iter_mut().enumerate() {
            let _span = tracing::debug_span!("chunk_idx: {}", chunk_idx).entered();

            let offset = 8 * chunk_idx;

            for (lookup_table_idx, this_bit_string_fold_result) in
                this_byte_lookup.iter_mut().enumerate()
            {
                for bit_position in 0..8 {
                    if lookup_table_idx & 1 << bit_position != 0 {
                        *this_bit_string_fold_result += lagrange_coeffs[offset + bit_position];
                    }
                }
            }
        }

        Self { lookup_table }
    }

    /// Evaluates a univariate polynomial at the precomputed point using the lookup table.
    ///
    /// This method takes the polynomial coefficients packed as bytes (8 coefficients per byte)
    /// and uses the precomputed lookup table to efficiently compute the evaluation.
    ///
    /// # Arguments
    /// * `coeffs_byte_chunks` - Iterator over bytes where each byte represents 8 polynomial
    ///   coefficients in binary (LSB first)
    ///
    /// # Returns
    /// The evaluation of the polynomial at the challenge point used during construction.
    #[inline]
    pub fn fold_one_bit_univariate(&self, coeffs_byte_chunks: impl Iterator<Item = u8>) -> F {
        coeffs_byte_chunks
            .enumerate()
            .map(|(byte_chunk_idx, eight_coeffs)| {
                self.lookup_table[byte_chunk_idx][eight_coeffs as usize]
            })
            .sum()
    }
}
