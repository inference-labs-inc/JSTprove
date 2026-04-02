// Copyright 2023-2025 Irreducible Inc.

use std::cmp::min;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{
    field_buffer::FieldBuffer,
    multilinear::eq::{eq_ind_partial_eval, eq_ind_truncate_low_inplace, eq_one_var},
};

use super::round_evals::{RoundEvals2, round_coeffs_by_eq};

// A helper struct that implements an Mlecheck degree lowering logic using tricks from [Gruen24]
// section 3.2 (hence the name). See a docstring to `bivariate_product_mle::new` for more in-depth
// explanation.
//
// It's initialized with a point from an evaluation claim, and takes care of folding the eq
// indicator expansion, updating prefix product and multiplying by the linear part (denoted (3), (1)
// and (2) in the mentioned docstring).
//
// The eq indicator may be treated as an outer product of "chunk" and "suffix" sub-eq-indicators.
// "Chunk" is instantiated over lower indexed variables and "suffix" over the remaining variables.
//
// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[derive(Debug, Clone)]
pub struct Gruen32<P: PackedField> {
    n_vars_remaining: usize,
    chunk_eq_expansion: FieldBuffer<P>,
    suffix_eq_expansion: FieldBuffer<P>,
    eval_point: Vec<P::Scalar>,
    eq_prefix_eval: P::Scalar,
}

impl<F: Field, P: PackedField<Scalar = F>> Gruen32<P> {
    pub fn new(eval_point: &[F]) -> Self {
        Self::new_with_suffix(eval_point.len(), eval_point)
    }

    pub fn new_with_suffix(max_chunk_vars: usize, eval_point: &[F]) -> Self {
        let n_vars_remaining = eval_point.len();

        let truncated_eval_point = &eval_point[..n_vars_remaining.saturating_sub(1)];

        let chunk_vars = min(max_chunk_vars, truncated_eval_point.len());
        let (chunk_eval_point, suffix_eval_point) = truncated_eval_point.split_at(chunk_vars);

        let chunk_eq_expansion = eq_ind_partial_eval(chunk_eval_point);
        let suffix_eq_expansion = eq_ind_partial_eval(suffix_eval_point);

        Self {
            n_vars_remaining,
            chunk_eq_expansion,
            suffix_eq_expansion,
            eval_point: eval_point.to_vec(),
            eq_prefix_eval: F::ONE,
        }
    }

    pub fn eval_point(&self) -> &[F] {
        &self.eval_point
    }

    /// Returns the coordinate value of the evaluation point for the next variable to be bound.
    pub fn next_coordinate(&self) -> F {
        self.eval_point[self.n_vars_remaining - 1]
    }

    pub fn eq_expansion(&self) -> &FieldBuffer<P> {
        assert_eq!(self.suffix_eq_expansion.log_len(), 0);
        &self.chunk_eq_expansion
    }

    pub fn chunk_eq_expansion(&self) -> &FieldBuffer<P> {
        &self.chunk_eq_expansion
    }

    pub fn suffix_eq_expansion(&self) -> &FieldBuffer<P> {
        &self.suffix_eq_expansion
    }

    pub fn n_vars_remaining(&self) -> usize {
        self.n_vars_remaining
    }

    // An interpolation routine for degree-2 round polynomials. Takes P'(x) evals and sum claim on
    // P(x).
    #[allow(dead_code)]
    pub fn interpolate2(
        &self,
        sum: F,
        prime_evals: RoundEvals2<F>,
    ) -> (RoundCoeffs<F>, RoundCoeffs<F>) {
        // Value of evaluation point in currently specialized variable
        let alpha = self.next_coordinate();
        // Degree-2 interpolation
        let prime_coeffs = prime_evals.interpolate_eq(sum, alpha);
        // Multiply by linear eq indicator part (2) and prefix product (1)
        let round_coeffs = round_coeffs_by_eq(&prime_coeffs, alpha) * self.eq_prefix_eval;
        (prime_coeffs, round_coeffs)
    }

    pub fn fold(&mut self, challenge: F) {
        assert!(self.n_vars_remaining > 0);

        // Eq indicator folding is just an xor. Remember that we are one variable less than other
        // multilinears.
        debug_assert_eq!(
            self.chunk_eq_expansion.log_len() + self.suffix_eq_expansion.log_len(),
            self.n_vars_remaining - 1
        );
        // High-to-low evaluation order means we need to fold suffix first.
        if self.suffix_eq_expansion.log_len() > 0 {
            let new_log_len = self.suffix_eq_expansion.log_len() - 1;
            eq_ind_truncate_low_inplace(&mut self.suffix_eq_expansion, new_log_len);
        } else if self.chunk_eq_expansion.log_len() > 0 {
            let new_log_len = self.chunk_eq_expansion.log_len() - 1;
            eq_ind_truncate_low_inplace(&mut self.chunk_eq_expansion, new_log_len);
        }

        // Update the prefix product (1)
        let alpha = self.next_coordinate();
        self.eq_prefix_eval *= eq_one_var(challenge, alpha);

        self.n_vars_remaining -= 1;
    }
}
