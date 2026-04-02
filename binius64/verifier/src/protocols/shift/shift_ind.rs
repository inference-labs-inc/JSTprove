// Copyright 2025 Irreducible Inc.

//! Shift indicator partial evaluation functions.
//!
//! This module provides functions for computing partial evaluations of shift indicator
//! multilinear extensions and their helper polynomials.

use binius_field::Field;
use binius_math::FieldBuffer;

/// Partial evaluation of the shift indicator helper polynomials $\sigma, \sigma'$ over all i on the
/// hypercube.
///
/// Given fixed j and s, computes sigma and sigma_prime for all possible i values.
/// Returns (sigma, sigma_prime) as FieldBuffers.
pub fn partial_eval_sigmas<F: Field>(r_j: &[F], r_s: &[F]) -> (FieldBuffer<F>, FieldBuffer<F>) {
    assert_eq!(
        r_j.len(),
        r_s.len(),
        "r_j and r_s must have the same length"
    );

    let n = r_j.len();
    let mut sigma = FieldBuffer::zeros(n);
    let mut sigma_prime = FieldBuffer::<F>::zeros(n);
    sigma[0] = F::ONE;

    // Process each bit position
    for k in 0..n {
        let j_k = r_j[k];
        let s_k = r_s[k];

        // Precompute boolean combinations for this bit
        let both = j_k * s_k;
        let j_one_s = j_k - both; // j_k * (1 - s_k)
        let one_j_s = s_k - both; // (1 - j_k) * s_k
        let xor = j_k + s_k;
        let eq = F::ONE + xor;

        // Update arrays for this bit position
        for i in 0..(1 << k) {
            // Update upper halves first (i_k = 1)
            sigma[(1 << k) | i] = j_one_s * sigma[i];
            sigma_prime[(1 << k) | i] = one_j_s * sigma[i] + eq * sigma_prime[i];

            // Update lower halves (i_k = 0)
            let sigma_i = sigma[i];
            let sigma_prime_i = sigma_prime[i];
            sigma[i] = eq * sigma_i + j_one_s * sigma_prime_i;
            sigma_prime[i] *= one_j_s;
        }
    }

    (sigma, sigma_prime)
}

/// Partial evaluation of the shift indicator helper polynomial $\phi$ over all i on the hypercube.
///
/// Given fixed s, computes phi for all possible i values.
pub fn partial_eval_phi<F: Field>(r_s: &[F]) -> FieldBuffer<F> {
    let n = r_s.len();
    let mut phi = FieldBuffer::<F>::zeros(n);

    // Process each bit position
    for k in 0..n {
        let s_k = r_s[k];

        // Update arrays for this bit position
        for i in 0..(1 << k) {
            // Update for i_k = 1
            phi[(1 << k) | i] = s_k + (F::ONE + s_k) * phi[i];
            let temp = phi[(1 << k) | i] - s_k;
            phi[i] += temp;
        }
    }

    phi
}

/// Partial evaluation of transposed sigma for SLL.
///
/// Since sll_ind(i, j, s) = srl_ind(j, i, s), this computes sigma with i and j swapped.
pub fn partial_eval_sigmas_transpose<F: Field>(r_j: &[F], r_s: &[F]) -> FieldBuffer<F> {
    assert_eq!(
        r_j.len(),
        r_s.len(),
        "r_j and r_s must have the same length"
    );

    let n = r_j.len();
    let mut sigma_transpose = FieldBuffer::zeros(n);
    let mut sigma_transpose_prime = FieldBuffer::<F>::zeros(n);
    sigma_transpose[0] = F::ONE;

    // Process each bit position
    for k in 0..n {
        let j_k = r_j[k];
        let s_k = r_s[k];

        // Precompute boolean combinations for this bit (with i and j swapped)
        let both = j_k * s_k;
        let xor = j_k + s_k;
        let eq = F::ONE + xor;
        let zero = eq + both;

        // Update arrays for this bit position
        for i in 0..(1 << k) {
            // Update for i_k = 1
            sigma_transpose[(1 << k) | i] =
                xor * sigma_transpose[i] + zero * sigma_transpose_prime[i];
            sigma_transpose_prime[(1 << k) | i] = both * sigma_transpose_prime[i];

            // Update for i_k = 0
            let sigma_t = sigma_transpose[i];
            sigma_transpose_prime[i] = both * sigma_t + xor * sigma_transpose_prime[i];
            sigma_transpose[i] = zero * sigma_t;
        }
    }

    sigma_transpose
}

#[cfg(test)]
mod tests {
    use binius_field::BinaryField128bGhash;
    use binius_math::{
        multilinear::shift::{rotr_ind, sll_ind, sra_ind, srl_ind},
        test_utils::{index_to_hypercube_point, random_scalars},
    };
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    /// Computes the partial evaluation of the logical left shift indicator multilinear extension.
    ///
    /// Given fixed values for `j` and `s`, this function returns a [`FieldBuffer`] containing the
    /// multilinear extension evaluations for all `i` values on the hypercube of dimension `k`.
    ///
    /// # Arguments
    /// * `j` - Slice of field elements representing the input bit position (length `k`)
    /// * `s` - Slice of field elements representing the shift amount (length `k`)
    ///
    /// # Returns
    /// A [`FieldBuffer`] where the element at index `i_idx` contains
    /// `sll_ind(index_to_hypercube_point(k, i_idx), j, s)`
    ///
    /// # Panics
    /// Panics if `j` and `s` don't have the same length.
    fn sll_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> FieldBuffer<F> {
        // sll_ind(i, j, s) = srl_ind(j, i, s) by transposition
        // So we can use the transpose helper directly
        partial_eval_sigmas_transpose(j, s)
    }

    /// Computes the partial evaluation of the logical right shift indicator multilinear extension.
    ///
    /// Given fixed values for `j` and `s`, this function returns a [`FieldBuffer`] containing the
    /// multilinear extension evaluations for all `i` values on the hypercube of dimension `k`.
    ///
    /// # Arguments
    /// * `j` - Slice of field elements representing the input bit position (length `k`)
    /// * `s` - Slice of field elements representing the shift amount (length `k`)
    ///
    /// # Returns
    /// A [`FieldBuffer`] where the element at index `i_idx` contains
    /// `srl_ind(index_to_hypercube_point(k, i_idx), j, s)`
    ///
    /// # Panics
    /// Panics if `j` and `s` don't have the same length.
    fn srl_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> FieldBuffer<F> {
        let (sigma, _) = partial_eval_sigmas(j, s);
        sigma
    }

    /// Computes the partial evaluation of the arithmetic right shift indicator multilinear
    /// extension.
    ///
    /// Given fixed values for `j` and `s`, this function returns a [`FieldBuffer`] containing the
    /// multilinear extension evaluations for all `i` values on the hypercube of dimension `k`.
    ///
    /// # Arguments
    /// * `j` - Slice of field elements representing the input bit position (length `k`)
    /// * `s` - Slice of field elements representing the shift amount (length `k`)
    ///
    /// # Returns
    /// A [`FieldBuffer`] where the element at index `i_idx` contains
    /// `sra_ind(index_to_hypercube_point(k, i_idx), j, s)`
    ///
    /// # Panics
    /// Panics if `j` and `s` don't have the same length.
    fn sra_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> FieldBuffer<F> {
        assert_eq!(j.len(), s.len(), "j and s must have the same length");

        let (sigma, _) = partial_eval_sigmas(j, s);
        let phi = partial_eval_phi(s);
        let j_prod = j.iter().product::<F>();

        let n = j.len();
        let mut result = FieldBuffer::zeros(n);
        for i in 0..(1 << n) {
            result[i] = sigma[i] + j_prod * phi[i];
        }

        result
    }

    /// Computes the partial evaluation of the rotate right indicator multilinear extension.
    ///
    /// Given fixed values for `j` and `s`, this function returns a [`FieldBuffer`] containing the
    /// multilinear extension evaluations for all `i` values on the hypercube of dimension `k`.
    ///
    /// # Arguments
    /// * `j` - Slice of field elements representing the input bit position (length `k`)
    /// * `s` - Slice of field elements representing the shift amount (length `k`)
    ///
    /// # Returns
    /// A [`FieldBuffer`] where the element at index `i_idx` contains
    /// `rotr_ind(index_to_hypercube_point(k, i_idx), j, s)`
    ///
    /// # Panics
    /// Panics if `j` and `s` don't have the same length.
    fn rotr_ind_partial_eval<F: Field>(j: &[F], s: &[F]) -> FieldBuffer<F> {
        let (sigma, sigma_prime) = partial_eval_sigmas(j, s);

        let n = j.len();
        let mut result = FieldBuffer::zeros(n);
        for i in 0..(1 << n) {
            result[i] = sigma[i] + sigma_prime[i];
        }

        result
    }

    // Type alias for shift indicator functions
    type ShiftIndicatorFn<F> = fn(&[F], &[F], &[F]) -> F;

    // Helper function to test partial evaluation against direct function calls
    fn test_partial_eval_helper<F: Field>(
        partial_eval_fn: fn(&[F], &[F]) -> FieldBuffer<F>,
        direct_fn: ShiftIndicatorFn<F>,
        j: &[F],
        s: &[F],
    ) {
        let n = j.len();
        let partial_eval = partial_eval_fn(j, s);

        // Check that the buffer has the right size
        assert_eq!(partial_eval.len(), 1 << n);

        // Check each value against the direct function
        for i_idx in 0..(1 << n) {
            let i = index_to_hypercube_point::<F>(n, i_idx);
            let expected = direct_fn(&i, j, s);
            let actual = partial_eval.get(i_idx);
            assert_eq!(
                actual, expected,
                "Mismatch at i_idx={}, i={:?}, j={:?}, s={:?}",
                i_idx, i, j, s
            );
        }
    }

    #[test]
    fn test_sll_ind_partial_eval() {
        let mut rng = StdRng::seed_from_u64(0);
        let n = 6;

        // Test with random j and s
        let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
        let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

        test_partial_eval_helper(sll_ind_partial_eval, sll_ind, &j, &s);
    }

    #[test]
    fn test_srl_ind_partial_eval() {
        let mut rng = StdRng::seed_from_u64(0);
        let n = 6;

        // Test with random j and s
        let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
        let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

        test_partial_eval_helper(srl_ind_partial_eval, srl_ind, &j, &s);
    }

    #[test]
    fn test_sra_ind_partial_eval() {
        let mut rng = StdRng::seed_from_u64(0);
        let n = 6;

        // Test with random j and s
        let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
        let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

        test_partial_eval_helper(sra_ind_partial_eval, sra_ind, &j, &s);
    }

    #[test]
    fn test_rotr_ind_partial_eval() {
        let mut rng = StdRng::seed_from_u64(0);
        let n = 6;

        // Test with random j and s
        let j = random_scalars::<BinaryField128bGhash>(&mut rng, n);
        let s = random_scalars::<BinaryField128bGhash>(&mut rng, n);

        test_partial_eval_helper(rotr_ind_partial_eval, rotr_ind, &j, &s);
    }
}
