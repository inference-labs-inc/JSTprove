// Copyright 2024-2025 Irreducible Inc.

//! Different implementations for the [`DomainContext`] trait.

use binius_field::BinaryField;

use super::DomainContext;
use crate::{BinarySubspace, binary_subspace::BinarySubspaceIterator};

/// Given an $\ell$-dimensional subspace, this computes the evaluations of the normalized subspace
/// polynomials $hat{W}_i$ on the basis.
///
/// The dimension of the subspace is $\ell$. This returns a vector of $\ell$ vectors. The $i$th
/// vector contains $\ell - i$ entries and contains
///
/// $$
/// \left( \hat{W}_i(\beta_j) \right)_{j = i}^{\ell - 1}
/// $$
fn generate_evals_from_subspace<F: BinaryField>(subspace: &BinarySubspace<F>) -> Vec<Vec<F>> {
    let l = subspace.dim();
    let mut evals = Vec::with_capacity(l);

    // push $[W_0 (\beta_0), W_0 (\beta_1), ...] = [\beta_0, \beta_1, ...]$
    evals.push(subspace.basis().to_vec());
    for i in 1..l {
        // push $[W_i (\beta_i), W_i (\beta_(i+1)), ...]$
        evals.push(Vec::with_capacity(l - i));
        for k in 1..evals[i - 1].len() {
            // $W_i (X) = W_(i-1) (X) * [ W_(i-1) (X) + W_(i-1) (\beta_(i-1)) ]$
            // hence: $W_i (\beta_j) = W_(i-1) (\beta_(j)) * [ W_(i-1) (\beta_j) + W_(i-1)
            // (\beta_(i-1)) ]$
            let val = evals[i - 1][k] * (evals[i - 1][k] + evals[i - 1][0]);
            evals[i].push(val);
        }
    }

    // normalize so that evaluations of $W_i$ are replaced by evaluations of $hat{W}_i$
    for evals_i in evals.iter_mut() {
        let w_i_b_i_inverse = evals_i[0].invert().unwrap();
        for eval_i_j in evals_i.iter_mut() {
            *eval_i_j *= w_i_b_i_inverse;
        }
    }

    evals
}

/// Works for any $S^{(0)}$ and computes twiddles on-the-fly.
///
/// On-the-fly twiddle computation should not be used for running a big NTT or for a complete FRI
/// folding. In both cases, all twiddles will be accessed eventually, so they might as well be
/// precomputed. But it could possibly be used e.g. in the FRI verifier, which only accesses a few
/// selected twiddles.
#[derive(Debug)]
pub struct GenericOnTheFly<F> {
    /// The subspace polynomial evaluations.
    ///
    /// Contains $\ell$ vectors. The $i$th vector stores
    ///
    /// $$
    /// \left( \hat{W}_i(\beta_j) \right)_{j = i}^{\ell - 1}.
    /// $$
    evals: Vec<Vec<F>>,
}

impl<F: BinaryField> GenericOnTheFly<F> {
    /// Given a basis of $S^{(0)}$, computes a compatible [`DomainContext`].
    ///
    /// This will _not_ precompute the twiddles; instead they will be computed on-the-fly.
    pub fn generate_from_subspace(subspace: &BinarySubspace<F>) -> Self {
        Self {
            evals: generate_evals_from_subspace(subspace),
        }
    }
}

impl<F: BinaryField> DomainContext for GenericOnTheFly<F> {
    type Field = F;

    fn log_domain_size(&self) -> usize {
        self.evals.len()
    }

    fn subspace(&self, i: usize) -> BinarySubspace<F> {
        if i == 0 {
            return BinarySubspace::with_dim(0);
        }
        BinarySubspace::new_unchecked(self.evals[self.log_domain_size() - i].clone())
    }

    fn twiddle(&self, layer: usize, block: usize) -> F {
        let v = &self.evals[self.log_domain_size() - layer - 1];
        BinarySubspace::new_unchecked(&v[1..]).get(block)
    }
}

/// Works for any $S^{(0)}$ and pre-computes twiddles.
#[derive(Debug)]
pub struct GenericPreExpanded<F> {
    /// The $i$'th vector stores $[hat{W}_i (\beta_i), \hat{W}_i (\beta_(i+1)), ...]$.
    evals: Vec<Vec<F>>,
    /// The $n - i - 1$'th vector stores $[0, \hat{W}_i (\beta_(i+1)), \hat{W}_i (\beta_(i+2)),
    /// \hat{W}_i (\beta_(i+2) + \beta_(i+1)), \hat{W}_i (\beta_(i+3)), ...]$.
    ///
    /// Notice the absence of $\beta_i$ in this. (Which satisfies $\hat{W}_i (\beta_i) = 1$
    /// and is absorbed in the butterfly operation itself rather than what we call "twiddles".)
    expanded: Vec<Vec<F>>,
}

impl<F: BinaryField> GenericPreExpanded<F> {
    /// Given a basis of $S^{(0)}$, computes a compatible [`DomainContext`].
    ///
    /// This will _precompute_ the twiddles.
    pub fn generate_from_subspace(subspace: &BinarySubspace<F>) -> Self {
        let evals = generate_evals_from_subspace(subspace);

        let mut expanded = Vec::with_capacity(evals.len());
        for basis in evals.iter().rev() {
            let mut expanded_i = Vec::with_capacity(1 << (basis.len() - 1));
            expanded_i.push(F::ZERO);
            for i in 1..basis.len() {
                for j in 0..expanded_i.len() {
                    expanded_i.push(expanded_i[j] + basis[i]);
                }
            }
            assert_eq!(expanded_i.len(), 1 << (basis.len() - 1));
            expanded.push(expanded_i)
        }
        assert_eq!(expanded.len(), evals.len());

        Self { evals, expanded }
    }
}

impl<F: BinaryField> DomainContext for GenericPreExpanded<F> {
    type Field = F;

    fn log_domain_size(&self) -> usize {
        self.evals.len()
    }

    fn subspace(&self, i: usize) -> BinarySubspace<F> {
        if i == 0 {
            return BinarySubspace::with_dim(0);
        }
        BinarySubspace::new_unchecked(self.evals[self.log_domain_size() - i].clone())
    }

    fn twiddle(&self, layer: usize, block: usize) -> F {
        self.expanded[layer][block]
    }

    fn iter_twiddles(&self, layer: usize, log_step_by: usize) -> impl Iterator<Item = F> {
        self.expanded[layer]
            .iter()
            .step_by(1 << log_step_by)
            .copied()
    }
}

/// Provides a field element of trace 1.
pub trait TraceOneElement {
    /// Returns a field element which has trace 1.
    fn trace_one_element() -> Self;
}

impl TraceOneElement for binius_field::BinaryField128bGhash {
    fn trace_one_element() -> Self {
        Self::new(1 << 121)
    }
}

/// Computes the first `num_basis_elements` Gao-Mateer basis elements of the field.
///
/// ## Preconditions
///
/// - The degree (over $\mathbb{F}_2$) of the field needs to be a tower of two. For example, it does
///   **not** work with $\mathbb{F}_{2^3}$, but it works with $\mathbb{F}_{2^4}$.
/// - `num_basis_elements` must be nonzero
fn gao_mateer_basis<F: BinaryField + TraceOneElement>(num_basis_elements: usize) -> Vec<F> {
    assert!(F::N_BITS.is_power_of_two());

    // this is beta_(F::N_BITS - 1)
    // e.g. for a 128-bit field, this is beta_127
    let mut beta = F::trace_one_element();

    // we compute beta_126, then beta_125, etc, by repeatedly applying x |-> x^2 + x
    for _i in 0..(F::N_BITS - num_basis_elements) {
        beta = beta.square() + beta;
    }

    // we save beta_0, beta_1, ..., beta_(num_basis_elements - 1)
    let mut basis = vec![F::ZERO; num_basis_elements];
    basis[num_basis_elements - 1] = beta;
    for i in (1..num_basis_elements).rev() {
        basis[i - 1] = basis[i].square() + basis[i];
    }

    // check that beta_0 = 1, which must be necessarily true if the trace 1 element above indeed
    // has trace 1
    assert_eq!(basis[0], F::ONE);

    basis
}

/// Produces a specific "Gao-Mateer" $S^{(0)}$ and computes twiddles on-the-fly. Only works for
/// binary fields whose degree over $\mathbb{F}_2$ is a power of two.
///
/// A Gao-Mateer basis of the binary field $\mathbb{F}_{2^{2^k}}$ is any $\mathbb{F}_2$-basis
/// $(\beta_0, \beta_1,..., \beta_{2^k - 1})$ with the following properties:
/// - $\beta_{2^k-1}$ has trace 1
/// - $\beta_i = \beta_{i+1}^2 + \beta_{i+1}$
///
/// This implies some nice properties:
/// - the basis elements with a small index are in a small subfield, in fact
///   $(\beta_0,...,\beta_{2^l - 1})$ is a basis of $\mathbb{F}_{2^{2^l}}$ for any $l$, and in
///   particular $\beta_0 = 1$
/// - The subspace polynomial $W_i$ of $\operatorname{span} {\beta_0, ..., \beta_{i-1}}$ is defined
///   over $\mathbb{F}_2$, i.e., its coefficients are just $0$ or $1$.
/// - The subspace polynomial is "auto-normalized", meaning $W_i (\beta_i) = 1$ so that $\hat{W}_i =
///   W_i$.
/// - The evaluations of the subspace polynomials are just the basis elements: $W_i (\beta_{i + r})
///   = \beta_r$ for any $i$ and $r$.
/// - The previous point together with the first point implies that twiddles in early layers lie in
///   a small subfield. This could potentially be used to speed up the NTT if one can implement
///   multiplication with an element from a small subfield more efficiently.
/// - The folding maps for FRI are all $x \mapsto x^2 + x$, no normalization factors needed.
///
/// On-the-fly twiddle computation should not be used for running a big NTT or for a complete FRI
/// folding. In both cases, all twiddles will be accessed eventually, so they might as well be
/// precomputed. But it could possibly be used e.g. in the FRI verifier, which only accesses a few
/// selected twiddles.
#[derive(Debug)]
pub struct GaoMateerOnTheFly<F> {
    /// Stores $[\beta_0, ..., \beta_{\ell-1}]$.
    basis: Vec<F>,
}

impl<F: BinaryField + TraceOneElement> GaoMateerOnTheFly<F> {
    /// Given the intended size of $S^{(0)}$, computes a "nice" Gao-Mateer [`DomainContext`].
    ///
    /// This will _not_ precompute the twiddles; instead they will be computed on-the-fly.
    ///
    /// ## Preconditions
    ///
    /// - The degree (over $\mathbb{F}_2$) of the field needs to be a tower of two. For example, it
    ///   does **not** work with $\mathbb{F}_{2^3}$, but it works with $\mathbb{F}_{2^4}$.
    /// - `log_domain_size` must be nonzero
    pub fn generate(log_domain_size: usize) -> Self {
        Self {
            basis: gao_mateer_basis(log_domain_size),
        }
    }
}

impl<F: BinaryField> DomainContext for GaoMateerOnTheFly<F> {
    type Field = F;

    fn log_domain_size(&self) -> usize {
        self.basis.len()
    }

    fn subspace(&self, i: usize) -> BinarySubspace<F> {
        BinarySubspace::new_unchecked(self.basis[..i].to_vec())
    }

    fn twiddle(&self, layer: usize, block: usize) -> F {
        BinarySubspace::new_unchecked(&self.basis[1..=layer]).get(block)
    }

    fn iter_twiddles(&self, layer: usize, log_step_by: usize) -> impl Iterator<Item = F> + '_ {
        BinarySubspaceIterator::new(&self.basis[1 + log_step_by..=layer])
    }
}

/// Produces a specific "Gao-Mateer" $S^{(0)}$ and pre-computes twiddles. Only works for binary
/// fields whose degree over $\mathbb{F}_2$ is a power of two.
///
/// For an explanation of this $S^{(0)}$, see [`GaoMateerOnTheFly`].
#[derive(Debug)]
pub struct GaoMateerPreExpanded<F> {
    /// Stores $[\beta_0, \beta_1, ...]$.
    basis: Vec<F>,
    /// Stores $[0, \beta_1, \beta_2, \beta_2 + \beta_1, \beta_3, \beta_3 + \beta_1, ...]$.
    ///
    /// Notice the absence of $\beta_0$. (Which is $\beta_0 = 1$ and is absorbed in the butterfly
    /// operation itself rather than what we call "twiddles".)
    expanded: Vec<F>,
}

impl<F: BinaryField + TraceOneElement> GaoMateerPreExpanded<F> {
    /// Given the intended size of $S^{(0)}$, computes a "nice" Gao-Mateer [`DomainContext`].
    ///
    /// This will _precompute_ the twiddles.
    ///
    /// ## Preconditions
    ///
    /// - The degree (over $\mathbb{F}_2$) of the field needs to be a tower of two. For example, it
    ///   does **not** work with $\mathbb{F}_{2^3}$, but it works with $\mathbb{F}_{2^4}$.
    /// - `log_domain_size` must be nonzero
    pub fn generate(log_domain_size: usize) -> Self {
        let basis: Vec<F> = gao_mateer_basis(log_domain_size);

        let mut expanded = Vec::with_capacity(1 << log_domain_size);
        expanded.push(F::ZERO);
        for i in 1..log_domain_size {
            for j in 0..expanded.len() {
                expanded.push(expanded[j] + basis[i]);
            }
        }
        assert_eq!(expanded.len(), 1usize << (log_domain_size - 1));

        Self { basis, expanded }
    }
}

impl<F: BinaryField> DomainContext for GaoMateerPreExpanded<F> {
    type Field = F;

    fn log_domain_size(&self) -> usize {
        self.basis.len()
    }

    fn subspace(&self, i: usize) -> BinarySubspace<F> {
        BinarySubspace::new_unchecked(self.basis[..i].to_vec())
    }

    fn twiddle(&self, _layer: usize, block: usize) -> F {
        self.expanded[block]
    }

    fn iter_twiddles(&self, layer: usize, log_step_by: usize) -> impl Iterator<Item = F> {
        self.expanded[..1 << layer]
            .iter()
            .step_by(1 << log_step_by)
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::B128;

    fn test_equivalence<F: BinaryField>(
        dc_1: &impl DomainContext<Field = F>,
        dc_2: &impl DomainContext<Field = F>,
        log_domain_size: usize,
    ) {
        assert_eq!(dc_1.log_domain_size(), log_domain_size);
        assert_eq!(dc_2.log_domain_size(), log_domain_size);

        for i in 0..log_domain_size {
            assert_eq!(dc_1.subspace(i), dc_2.subspace(i));

            for block in 0..1 << i {
                assert_eq!(dc_1.twiddle(i, block), dc_2.twiddle(i, block));
            }
        }
        assert_eq!(
            dc_1.subspace(log_domain_size),
            dc_2.subspace(log_domain_size)
        )
    }

    #[test]
    fn test_generic() {
        const LOG_SIZE: usize = 5;

        let subspace = BinarySubspace::with_dim(LOG_SIZE);

        let dc_otf = GenericOnTheFly::<B128>::generate_from_subspace(&subspace);
        let dc_pre = GenericPreExpanded::<B128>::generate_from_subspace(&subspace);

        test_equivalence(&dc_otf, &dc_pre, LOG_SIZE);
    }

    #[test]
    fn test_gao_mateer() {
        const LOG_SIZE: usize = 5;

        let dc_gm_otf = GaoMateerOnTheFly::<B128>::generate(LOG_SIZE);
        let dc_gm_pre = GaoMateerPreExpanded::<B128>::generate(LOG_SIZE);
        let dc_generic_otf =
            GenericOnTheFly::<B128>::generate_from_subspace(&dc_gm_otf.subspace(LOG_SIZE));

        test_equivalence(&dc_gm_otf, &dc_gm_pre, LOG_SIZE);
        test_equivalence(&dc_gm_otf, &dc_generic_otf, LOG_SIZE);
    }

    #[test]
    fn test_iter_layer() {
        const LOG_SIZE: usize = 7;

        let dc_gm_otf = GaoMateerOnTheFly::<B128>::generate(LOG_SIZE);
        let dc_gm_pre = GaoMateerPreExpanded::<B128>::generate(LOG_SIZE);
        let subspace = BinarySubspace::with_dim(LOG_SIZE);
        let dc_generic_otf = GenericOnTheFly::<B128>::generate_from_subspace(&subspace);
        let dc_generic_pre = GenericPreExpanded::<B128>::generate_from_subspace(&subspace);

        // Test that iter_layer returns the same values as calling twiddle individually
        for layer in 0..LOG_SIZE {
            let expected: Vec<_> = (0..1 << layer)
                .map(|block| dc_gm_pre.twiddle(layer, block))
                .collect();

            assert_eq!(
                dc_gm_otf.iter_twiddles(layer, 0).collect::<Vec<_>>(),
                expected,
                "GaoMateerOnTheFly iter_layer mismatch at layer {}",
                layer
            );
            assert_eq!(
                dc_gm_pre.iter_twiddles(layer, 0).collect::<Vec<_>>(),
                expected,
                "GaoMateerPreExpanded iter_layer mismatch at layer {}",
                layer
            );
            assert_eq!(
                dc_generic_otf.iter_twiddles(layer, 0).collect::<Vec<_>>(),
                dc_generic_pre.iter_twiddles(layer, 0).collect::<Vec<_>>(),
                "Generic iter_layer mismatch at layer {}",
                layer
            );
        }
    }
}
