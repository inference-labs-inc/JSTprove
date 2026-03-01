use std::ops::{Add, Index, IndexMut, Mul};

use arith::Field;
use ark_std::log2;

use crate::{EqPolynomial, MultilinearExtension, MutableMultilinearExtension};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiLinearPoly<F: Field> {
    pub coeffs: Vec<F>,
}

impl<F: Field> MultiLinearPoly<F> {
    #[inline]
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());

        Self { coeffs: evals }
    }

    /// Sample a random polynomials.
    #[inline]
    pub fn random(nv: usize, mut rng: impl rand::RngCore) -> Self {
        let coeff = (0..1 << nv).map(|_| F::random_unsafe(&mut rng)).collect();
        Self { coeffs: coeff }
    }

    #[inline]
    pub fn get_num_vars(&self) -> usize {
        log2(self.coeffs.len()) as usize
    }

    // TODO: optimize this function
    #[inline]
    pub fn interpolate_over_hypercube_impl(evals: &[F]) -> Vec<F> {
        let mut coeffs = evals.to_vec();
        let num_vars = log2(evals.len());

        for i in 1..=num_vars {
            let chunk_size = 1 << i;

            coeffs.chunks_mut(chunk_size).for_each(|chunk| {
                let half_chunk = chunk_size >> 1;
                let (left, right) = chunk.split_at_mut(half_chunk);
                right
                    .iter_mut()
                    .zip(left.iter())
                    .for_each(|(a, b)| *a -= *b);
            })
        }

        coeffs
    }

    // interpolate Z evaluations over boolean hypercube {0, 1}^n
    #[inline]
    pub fn interpolate_over_hypercube(&self) -> Vec<F> {
        // Take eq poly as an example:
        //
        // The evaluation format of an eq poly over {0, 1}^2 follows:
        // eq(\vec{r}, \vec{x}) with \vec{x} order in x0 x1
        //
        //     00             01            10          11
        // (1-r0)(1-r1)    (1-r0)r1      r0(1-r1)      r0r1
        //
        // The interpolated version over x0 x1 (ordered in x0 x1) follows:
        //
        //     00               01                  10                11
        // (1-r0)(1-r1)    (1-r0)(2r1-1)      (2r0-1)(1-r1)     (2r0-1)(2r1-1)

        // NOTE(Hang): I think the original implementation of this dense multilinear
        // polynomial requires a resizing of coeffs by num vars,
        // e.g., when sumchecking - the num_var reduces, while Z evals can reuse the
        // whole space, which means we cannot simply relying Z's size itself.
        Self::interpolate_over_hypercube_impl(&self.coeffs)
    }

    /// Evaluate the polynomial at the top variable
    #[inline]
    pub fn fix_top_variable<AF: Field + Mul<F, Output = F>>(&mut self, r: AF) {
        let n = self.coeffs.len() / 2;
        let (left, right) = self.coeffs.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += r * (*b - *a);
        });
        self.coeffs.truncate(n);
    }

    #[inline]
    pub fn fix_bottom_variable(&mut self, point: &F) {
        let n = self.coeffs.len() / 2;

        // evaluate single variable of partial point from left to right

        for i in 0..n {
            self.coeffs[i] =
                self.coeffs[i] + (self.coeffs[(i << 1) + 1] - self.coeffs[i << 1]) * point;
        }
        self.coeffs.truncate(n);
    }

    /// Hyperplonk's implementation
    /// Evaluate the polynomial at a set of variables, from top to bottom
    /// This is equivalent to `evaluate` when partial_point.len() = nv
    ///
    /// This matches the behavior of Hyperplonk's polynomial evaluation
    #[inline]
    pub fn eval_reverse_order<AF: Field + Mul<F, Output = F>>(&self, partial_point: &[AF]) -> F {
        // evaluate single variable of partial point from left to right
        let mut tmp = self.clone();
        partial_point
            .iter()
            .for_each(|point| tmp.fix_top_variable(*point));

        tmp.coeffs[0]
    }

    /// Hyperplonk's implementation
    /// Evaluate the polynomial at a set of variables, from bottom to top
    /// This is equivalent to `evaluate` when partial_point.len() = nv
    #[inline]
    pub fn fix_variables<AF: Field + Mul<F, Output = F>>(&mut self, partial_point: &[AF]) {
        // evaluate single variable of partial point from left to right
        partial_point
            .iter()
            .rev() // need to reverse the order of the point
            .for_each(|point| self.fix_top_variable(*point));
    }

    /// Jolt's implementation
    /// Evaluate the polynomial at a set of variables, from bottom to top
    /// This is equivalent to `evaluate_with_buffer`, but slower
    /// returns Z(r) in O(n) time
    #[inline]
    pub fn evaluate_jolt(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::evals_jolt(r);
        assert_eq!(chis.len(), self.coeffs.len());
        self.coeffs
            .iter()
            .zip(chis.iter())
            .map(|(c, chi)| *c * *chi)
            .sum()
    }

    #[inline]
    /// Expander's implementation
    /// Generic method to evaluate a multilinear polynomial.
    /// This is the preferred method to evaluate a multilinear polynomial
    /// as it does not require additional memory.
    pub fn evaluate_with_buffer<ChallengeF, EvalF>(
        evals: &[F],
        point: &[ChallengeF],
        scratch: &mut [EvalF],
    ) -> EvalF
    where
        ChallengeF: Field + Mul<F, Output = EvalF>,
        EvalF: Field
            + From<F>
            + Mul<F, Output = EvalF>
            + Add<F, Output = EvalF>
            + Mul<ChallengeF, Output = EvalF>,
    {
        assert_eq!(1 << point.len(), evals.len());
        assert!(scratch.len() >= evals.len());

        if point.is_empty() {
            evals[0].into()
        } else {
            for i in 0..(evals.len() >> 1) {
                scratch[i] = point[0] * (evals[i * 2 + 1] - evals[i * 2]) + evals[i * 2];
            }

            let mut cur_eval_size = evals.len() >> 2;
            for r in point.iter().skip(1) {
                for i in 0..cur_eval_size {
                    scratch[i] = scratch[i * 2] + (scratch[i * 2 + 1] - scratch[i * 2]) * *r;
                }
                cur_eval_size >>= 1;
            }
            scratch[0]
        }
    }
}

impl<F: Field> Add<&MultiLinearPoly<F>> for MultiLinearPoly<F> {
    type Output = Self;

    fn add(self, other: &MultiLinearPoly<F>) -> Self::Output {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        MultiLinearPoly { coeffs }
    }
}

impl<F: Field> Mul<&F> for &MultiLinearPoly<F> {
    type Output = MultiLinearPoly<F>;

    fn mul(self, scalar: &F) -> Self::Output {
        let coeffs = self.coeffs.iter().map(|c| *c * scalar).collect();
        MultiLinearPoly { coeffs }
    }
}

impl<F: Field> Index<usize> for MultiLinearPoly<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}

impl<F: Field> MultilinearExtension<F> for MultiLinearPoly<F> {
    fn num_vars(&self) -> usize {
        self.get_num_vars()
    }

    fn hypercube_basis(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    fn hypercube_basis_ref(&self) -> &[F] {
        &self.coeffs
    }

    fn evaluate_with_buffer<ChallengeF, EvalF>(
        &self,
        point: &[ChallengeF],
        scratch: &mut [EvalF],
    ) -> EvalF
    where
        ChallengeF: Field + Mul<F, Output = EvalF>,
        EvalF: Field
            + From<F>
            + Mul<F, Output = EvalF>
            + Add<F, Output = EvalF>
            + Mul<ChallengeF, Output = EvalF>,
    {
        Self::evaluate_with_buffer(&self.coeffs, point, scratch)
    }

    fn interpolate_over_hypercube(&self) -> Vec<F> {
        self.interpolate_over_hypercube()
    }
}

impl<F: Field> IndexMut<usize> for MultiLinearPoly<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coeffs[index]
    }
}

impl<F: Field> MutableMultilinearExtension<F> for MultiLinearPoly<F> {
    fn fix_top_variable<AF: Field + std::ops::Mul<F, Output = F>>(&mut self, r: AF) {
        self.fix_top_variable(r)
    }

    fn fix_variables<AF: Field + std::ops::Mul<F, Output = F>>(&mut self, vars: &[AF]) {
        self.fix_variables(vars)
    }

    fn interpolate_over_hypercube_in_place(&mut self) {
        let num_vars = self.num_vars();
        for i in 1..=num_vars {
            let chunk_size = 1 << i;

            self.coeffs.chunks_mut(chunk_size).for_each(|chunk| {
                let half_chunk = chunk_size >> 1;
                let (left, right) = chunk.split_at_mut(half_chunk);
                right
                    .iter_mut()
                    .zip(left.iter())
                    .for_each(|(a, b)| *a -= *b);
            })
        }
    }

    fn lift_to_n_vars(&mut self, n_vars: usize) {
        self.coeffs.resize(1 << n_vars, F::ZERO);
    }
}
