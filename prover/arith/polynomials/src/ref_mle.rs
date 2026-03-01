use std::ops::{Add, Index, IndexMut, Mul};

use arith::Field;

use crate::MultiLinearPoly;

pub trait MultilinearExtension<F: Field>: Index<usize, Output = F> + Send + Sync {
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
            + Mul<ChallengeF, Output = EvalF>;

    fn evaluate<ChallengeF, EvalF>(&self, point: &[ChallengeF]) -> EvalF
    where
        ChallengeF: Field + Mul<F, Output = EvalF>,
        EvalF: Field
            + From<F>
            + Mul<F, Output = EvalF>
            + Add<F, Output = EvalF>
            + Mul<ChallengeF, Output = EvalF>,
    {
        let mut scratch = vec![EvalF::ZERO; self.hypercube_size()];
        self.evaluate_with_buffer(point, &mut scratch)
    }

    fn num_vars(&self) -> usize;

    #[inline(always)]
    fn hypercube_size(&self) -> usize {
        1 << self.num_vars()
    }

    fn hypercube_basis(&self) -> Vec<F>;

    fn hypercube_basis_ref(&self) -> &[F];

    fn interpolate_over_hypercube(&self) -> Vec<F>;
}

#[derive(Debug, Clone)]
pub struct RefMultiLinearPoly<'ref_life, F: Field> {
    pub coeffs: &'ref_life [F],
}

impl<'ref_life, 'outer: 'ref_life, F: Field> RefMultiLinearPoly<'ref_life, F> {
    #[inline(always)]
    pub fn from_ref(evals: &'outer [F]) -> Self {
        Self { coeffs: evals }
    }
}

impl<'a, F: Field> Index<usize> for RefMultiLinearPoly<'a, F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.hypercube_size());
        &self.coeffs[index]
    }
}

impl<'a, F: Field> MultilinearExtension<F> for RefMultiLinearPoly<'a, F> {
    #[inline(always)]
    fn num_vars(&self) -> usize {
        assert!(self.coeffs.len().is_power_of_two());
        self.coeffs.len().ilog2() as usize
    }

    #[inline(always)]
    fn hypercube_basis(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }

    #[inline(always)]
    fn hypercube_basis_ref(&self) -> &[F] {
        self.coeffs
    }

    #[inline(always)]
    fn interpolate_over_hypercube(&self) -> Vec<F> {
        MultiLinearPoly::interpolate_over_hypercube_impl(self.coeffs)
    }

    #[inline(always)]
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
        MultiLinearPoly::evaluate_with_buffer(self.coeffs, point, scratch)
    }
}

pub trait MutableMultilinearExtension<F: Field>:
    MultilinearExtension<F> + IndexMut<usize, Output = F>
{
    fn fix_top_variable<AF: Field + Mul<F, Output = F>>(&mut self, r: AF);

    fn fix_variables<AF: Field + Mul<F, Output = F>>(&mut self, vars: &[AF]);

    fn interpolate_over_hypercube_in_place(&mut self);

    fn lift_to_n_vars(&mut self, n_vars: usize);
}

#[derive(Debug)]
pub struct MutRefMultiLinearPoly<'ref_life, F: Field> {
    pub coeffs: &'ref_life mut Vec<F>,
}

impl<'ref_life, 'outer_mut: 'ref_life, F: Field> MutRefMultiLinearPoly<'ref_life, F> {
    #[inline(always)]
    pub fn from_ref(evals: &'outer_mut mut Vec<F>) -> Self {
        Self { coeffs: evals }
    }
}

impl<'a, F: Field> Index<usize> for MutRefMultiLinearPoly<'a, F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.hypercube_size());
        &self.coeffs[index]
    }
}

impl<'a, F: Field> IndexMut<usize> for MutRefMultiLinearPoly<'a, F> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.hypercube_size());
        &mut self.coeffs[index]
    }
}

impl<'a, F: Field> MultilinearExtension<F> for MutRefMultiLinearPoly<'a, F> {
    #[inline(always)]
    fn num_vars(&self) -> usize {
        assert!(self.coeffs.len().is_power_of_two());
        self.coeffs.len().ilog2() as usize
    }

    #[inline(always)]
    fn hypercube_basis(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    #[inline(always)]
    fn hypercube_basis_ref(&self) -> &[F] {
        self.coeffs
    }

    #[inline(always)]
    fn interpolate_over_hypercube(&self) -> Vec<F> {
        MultiLinearPoly::interpolate_over_hypercube_impl(self.coeffs)
    }

    #[inline(always)]
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
        MultiLinearPoly::evaluate_with_buffer(self.coeffs, point, scratch)
    }
}

impl<'a, F: Field> MutableMultilinearExtension<F> for MutRefMultiLinearPoly<'a, F> {
    #[inline(always)]
    fn fix_top_variable<AF: Field + Mul<F, Output = F>>(&mut self, r: AF) {
        let n = self.hypercube_size() / 2;
        let (left, right) = self.coeffs.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += r * (*b - *a);
        });
        self.coeffs.truncate(n)
    }

    #[inline(always)]
    fn fix_variables<AF: Field + Mul<F, Output = F>>(&mut self, vars: &[AF]) {
        // evaluate single variable of partial point from left to right
        // need to reverse the order of the point
        vars.iter().rev().for_each(|p| self.fix_top_variable(*p))
    }

    #[inline(always)]
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

    #[inline(always)]
    fn lift_to_n_vars(&mut self, n_vars: usize) {
        self.coeffs.resize(1 << n_vars, F::ZERO);
    }
}
