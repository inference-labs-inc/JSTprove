// Copyright 2024-2025 Irreducible Inc.

use std::{
    iter::repeat_with,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
};

use binius_field::{ExtensionField, Field};
use bytemuck::zeroed_slice_box;
use getset::CopyGetters;
use rand::RngCore;

/// A matrix over a field.
#[derive(Debug, Clone, PartialEq, Eq, CopyGetters)]
pub struct Matrix<F: Field> {
    /// The number of rows.
    #[getset(get_copy = "pub")]
    m: usize,
    /// The number of columns.
    #[getset(get_copy = "pub")]
    n: usize,
    elements: Box<[F]>,
}

impl<F: Field> Matrix<F> {
    pub fn new(m: usize, n: usize, elements: &[F]) -> Self {
        assert_eq!(
            elements.len(),
            m * n,
            "precondition: elements length must equal m * n"
        );
        Self {
            m,
            n,
            elements: elements.into(),
        }
    }

    pub fn zeros(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            elements: zeroed_slice_box(m * n),
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut out = Self::zeros(n, n);
        for i in 0..n {
            out[(i, i)] = F::ONE;
        }
        out
    }

    fn fill_identity(&mut self) {
        assert_eq!(self.m, self.n);
        self.elements.fill(F::ZERO);
        for i in 0..self.n {
            self[(i, i)] = F::ONE;
        }
    }

    pub const fn elements(&self) -> &[F] {
        &self.elements
    }

    pub fn random(m: usize, n: usize, mut rng: impl RngCore) -> Self {
        Self {
            m,
            n,
            elements: repeat_with(|| F::random(&mut rng)).take(m * n).collect(),
        }
    }

    pub const fn dim(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.dim(), other.dim());
        self.elements.copy_from_slice(&other.elements);
    }

    pub fn mul_into(a: &Self, b: &Self, c: &mut Self) {
        assert_eq!(a.n(), b.m());
        assert_eq!(a.m(), c.m());
        assert_eq!(b.n(), c.n());

        for i in 0..c.m() {
            for j in 0..c.n() {
                c[(i, j)] = (0..a.n()).map(|k| a[(i, k)] * b[(k, j)]).sum();
            }
        }
    }

    pub fn mul_vec_into<FE: ExtensionField<F>>(&self, x: &[FE], y: &mut [FE]) {
        assert_eq!(self.n(), x.len());
        assert_eq!(self.m(), y.len());

        for i in 0..y.len() {
            y[i] = (0..self.n()).map(|j| x[j] * self[(i, j)]).sum();
        }
    }

    /// Invert a square matrix
    ///
    /// ## Preconditions
    ///
    /// * Matrix must be square (m == n)
    /// * Matrix must be non-singular (invertible)
    /// * `out` must have the same dimensions as `self`
    pub fn inverse_into(&self, out: &mut Self) {
        assert_eq!(self.dim(), out.dim());
        assert_eq!(self.m, self.n, "precondition: matrix must be square");

        let n = self.n;

        let mut tmp = self.clone();
        out.fill_identity();

        let mut row_buffer = vec![F::ZERO; n];

        for i in 0..n {
            // Find the pivot row
            let pivot = (i..n)
                .find(|&pivot| tmp[(pivot, i)] != F::ZERO)
                .expect("precondition: matrix must be non-singular");
            if pivot != i {
                tmp.swap_rows(i, pivot, &mut row_buffer);
                out.swap_rows(i, pivot, &mut row_buffer);
            }

            // Normalize the pivot
            let scalar = tmp[(i, i)]
                .invert()
                .expect("pivot is checked to be non-zero above");
            tmp.scale_row(i, scalar);
            out.scale_row(i, scalar);

            // Clear the pivot column
            for j in (0..i).chain(i + 1..n) {
                let scalar = tmp[(j, i)];
                tmp.sub_pivot_row(j, i, scalar);
                out.sub_pivot_row(j, i, scalar);
            }
        }

        debug_assert_eq!(tmp, Self::identity(n));
    }

    fn row_ref(&self, i: usize) -> &[F] {
        assert!(i < self.m);
        &self.elements[i * self.n..(i + 1) * self.n]
    }

    fn row_mut(&mut self, i: usize) -> &mut [F] {
        assert!(i < self.m);
        &mut self.elements[i * self.n..(i + 1) * self.n]
    }

    fn swap_rows(&mut self, i0: usize, i1: usize, buffer: &mut [F]) {
        assert!(i0 < self.m);
        assert!(i1 < self.m);
        assert_eq!(buffer.len(), self.n);

        if i0 == i1 {
            return;
        }

        buffer.copy_from_slice(self.row_ref(i1));
        self.elements
            .copy_within(i0 * self.n..(i0 + 1) * self.n, i1 * self.n);
        self.row_mut(i0).copy_from_slice(buffer);
    }

    fn scale_row(&mut self, i: usize, scalar: F) {
        for x in self.row_mut(i) {
            *x *= scalar;
        }
    }

    fn sub_pivot_row(&mut self, i0: usize, i1: usize, scalar: F) {
        assert!(i0 < self.m);
        assert!(i1 < self.m);

        for j in 0..self.n {
            let x = self[(i1, j)];
            self[(i0, j)] -= x * scalar;
        }
    }
}

impl<F: Field> Index<(usize, usize)> for Matrix<F> {
    type Output = F;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        assert!(i < self.m);
        assert!(j < self.n);
        &self.elements[i * self.n + j]
    }
}

impl<F: Field> IndexMut<(usize, usize)> for Matrix<F> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        assert!(i < self.m);
        assert!(j < self.n);
        &mut self.elements[i * self.n + j]
    }
}

impl<F: Field> Add<Self> for &Matrix<F> {
    type Output = Matrix<F>;

    fn add(self, rhs: Self) -> Matrix<F> {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

impl<F: Field> Sub<Self> for &Matrix<F> {
    type Output = Matrix<F>;

    fn sub(self, rhs: Self) -> Matrix<F> {
        let mut out = self.clone();
        out -= rhs;
        out
    }
}

impl<F: Field> AddAssign<&Self> for Matrix<F> {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.dim(), rhs.dim());
        for (a_ij, &b_ij) in self.elements.iter_mut().zip(rhs.elements.iter()) {
            *a_ij += b_ij;
        }
    }
}

impl<F: Field> SubAssign<&Self> for Matrix<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.dim(), rhs.dim());
        for (a_ij, &b_ij) in self.elements.iter_mut().zip(rhs.elements.iter()) {
            *a_ij -= b_ij;
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::test_utils::B128;

    type F = B128;

    proptest! {
        #[test]
        fn test_left_linearity(c_m in 0..8usize, c_n in 0..8usize, a_n in 0..8usize) {
            let mut rng = StdRng::seed_from_u64(0);
            let a0 = Matrix::<F>::random(c_m, a_n, &mut rng);
            let a1 = Matrix::<F>::random(c_m, a_n, &mut rng);
            let b = Matrix::<F>::random(a_n, c_n, &mut rng);
            let mut c0 = Matrix::<F>::zeros(c_m, c_n);
            let mut c1 = Matrix::<F>::zeros(c_m, c_n);

            let a0p1 = &a0 + &a1;
            let mut c0p1 = Matrix::<F>::zeros(c_m, c_n);

            Matrix::mul_into(&a0, &b, &mut c0);
            Matrix::mul_into(&a1, &b, &mut c1);
            Matrix::mul_into(&a0p1, &b, &mut c0p1);

            assert_eq!(c0p1, &c0 + &c1);
        }

        #[test]
        fn test_right_linearity(c_m in 0..8usize, c_n in 0..8usize, a_n in 0..8usize) {
            let mut rng = StdRng::seed_from_u64(0);
            let a = Matrix::<F>::random(c_m, a_n, &mut rng);
            let b0 = Matrix::<F>::random(a_n, c_n, &mut rng);
            let b1 = Matrix::<F>::random(a_n, c_n, &mut rng);
            let mut c0 = Matrix::<F>::zeros(c_m, c_n);
            let mut c1 = Matrix::<F>::zeros(c_m, c_n);

            let b0p1 = &b0 + &b1;
            let mut c0p1 = Matrix::<F>::zeros(c_m, c_n);

            Matrix::mul_into(&a, &b0, &mut c0);
            Matrix::mul_into(&a, &b1, &mut c1);
            Matrix::mul_into(&a, &b0p1, &mut c0p1);

            assert_eq!(c0p1, &c0 + &c1);
        }

        #[test]
        fn test_double_inverse(n in 0..8usize) {
            let mut rng = StdRng::seed_from_u64(0);
            let a = Matrix::<F>::random(n, n, &mut rng);
            let mut a_inv = Matrix::<F>::zeros(n, n);
            let mut a_inv_inv = Matrix::<F>::zeros(n, n);

            a.inverse_into(&mut a_inv);
            a_inv.inverse_into(&mut a_inv_inv);
            assert_eq!(a_inv_inv, a);
        }

        #[test]
        fn test_inverse(n in 0..8usize) {
            let mut rng = StdRng::seed_from_u64(0);
            let a = Matrix::<F>::random(n, n, &mut rng);
            let mut a_inv = Matrix::<F>::zeros(n, n);
            let mut prod = Matrix::<F>::zeros(n, n);

            a.inverse_into(&mut a_inv);

            Matrix::mul_into(&a, &a_inv, &mut prod);
            assert_eq!(prod, Matrix::<F>::identity(n));

            Matrix::mul_into(&a_inv, &a, &mut prod);
            assert_eq!(prod, Matrix::<F>::identity(n));
        }
    }
}
