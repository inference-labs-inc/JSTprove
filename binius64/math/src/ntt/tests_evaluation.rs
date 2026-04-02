// Copyright 2025 Irreducible Inc.
//! This module tests that running the NTT is equivalent to evaluating the respective polynomial at
//! the respective points of $S^{(0)}$.

use std::ops::{Add, AddAssign, Mul, MulAssign};

use binius_field::{BinaryField, Field};
use rand::prelude::*;

use crate::{
    BinarySubspace,
    ntt::{
        AdditiveNTT, DomainContext, NeighborsLastSingleThread,
        domain_context::{self},
    },
    test_utils::random_field_buffer,
};

/// Represents a univariate polynomial with coefficients in a field.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Polynomial<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    pub fn new(coefficients: Vec<F>) -> Self {
        let mut poly = Self { coefficients };
        poly.trim();
        poly
    }

    pub fn zero() -> Self {
        Self {
            coefficients: vec![],
        }
    }

    pub fn one() -> Self {
        Self {
            coefficients: vec![F::ONE],
        }
    }

    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    fn trim(&mut self) {
        while let Some(&last) = self.coefficients.last() {
            if last == F::ZERO {
                self.coefficients.pop();
            } else {
                break;
            }
        }
    }

    pub fn evaluate(&self, x: F) -> F {
        if self.coefficients.is_empty() {
            return F::ZERO;
        }

        // Horner's method for efficient evaluation
        self.coefficients
            .iter()
            .rev()
            .fold(F::ZERO, |acc, &coeff| acc * x + coeff)
    }
}

// Polynomial + Polynomial
impl<F: Field> Add for Polynomial<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coefficients.get(i).copied().unwrap_or(F::ZERO);
            let b = other.coefficients.get(i).copied().unwrap_or(F::ZERO);
            result.push(a + b);
        }

        Self::new(result)
    }
}

// Polynomial += Polynomial
impl<F: Field> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, other: Self) {
        *self = std::mem::take(self) + other;
    }
}

// Polynomial + Scalar
impl<F: Field> Add<F> for Polynomial<F> {
    type Output = Self;

    fn add(mut self, scalar: F) -> Self {
        if self.coefficients.is_empty() {
            self.coefficients.push(scalar);
        } else {
            self.coefficients[0] += scalar;
        }
        self.trim();
        self
    }
}

// Polynomial += Scalar
impl<F: Field> AddAssign<F> for Polynomial<F> {
    fn add_assign(&mut self, scalar: F) {
        if self.coefficients.is_empty() {
            self.coefficients.push(scalar);
        } else {
            self.coefficients[0] += scalar;
        }
        self.trim();
    }
}

// &Polynomial * &Polynomial
impl<F: Field> Mul<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, other: &Polynomial<F>) -> Polynomial<F> {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Polynomial::zero();
        }

        let result_len = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![F::ZERO; result_len];

        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }

        Polynomial::new(result)
    }
}

// Polynomial *= &Polynomial
impl<F: Field> MulAssign<&Polynomial<F>> for Polynomial<F> {
    fn mul_assign(&mut self, other: &Polynomial<F>) {
        *self = &*self * other;
    }
}

// &Polynomial * Scalar
impl<F: Field> Mul<F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, scalar: F) -> Polynomial<F> {
        if scalar == F::ZERO {
            return Polynomial::zero();
        }

        let coefficients = self
            .coefficients
            .iter()
            .map(|&coeff| coeff * scalar)
            .collect();

        Polynomial::new(coefficients)
    }
}

// Polynomial *= Scalar
impl<F: Field> MulAssign<F> for Polynomial<F> {
    fn mul_assign(&mut self, scalar: F) {
        if scalar == F::ZERO {
            *self = Self::zero();
        } else {
            for coeff in &mut self.coefficients {
                *coeff *= scalar;
            }
        }
    }
}

/// Computes the subspace polynomial of a given binary field subspace $V$.
///
/// That is, it computes $\prod_{a \in V} (X - a)$.
fn subspace_polynomial<F: BinaryField>(subspace: BinarySubspace<F>) -> Polynomial<F> {
    let mut poly = Polynomial::one();

    for elem in subspace.iter() {
        // the polynomial `X - elem`
        let linear_factor = Polynomial::new(vec![elem, F::ONE]);

        poly *= &linear_factor;
    }

    // deg(poly) = |subspace|
    assert_eq!(poly.degree(), 1 << subspace.dim());

    poly
}

/// Computes the novel polynomial basis associated to a domain context.
///
/// The novel basis is given by: $[1, \hat{W}_0, \hat{W}_1, \hat{W}_1 \hat{W}_0, \hat{W}_2,
/// \hat{W}_2 \hat{W}_0, ...]$
///
/// Here:
/// - $\hat{W}_i = W_i / W_i(beta_i)$
/// - $W_i$ is the subspace polynomial of `domain_context.subspace(i)`
/// - $beta_i$
fn novel_basis<DC: DomainContext>(domain_context: &DC) -> Vec<Polynomial<DC::Field>> {
    let log_d = domain_context.log_domain_size();
    let domain = domain_context.subspace(log_d);

    // collect subspace polynomials $W_i$ (this is *not* yet $\hat{W}_i$)
    let mut w_hat: Vec<_> = (0..log_d)
        .map(|i| domain.reduce_dim(i))
        .map(subspace_polynomial)
        .collect();
    // and normalize them to get $\hat{W}_i$
    for i in 0..log_d {
        let beta_i = domain.basis()[i];
        let eval = w_hat[i].evaluate(beta_i);
        w_hat[i] *= eval.invert().unwrap();
    }

    // construct novel polynomial basis
    let mut novel_basis = Vec::with_capacity(1 << log_d);
    novel_basis.push(Polynomial::one());
    for i in 0..log_d {
        for j in 0..novel_basis.len() {
            novel_basis.push(&novel_basis[j] * &w_hat[i])
        }
    }

    novel_basis
}

fn test_equivalence<F: BinaryField, NTT: AdditiveNTT<Field = F>>(ntt: &NTT) {
    let novel_basis = novel_basis(ntt.domain_context());
    let log_d = ntt.log_domain_size();

    // generate random coefficients for the novel basis
    let mut rng = StdRng::seed_from_u64(0);
    let novel_coeffs = random_field_buffer::<F>(&mut rng, log_d);

    // way 1 to compute evaluations: compute polynomial and evaluate
    assert_eq!(novel_basis.len(), novel_coeffs.len());
    let poly = novel_basis
        .iter()
        .zip(novel_coeffs.as_ref())
        .fold(Polynomial::zero(), |acc, (b, coeff)| acc + b * *coeff);
    let evals: Vec<_> = ntt
        .domain_context()
        .subspace(log_d)
        .iter()
        .map(|a| poly.evaluate(a))
        .collect();

    // way 2 to compute evaluations: use NTT
    let mut ntt_data = novel_coeffs.clone();
    ntt.forward_transform(ntt_data.to_mut(), 0, 0);

    // check equivalence
    assert_eq!(ntt_data.as_ref(), &evals);
}

#[test]
fn test_eval() {
    const LOG_D: usize = 6;
    type F = binius_field::BinaryField128bGhash;

    // GaoMateer domain context
    let domain_context = domain_context::GaoMateerPreExpanded::<F>::generate(LOG_D);
    let ntt = NeighborsLastSingleThread::new(domain_context);
    test_equivalence(&ntt);

    // Generic domain context with the usual subspace
    let subspace = BinarySubspace::with_dim(LOG_D);
    let domain_context = domain_context::GenericPreExpanded::<F>::generate_from_subspace(&subspace);
    let ntt = NeighborsLastSingleThread::new(domain_context);
    test_equivalence(&ntt);

    // Generic domain context context with a subspace whose first basis element is _not_ 1
    let basis = vec![F::new(5), F::new(7), F::new(22), F::new(95)];
    let subspace = BinarySubspace::new_unchecked(basis);
    let domain_context = domain_context::GenericPreExpanded::<F>::generate_from_subspace(&subspace);
    let ntt = NeighborsLastSingleThread::new(domain_context);
    test_equivalence(&ntt);
}
