// Copyright 2023-2025 Irreducible Inc.

use std::ops::{Add, AddAssign, Index, Mul, MulAssign};

use binius_field::{Field, field::FieldOps};
use binius_math::univariate::evaluate_univariate;

/// A univariate polynomial in monomial basis.
///
/// The coefficient at position `i` in the inner vector corresponds to the term $X^i$.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoundCoeffs<F>(pub Vec<F>);

impl<F> RoundCoeffs<F> {
    /// Truncate one coefficient from the polynomial to a more compact round proof.
    pub fn truncate(mut self) -> RoundProof<F> {
        self.0.pop();
        RoundProof(self)
    }
}

impl<F: FieldOps> RoundCoeffs<F> {
    /// Evaluate the polynomial at a point.
    pub fn evaluate(&self, x: F) -> F {
        evaluate_univariate(&self.0, x)
    }
}

impl<F: Field> Add<&Self> for RoundCoeffs<F> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: Field> AddAssign<&Self> for RoundCoeffs<F> {
    fn add_assign(&mut self, rhs: &Self) {
        if self.0.len() < rhs.0.len() {
            self.0.resize(rhs.0.len(), F::ZERO);
        }

        for (lhs_i, &rhs_i) in self.0.iter_mut().zip(rhs.0.iter()) {
            *lhs_i += rhs_i;
        }
    }
}

impl<F: Field> Mul<F> for RoundCoeffs<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<F: Field> MulAssign<F> for RoundCoeffs<F> {
    fn mul_assign(&mut self, rhs: F) {
        for coeff in &mut self.0 {
            *coeff *= rhs;
        }
    }
}

impl<F: Field> std::iter::Sum for RoundCoeffs<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, x| acc + &x)
    }
}

impl<F> Index<usize> for RoundCoeffs<F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        &self.0[index]
    }
}

/// A sumcheck round proof is a univariate polynomial in monomial basis with the coefficient of the
/// highest-degree term truncated off.
///
/// Since the verifier knows the claimed sum of the polynomial values at the points 0 and 1, the
/// high-degree term coefficient can be easily recovered. Truncating the coefficient off saves a
/// small amount of proof data.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoundProof<F>(pub RoundCoeffs<F>);

impl<F> RoundProof<F> {
    /// The truncated polynomial coefficients.
    pub fn coeffs(&self) -> &[F] {
        &self.0.0
    }
}

impl<F: FieldOps> RoundProof<F> {
    /// Recovers all univariate polynomial coefficients from the compressed round proof.
    ///
    /// The prover has sent coefficients for the purported ith round polynomial
    /// $r_i(X) = \sum_{j=0}^d a_j * X^j$.
    /// However, the prover has not sent the highest degree coefficient $a_d$.
    /// The verifier will need to recover this missing coefficient.
    ///
    /// Let $s$ denote the current round's claimed sum.
    /// The verifier expects the round polynomial $r_i$ to satisfy the identity
    /// $s = r_i(0) + r_i(1)$.
    /// Using
    ///     $r_i(0) = a_0$
    ///     $r_i(1) = \sum_{j=0}^d a_j$
    /// There is a unique $a_d$ that allows $r_i$ to satisfy the above identity.
    /// Specifically
    ///     $a_d = s - a_0 - \sum_{j=0}^{d-1} a_j$
    ///
    /// Not sending the whole round polynomial is an optimization.
    /// In the unoptimized version of the protocol, the verifier will halt and reject
    /// if given a round polynomial that does not satisfy the above identity.
    pub fn recover(self, sum: F) -> RoundCoeffs<F>
    where
        F: FieldOps,
    {
        let Self(RoundCoeffs(mut coeffs)) = self;
        let first_coeff = coeffs.first().cloned().unwrap_or_else(F::zero);
        let last_coeff = sum - first_coeff - coeffs.iter().cloned().sum::<F>();
        coeffs.push(last_coeff);
        RoundCoeffs(coeffs)
    }
}
