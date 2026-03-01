use std::{iter::Sum, ops::Mul};

use halo2curves::ff::Field;
use itertools::izip;

#[inline(always)]
pub(crate) fn powers_series<F: Field>(x: &F, n: usize) -> Vec<F> {
    let mut powers = vec![F::ONE];
    let mut cur = *x;
    for _ in 0..n - 1 {
        powers.push(cur);
        cur *= x;
    }
    powers
}

/// Given a univariate polynomial of coefficient form f(X) = c0 + c1 X + ... + cn X^n
/// and perform the division f(X) / (X - \alpha).
#[inline(always)]
pub(crate) fn univariate_degree_one_quotient<F: Field>(coeffs: &[F], alpha: F) -> (Vec<F>, F) {
    let mut div_coeffs = coeffs.to_vec();

    for i in (1..coeffs.len()).rev() {
        // c X^n = c X^n - c \alpha X^(n - 1) + c \alpha X^(n - 1)
        //       = c (X - \alpha) X^(n - 1) + c \alpha X^(n - 1)

        let remainder = div_coeffs[i] * alpha;
        div_coeffs[i - 1] += remainder;
    }

    let final_remainder = div_coeffs[0];
    let mut final_div_coeffs = div_coeffs[1..].to_owned();
    final_div_coeffs.resize(coeffs.len(), F::ZERO);

    (final_div_coeffs, final_remainder)
}

#[inline(always)]
pub(crate) fn univariate_roots_quotient<F: Field>(mut coeffs: Vec<F>, roots: &[F]) -> Vec<F> {
    roots.iter().enumerate().for_each(|(ith_root, r)| {
        for i in ((1 + ith_root)..coeffs.len()).rev() {
            // c X^n = c X^n - c \alpha X^(n - 1) + c \alpha X^(n - 1)
            //       = c (X - \alpha) X^(n - 1) + c \alpha X^(n - 1)

            let remainder = coeffs[i] * r;
            coeffs[i - 1] += remainder;
        }

        assert_eq!(coeffs[ith_root], F::ZERO);
    });

    coeffs[roots.len()..].to_owned()
}

#[inline(always)]
pub(crate) fn univariate_evaluate<F: Mul<F1, Output = F> + Sum + Copy, F1: Field>(
    coeffs: &[F],
    power_series: &[F1],
) -> F {
    izip!(coeffs, power_series).map(|(c, p)| *c * *p).sum()
}

// NOTE(HS) used by mercury prover only, perform division f(X) = q(X) (X^b - \alpha) + r(X)
#[allow(unused)]
#[inline(always)]
pub(crate) fn univariate_degree_b_quotient<F: Field>(
    coeffs: &[F],
    degree_b: usize,
    alpha: F,
) -> (Vec<F>, Vec<F>) {
    assert!(coeffs.len() >= degree_b);

    let mut local_coeffs = coeffs.to_vec();

    (degree_b..coeffs.len()).rev().for_each(|degree| {
        let remainder = alpha * local_coeffs[degree];
        local_coeffs[degree - degree_b] += remainder;
    });

    let quotient = local_coeffs[degree_b..].to_owned();
    let remainder = local_coeffs[..degree_b].to_owned();

    (quotient, remainder)
}

#[inline(always)]
pub(crate) fn polynomial_add<F: Field>(coeffs: &mut Vec<F>, weight: F, another_coeffs: &[F]) {
    if coeffs.len() < another_coeffs.len() {
        coeffs.resize(another_coeffs.len(), F::ZERO);
    }

    izip!(coeffs, another_coeffs).for_each(|(c, a)| *c += weight * *a);
}

#[inline(always)]
pub(crate) fn coeff_form_degree2_lagrange<F: Field>(roots: [F; 3], evals: [F; 3]) -> [F; 3] {
    let [r0, r1, r2] = roots;
    let [e0, e1, e2] = evals;

    let r0_nom = [r1 * r2, -r1 - r2, F::ONE];
    let r0_denom_inv = ((r0 - r1) * (r0 - r2)).invert().unwrap();
    let r0_weight = r0_denom_inv * e0;

    let r1_nom = [r0 * r2, -r0 - r2, F::ONE];
    let r1_denom_inv = ((r1 - r0) * (r1 - r2)).invert().unwrap();
    let r1_weight = r1_denom_inv * e1;

    let r2_nom = [r0 * r1, -r0 - r1, F::ONE];
    let r2_denom_inv = ((r2 - r0) * (r2 - r1)).invert().unwrap();
    let r2_weight = r2_denom_inv * e2;

    let combine = |a, b, c| a * r0_weight + b * r1_weight + c * r2_weight;

    [
        combine(r0_nom[0], r1_nom[0], r2_nom[0]),
        combine(r0_nom[1], r1_nom[1], r2_nom[1]),
        combine(r0_nom[2], r1_nom[2], r2_nom[2]),
    ]
}

#[cfg(test)]
mod test {
    use halo2curves::{bn256::Fr, ff::Field};

    use crate::*;

    #[test]
    fn test_univariate_degree_one_quotient() {
        {
            // x^3 + 1 = (x + 1)(x^2 - x + 1)
            let poly = vec![
                Fr::from(1u64),
                Fr::from(0u64),
                Fr::from(0u64),
                Fr::from(1u64),
            ];
            let point = -Fr::from(1u64);
            let (div, remainder) = univariate_degree_one_quotient(&poly, point);
            assert_eq!(
                div,
                vec![
                    Fr::from(1u64),
                    -Fr::from(1u64),
                    Fr::from(1u64),
                    Fr::from(0u64)
                ]
            );
            assert_eq!(remainder, Fr::ZERO)
        }
        {
            // x^3 - 1 = (x-1)(x^2 + x + 1)
            let poly = vec![
                -Fr::from(1u64),
                Fr::from(0u64),
                Fr::from(0u64),
                Fr::from(1u64),
            ];
            let point = Fr::from(1u64);
            let (div, remainder) = univariate_degree_one_quotient(&poly, point);
            assert_eq!(
                div,
                vec![
                    Fr::from(1u64),
                    Fr::from(1u64),
                    Fr::from(1u64),
                    Fr::from(0u64)
                ]
            );
            assert_eq!(remainder, Fr::ZERO)
        }
        {
            // x^3 + 6x^2 + 11x + 6 = (x + 1)(x + 2)(x + 3)
            let poly = vec![
                Fr::from(6u64),
                Fr::from(11u64),
                Fr::from(6u64),
                Fr::from(1u64),
            ];
            let point = Fr::from(1u64);
            let (div, remainder) = univariate_degree_one_quotient(&poly, point);
            assert_eq!(
                div,
                vec![
                    Fr::from(18u64),
                    Fr::from(7u64),
                    Fr::from(1u64),
                    Fr::from(0u64),
                ]
            );
            assert_eq!(remainder, Fr::from(24u64))
        }
    }
}
