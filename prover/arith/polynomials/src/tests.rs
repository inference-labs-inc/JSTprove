use super::*;
use arith::{FFTField, Field};
use ark_std::test_rng;
use halo2curves::bn256::Fr;

#[test]
fn test_scaled_eq_xr() {
    let mut rng = test_rng();
    for nv in 4..10 {
        let r: Vec<Fr> = (0..nv).map(|_| Fr::random_unsafe(&mut rng)).collect();
        let scalar = Fr::random_unsafe(&mut rng);

        // expander
        let mut eq_x_r1 = vec![Fr::zero(); 1 << nv];
        EqPolynomial::<Fr>::build_eq_x_r_with_buf(r.as_ref(), &scalar, &mut eq_x_r1);

        // jolt
        let eq_x_r2 = EqPolynomial::<Fr>::scaled_evals_jolt(r.as_ref(), &scalar);

        assert_eq!(eq_x_r1, eq_x_r2);
    }
}

#[test]
fn test_mle_eval() {
    let mut rng = test_rng();
    for nv in 4..10 {
        let mle = MultiLinearPoly::<Fr>::random(nv, &mut rng);
        let point = (0..nv)
            .map(|_| Fr::random_unsafe(&mut rng))
            .collect::<Vec<_>>();

        // jolt's method
        let eval_via_eq = mle.evaluate_jolt(&point);

        // hyperplonk's method
        let mut mle_eval = mle.clone();
        mle_eval.fix_variables(point.as_ref());
        assert!(mle_eval.coeffs.len() == 1);
        assert_eq!(eval_via_eq, mle_eval.coeffs[0]);

        // expander's method
        let mut buf = vec![Fr::zero(); 1 << nv];
        MultiLinearPoly::<Fr>::evaluate_with_buffer(
            mle.coeffs.as_ref(),
            point.as_ref(),
            buf.as_mut(),
        );
        assert_eq!(eval_via_eq, buf[0]);
    }
}

#[test]
fn test_eq_xr() {
    let mut rng = test_rng();
    for nv in 4..10 {
        let r: Vec<Fr> = (0..nv).map(|_| Fr::random_unsafe(&mut rng)).collect();

        // naive
        let eq_x_r0 = build_eq_x_r_for_test(r.as_ref());

        // hyperplonk
        let eq_x_r1 = EqPolynomial::<Fr>::build_eq_x_r(r.as_ref());
        assert_eq!(eq_x_r1, eq_x_r0);

        // expander
        let mut eq_x_r2 = vec![Fr::zero(); 1 << nv];
        EqPolynomial::<Fr>::build_eq_x_r_with_buf(r.as_ref(), &Fr::ONE, &mut eq_x_r2);
        assert_eq!(eq_x_r2, eq_x_r0);

        // jolt
        let eq_x_r3 = EqPolynomial::<Fr>::evals_jolt(r.as_ref());
        assert_eq!(eq_x_r3, eq_x_r0);
    }
}

#[test]
fn test_ref_multilinear_poly() {
    let mut rng = test_rng();
    for nv in 4..=10 {
        let es_len = 1 << nv;
        let es: Vec<Fr> = (0..es_len).map(|_| Fr::random_unsafe(&mut rng)).collect();
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random_unsafe(&mut rng)).collect();
        let mut scratch = vec![Fr::ZERO; es_len];

        let mle_from_ref = RefMultiLinearPoly::<Fr>::from_ref(&es);

        let actual_eval = mle_from_ref.evaluate_with_buffer(&point, &mut scratch);
        let expect_eval = MultiLinearPoly::evaluate_with_buffer(&es, &point, &mut scratch);

        drop(mle_from_ref);

        assert_eq!(actual_eval, expect_eval);

        drop(es);
    }
}

#[test]
fn test_mut_ref_multilinear_poly() {
    let mut rng = test_rng();
    for nv in 4..=10 {
        let es_len = 1 << nv;
        let mut es: Vec<Fr> = (0..es_len).map(|_| Fr::random_unsafe(&mut rng)).collect();
        let es_cloned = es.clone();
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random_unsafe(&mut rng)).collect();
        let mut scratch = vec![Fr::ZERO; es_len];

        let mut mle_from_mut_ref = MutRefMultiLinearPoly::<Fr>::from_ref(&mut es);

        mle_from_mut_ref.fix_variables(&point);
        let expect_eval = MultiLinearPoly::evaluate_with_buffer(&es_cloned, &point, &mut scratch);

        drop(mle_from_mut_ref);

        assert_eq!(es[0], expect_eval);

        drop(es);
    }
}

#[test]
fn test_ith_eq_vec_elem() {
    let mut rng = test_rng();
    for nv in 4..=10 {
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random_unsafe(&mut rng)).collect();
        let expected_evals = EqPolynomial::build_eq_x_r(&point);

        expected_evals
            .iter()
            .enumerate()
            .for_each(|(i, ev)| assert_eq!(EqPolynomial::ith_eq_vec_elem(&point, i), *ev))
    }
}

/// Naive method to build eq(x, r).
/// Only used for testing purpose.
// Evaluate
//      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
// over r, which is
//      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
fn build_eq_x_r_for_test<F: Field>(r: &[F]) -> Vec<F> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    // First, we build array for {1 - r_i}
    let one_minus_r: Vec<F> = r.iter().map(|ri| F::one() - ri).collect();

    let num_var = r.len();
    let mut eval = vec![];

    for i in 0..1 << num_var {
        let mut current_eval = F::one();
        let bit_sequence = bit_decompose(i, num_var);

        for (&bit, (ri, one_minus_ri)) in bit_sequence.iter().zip(r.iter().zip(one_minus_r.iter()))
        {
            current_eval *= if bit { *ri } else { *one_minus_ri };
        }
        eval.push(current_eval);
    }

    eval
}

/// Decompose an integer into a binary vector in little endian.
fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

#[test]
fn test_univariate_poly_evaluation() {
    let mut rng = test_rng();

    let po2 = 1024;
    let bits = 10;
    let point = Fr::random_unsafe(&mut rng);

    let univariate = UnivariatePoly::random(po2 - 1, &mut rng);
    let evaluation = univariate.evaluate(point);

    let lagrange = univariate.clone().fft();
    let another_evaluatopm = lagrange.evaluate(point);

    assert_eq!(another_evaluatopm, evaluation);

    // NOTE(HS) now we test point being on the smooth multiplicative subgroup
    let omega = Fr::two_adic_generator(bits);
    let omega_i = omega.exp(893_u128);

    let evaluation = univariate.evaluate(omega_i);
    let another_evaluation = lagrange.evaluate(omega_i);

    assert_eq!(another_evaluation, evaluation);
}

#[test]
fn test_univariate_degree_one_quotient() {
    {
        // x^3 + 1 = (x + 1)(x^2 - x + 1)
        let mut poly = UnivariatePoly::new(vec![Fr::ONE, Fr::ZERO, Fr::ZERO, Fr::ONE]);
        let point = -Fr::from(1u64);
        poly.root_vanishing_quotient(&[point]);

        assert_eq!(poly.coeffs, vec![Fr::ONE, -Fr::ONE, Fr::ONE, Fr::ZERO]);
    }
    {
        // x^3 - 1 = (x-1)(x^2 + x + 1)
        let poly = UnivariatePoly::new(vec![-Fr::ONE, Fr::ZERO, Fr::ZERO, Fr::ONE]);
        let point = Fr::from(1u64);
        let (div, remainder) = poly.degree_one_quotient(point);
        assert_eq!(div.coeffs, vec![Fr::ONE, Fr::ONE, Fr::ONE, Fr::ZERO]);
        assert_eq!(remainder, Fr::ZERO)
    }
    {
        // x^3 + 6x^2 + 11x + 6 = (x + 1)(x + 2)(x + 3)
        let poly = UnivariatePoly::new(vec![
            Fr::from(6u64),
            Fr::from(11u64),
            Fr::from(6u64),
            Fr::from(1u64),
        ]);
        let point = Fr::from(1u64);
        let (div, remainder) = poly.degree_one_quotient(point);
        assert_eq!(
            div.coeffs,
            vec![
                Fr::from(18u64),
                Fr::from(7u64),
                Fr::from(1u64),
                Fr::from(0u64),
            ]
        );
        assert_eq!(remainder, Fr::from(24u64))
    }
    {
        // x^3 + 6x^2 + 11x + 6 = (x + 1)(x + 2)(x + 3)
        let mut poly = UnivariatePoly::new(vec![
            Fr::from(6u64),
            Fr::from(11u64),
            Fr::from(6u64),
            Fr::from(1u64),
        ]);
        poly.root_vanishing_quotient(&[-Fr::ONE, -Fr::from(2u64), -Fr::from(3u64)]);
        assert_eq!(poly.coeffs, vec![Fr::ONE, Fr::ZERO, Fr::ZERO, Fr::ZERO]);
    }
}
