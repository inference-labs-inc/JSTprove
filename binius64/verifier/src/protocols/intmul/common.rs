// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, field::FieldOps};
use itertools::{iterate, izip};

#[derive(Debug, Clone, PartialEq)]
pub struct IntMulOutput<F> {
    pub eval_point: Vec<F>,
    pub a_evals: Vec<F>,
    pub b_evals: Vec<F>,
    pub c_lo_evals: Vec<F>,
    pub c_hi_evals: Vec<F>,
}

/// Output of Phase 1: GKR reduction of the exponentiation product tree.
///
/// Contains the evaluation point after prodcheck and the $2^k$ leaf evaluations of
/// $\widetilde{Q_i}$.
pub struct Phase1Output<F> {
    pub eval_point: Vec<F>,
    pub b_leaves_evals: Vec<F>,
}

pub struct Phase2Output<F> {
    pub twisted_eval_points: Vec<Vec<F>>,
    pub twisted_evals: Vec<F>,
}

/// Output of Phase 3: batched Frobenius selector sumcheck and LO * HI product sumcheck.
///
/// Contains the new evaluation point, $\widetilde{b}$ exponent evaluations, the selector
/// evaluation on $\widetilde{P}$, and root evaluations of $\widetilde{\textsf{LO}}$ and
/// $\widetilde{\textsf{HI}}$.
#[derive(Debug, Clone)]
pub struct Phase3Output<F> {
    pub eval_point: Vec<F>,
    pub b_exponent_evals: Vec<F>,
    pub selector_eval: F,
    pub c_lo_root_eval: F,
    pub c_hi_root_eval: F,
}

/// Construct the [`Phase3Output`] from the prover's claimed evaluations.
///
/// Splits the selector prover evals into individual $\widetilde{b}$ exponent evaluations and
/// the selector evaluation, and extracts the two $\widetilde{c}$ root evaluations.
pub fn make_phase_3_output<F: FieldOps>(
    log_bits: usize,
    eval_point: &[F],
    selector_prover_evals: &[F],
    c_root_prover_evals: Vec<F>,
) -> Phase3Output<F> {
    assert_eq!(selector_prover_evals.len(), 1 + (1 << log_bits));
    let (selector_eval, b_exponent_evals) = selector_prover_evals
        .split_last()
        .expect("non-empty selector sumcheck output");

    let Ok([c_lo_root_eval, c_hi_root_eval]) = TryInto::<[F; 2]>::try_into(c_root_prover_evals)
    else {
        unreachable!("expect two multilinears in the c_root prover in phase 3")
    };

    Phase3Output {
        eval_point: eval_point.to_vec(),
        b_exponent_evals: b_exponent_evals.to_vec(),
        selector_eval: selector_eval.clone(),
        c_lo_root_eval,
        c_hi_root_eval,
    }
}

/// Output of Phase 4: all but last GKR layer for $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
/// $\widetilde{c}_{\textsf{hi}}$.
///
/// Contains the evaluation point and leaf evaluations for each of the three product trees at
/// depth `log_bits - 1`.
pub struct Phase4Output<F> {
    pub eval_point: Vec<F>,
    pub a_evals: Vec<F>,
    pub c_lo_evals: Vec<F>,
    pub c_hi_evals: Vec<F>,
}

/// Output of Phase 5: final GKR layer, $\widetilde{b}$ rerandomization, and parity zerocheck.
///
/// Contains the final evaluation point and all leaf-level exponent evaluations for
/// $\widetilde{a}$, $\widetilde{b}$, $\widetilde{c}_{\textsf{lo}}$,
/// $\widetilde{c}_{\textsf{hi}}$, plus the parity-check values $a_0$, $b_0$, $c_{\textsf{lo},0}$.
pub struct Phase5Output<F> {
    pub eval_point: Vec<F>,
    pub scaled_a_c_exponent_evals: Vec<F>,
    pub b_exponent_evals: Vec<F>,
    pub a_0_eval: F,
    pub b_0_eval: F,
    pub c_lo_0_eval: F,
}

/// Compute the inverse Frobenius endomorphism $\varphi^{-i}(x)$.
///
/// The Frobenius endomorphism on $\mathbb{F}_{2^d}$ is $\varphi(x) = x^2$, so $\varphi^i(x) =
/// x^{2^i}$. Its order is $d$ (the extension degree), meaning $\varphi^d = \textsf{id}$.
/// Therefore $\varphi^{-i} = \varphi^{d - i}$, and we compute $\varphi^{-i}(x) = x^{2^{d-i}}$
/// by repeated squaring $d - i$ times.
fn inv_frobenius<F>(x: F, i: usize) -> F
where
    F: FieldOps,
    F::Scalar: BinaryField,
{
    let degree = F::Scalar::N_BITS;
    iterate(x, |g| g.clone().square())
        .nth(degree - i)
        .expect("infinite iterator")
}

/// Compute the inverse Frobenius sequence $[\varphi^{0}(x), \varphi^{-1}(x), \ldots,
/// \varphi^{-(n-1)}(x)]$ where $d$ is the extension degree of $\mathbb{F}_{2^d}$.
fn inv_frobenius_sequence<F>(x: F, n: usize) -> Vec<F>
where
    F: FieldOps,
    F::Scalar: BinaryField,
{
    let degree = F::Scalar::N_BITS;
    assert!(n <= degree + 1);
    let mut seq: Vec<F> = iterate(x, |g| g.clone().square())
        .take(degree + 1)
        .collect();
    seq.reverse();
    seq.truncate(n);
    seq
}

/// Apply inverse Frobenius twists to the leaf evaluation claims from Phase 1.
///
/// This reduces $2^k$ evaluation claims on $2^k$ separate multilinears $\widetilde{Q_i}$ at a
/// shared point $r$ to $2^k$ claims on a single multilinear $\widetilde{P}$ at $2^k$ different
/// points. Concretely, given claims $(r, s_i)$ where $s_i = \widetilde{Q_i}(r)$ and
/// $\widetilde{Q_i}(x) = \widetilde{P}(x)^{2^i}$, this applies $\varphi^{-i}$ (the inverse
/// Frobenius endomorphism) to both the evaluation point and the evaluation value. This linearizes
/// the degree-$2^i$ relation into a degree-1 claim: $\varphi^{-i}(s_i) =
/// \widetilde{P}(\varphi^{-i}(r))$, since $\varphi^{-i}(x^{2^i}) = x$ in $\mathbb{F}_{2^d}$.
///
/// # Arguments
///
/// * `k` - The log of the bit-width; there are $2^k$ leaf claims.
/// * `eval_point` - The shared evaluation point $r$.
/// * `evals` - The $2^k$ evaluations $s_0, \ldots, s_{2^k - 1}$.
pub fn frobenius_twist<F>(k: usize, eval_point: &[F], evals: &[F]) -> Phase2Output<F>
where
    F: FieldOps,
    F::Scalar: BinaryField,
{
    let n = 1 << k;
    assert_eq!(evals.len(), n);

    // Precompute inv_frobenius_sequence for each coordinate in eval_point.
    let coord_seqs: Vec<Vec<F>> = eval_point
        .iter()
        .map(|coord| inv_frobenius_sequence(coord.clone(), n))
        .collect();

    let twisted_eval_points = (0..n)
        .map(|i| coord_seqs.iter().map(|seq| seq[i].clone()).collect())
        .collect();

    let twisted_evals = evals
        .iter()
        .enumerate()
        .map(|(i, eval)| inv_frobenius(eval.clone(), i))
        .collect();

    Phase2Output {
        twisted_eval_points,
        twisted_evals,
    }
}

pub fn normalize_a_c_exponent_evals<F, E>(log_bits: usize, evals: Vec<E>) -> [Vec<E>; 3]
where
    F: BinaryField,
    E: FieldOps<Scalar = F> + From<F>,
{
    assert_eq!(evals.len(), 3 << log_bits);

    // for i in 0..1 << log_bits: evals[i] = (1-EvalMLE_i)*1 + EvalMLE_i*g^{2^i} =
    // EvalMLE_i*(g^{2^i}-1) + 1 where EvalMLE_i is the evaluation of the multilinear extension of
    // bit i of the exponents of `a` (the point of evaluation is irrelevant in this function)
    // we can then compute desired evaluation EvalMLE_i as (evals[i] - 1) / (g^{2^i}-1)
    // similarly for `c` for evals[1 << log_bits..3 << log_bits] and i in 0..2 << log_bits

    let mut a_scaled_evals = evals;
    let mut c_lo_scaled_evals = a_scaled_evals.split_off(1 << log_bits);
    let mut c_hi_scaled_evals = c_lo_scaled_evals.split_off(1 << log_bits);

    // Compute the normalization factors (conjugate - 1)^{-1} in F, then convert to E.
    let inv_factors: Vec<E> = iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square())
        .take(2 << log_bits)
        .map(|conjugate| E::from((conjugate - F::ONE).invert().expect("non-zero")))
        .collect();

    let (lo_inv_factors, hi_inv_factors) = inv_factors.split_at(1 << log_bits);

    fn normalize<E: FieldOps>(eval: &mut E, inv_factor: &E) {
        *eval -= E::one();
        *eval *= inv_factor.clone();
    }

    for (inv_factor, a_eval, c_lo_eval) in
        izip!(lo_inv_factors, &mut a_scaled_evals, &mut c_lo_scaled_evals)
    {
        normalize(a_eval, inv_factor);
        normalize(c_lo_eval, inv_factor);
    }

    for (inv_factor, c_hi_eval) in izip!(hi_inv_factors, &mut c_hi_scaled_evals) {
        normalize(c_hi_eval, inv_factor);
    }

    [a_scaled_evals, c_lo_scaled_evals, c_hi_scaled_evals]
}
