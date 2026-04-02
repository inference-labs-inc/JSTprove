// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, Field, field::FieldOps};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
    multilinear::{eq::eq_ind, evaluate::evaluate_inplace_scalars},
    univariate::evaluate_univariate,
};
use itertools::{Itertools, izip};

use super::{
    common::{
        IntMulOutput, Phase1Output, Phase2Output, Phase3Output, Phase4Output, Phase5Output,
        frobenius_twist, make_phase_3_output, normalize_a_c_exponent_evals,
    },
    error::Error,
};
use crate::protocols::{
    prodcheck::{self, MultilinearEvalClaim},
    sumcheck::{BatchSumcheckOutput, batch_verify},
};

/// Verify one layer of a batched bivariate product MLE tree.
///
/// Given evaluations of product MLEs at a shared point, runs a sumcheck reducing them to
/// evaluations of the left/right factor MLEs at a new random point. Returns the new evaluation
/// point (challenges) and the claimed factor evaluations.
#[allow(clippy::type_complexity)]
fn verify_multi_bivariate_product_mle_layer<F, C>(
    eval_point: &[C::Elem],
    evals: &[C::Elem],
    channel: &mut C,
) -> Result<(Vec<C::Elem>, Vec<C::Elem>), Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    let n_vars = eval_point.len();

    let BatchSumcheckOutput {
        batch_coeff,
        mut challenges,
        eval,
    } = batch_verify(n_vars, 3, evals, channel)?;

    challenges.reverse();

    let multilinear_evals = channel.recv_many(2 * evals.len())?;

    let eq_ind_eval = eq_ind(eval_point, &challenges);
    let expected_unbatched_terms = multilinear_evals
        .iter()
        .tuples()
        .map(|(left, right)| eq_ind_eval.clone() * left * right)
        .collect::<Vec<_>>();

    let expected_eval = evaluate_univariate(&expected_unbatched_terms, batch_coeff);
    channel.assert_zero(expected_eval - eval)?;

    Ok((challenges, multilinear_evals))
}

/// Verify Phase 1: GKR step on the exponentiation product tree.
///
/// Runs prodcheck verification to reduce the root claim on $\widetilde{Q}$ to $2^k$ leaf
/// evaluation claims, then verifies the leaf evaluations against the prover's claimed values.
fn verify_phase_1<F, C>(
    log_bits: usize,
    initial_eval_point: &[C::Elem],
    initial_b_eval: C::Elem,
    channel: &mut C,
) -> Result<Phase1Output<C::Elem>, Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    let n_vars = initial_eval_point.len();

    // Run prodcheck verification
    let claim = MultilinearEvalClaim {
        eval: initial_b_eval,
        point: initial_eval_point.to_vec(),
    };
    let output_claim = prodcheck::verify(log_bits, claim, channel)?;

    // Split output point: first n are x-point, last k are z-challenges
    let (eval_point, z_suffix) = output_claim.point.split_at(n_vars);

    // Read 2^k leaf evaluations from channel
    let b_leaves_evals = channel.recv_many(1 << log_bits)?;

    // Verify: output_claim.eval = multilinear_eval(b_leaves_evals, z_suffix)
    // The leaf evals form a multilinear over log_bits variables; evaluate at z_suffix
    let expected_eval = evaluate_inplace_scalars(b_leaves_evals.clone(), z_suffix);

    channel.assert_zero(expected_eval - output_claim.eval)?;

    Ok(Phase1Output {
        eval_point: eval_point.to_vec(),
        b_leaves_evals,
    })
}

/// Verify Phase 3: batched Frobenius selector sumcheck and LO * HI product sumcheck.
///
/// Batches two sumchecks: (a) the Frobenius-twisted selector sumcheck reducing the Phase 2
/// claims to exponent evaluations on $\widetilde{b}$ and a selector on $\widetilde{P}$, and
/// (b) the product claim $\widetilde{\textsf{LO}} \cdot \widetilde{\textsf{HI}}$.
fn verify_phase_3<F, C>(
    log_bits: usize,
    twisted_eval_points: Vec<Vec<C::Elem>>,
    twisted_evals: Vec<C::Elem>,
    c_eval_point: &[C::Elem],
    c_eval: C::Elem,
    channel: &mut C,
) -> Result<Phase3Output<C::Elem>, Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    let n_vars = c_eval_point.len();

    assert_eq!(twisted_eval_points.len(), 1 << log_bits);

    for twisted_eval_point in &twisted_eval_points {
        assert_eq!(twisted_eval_point.len(), c_eval_point.len());
    }

    let evals = iter::chain(twisted_evals, [c_eval]).collect::<Vec<_>>();

    let BatchSumcheckOutput {
        batch_coeff,
        mut challenges,
        eval,
    } = batch_verify(n_vars, 3, &evals, channel)?;
    challenges.reverse();

    let selector_prover_evals = channel.recv_many((1 << log_bits) + 1)?;
    let c_root_prover_evals = channel.recv_many(2)?;

    let output = make_phase_3_output(
        log_bits,
        &challenges,
        &selector_prover_evals,
        c_root_prover_evals,
    );
    let Phase3Output {
        eval_point,
        b_exponent_evals,
        selector_eval,
        c_lo_root_eval,
        c_hi_root_eval,
    } = &output;

    let mut expected_unbatched_terms = Vec::with_capacity((1 << log_bits) + 1);

    for (twisted_eval_point, b_exponent_eval) in izip!(twisted_eval_points, b_exponent_evals) {
        let twisted_eq_eval = eq_ind(&twisted_eval_point, eval_point);
        let one = C::Elem::one();
        let expected = twisted_eq_eval
            * (b_exponent_eval.clone() * (selector_eval.clone() - one.clone()) + one);
        expected_unbatched_terms.push(expected);
    }

    let c_eq_eval = eq_ind(c_eval_point, eval_point);
    expected_unbatched_terms.extend([c_eq_eval * c_lo_root_eval * c_hi_root_eval]);

    let expected_batched_eval = evaluate_univariate(&expected_unbatched_terms, batch_coeff);

    channel.assert_zero(expected_batched_eval - eval)?;

    Ok(output)
}

/// Verify Phase 4: all but last layer of the GKR product trees for $\widetilde{a}$,
/// $\widetilde{c}_{\textsf{lo}}$, and $\widetilde{c}_{\textsf{hi}}$.
///
/// Iteratively applies batched bivariate product sumchecks, doubling the number of leaf
/// evaluations at each layer, reducing root claims to leaf claims at depth `log_bits - 1`.
fn verify_phase_4<F, C>(
    log_bits: usize,
    eval_point: &[C::Elem],
    a_root_eval: C::Elem,
    c_lo_root_eval: C::Elem,
    c_hi_root_eval: C::Elem,
    channel: &mut C,
) -> Result<Phase4Output<C::Elem>, Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    assert!(log_bits >= 1);

    let mut eval_point = eval_point.to_vec();
    let mut evals = vec![a_root_eval, c_lo_root_eval, c_hi_root_eval];

    for depth in 0..log_bits - 1 {
        assert_eq!(evals.len(), 3 << depth);

        let (challenges, multilinear_evals) =
            verify_multi_bivariate_product_mle_layer(&eval_point, &evals, channel)?;

        eval_point = challenges;
        evals = multilinear_evals;
    }

    assert_eq!(evals.len(), 3 << (log_bits - 1));
    let c_hi_evals = evals.split_off(2 << (log_bits - 1));
    let c_lo_evals = evals.split_off(1 << (log_bits - 1));
    let a_evals = evals;

    Ok(Phase4Output {
        eval_point,
        a_evals,
        c_lo_evals,
        c_hi_evals,
    })
}

/// Verify Phase 5: final GKR layer, $\widetilde{b}$ rerandomization, and parity zerocheck.
///
/// Batches three sumchecks: (a) the final (widest) bivariate product layer for $\widetilde{a}$,
/// $\widetilde{c}_{\textsf{lo}}$, $\widetilde{c}_{\textsf{hi}}$, (b) a rerandomization sumcheck
/// on the $\widetilde{b}$ exponent evaluations, and (c) a zerocheck verifying $a_0 \cdot b_0 =
/// c_{\textsf{lo},0}$.
#[allow(clippy::too_many_arguments)]
fn verify_phase_5<F, C>(
    log_bits: usize,
    a_c_eval_point: &[F],
    a_evals: &[F],
    c_lo_evals: &[F],
    c_hi_evals: &[F],
    b_eval_point: &[F],
    b_exponent_evals: &[F],
    channel: &mut C,
) -> Result<Phase5Output<F>, Error>
where
    F: Field,
    C: IPVerifierChannel<F, Elem = F>,
{
    assert!(log_bits >= 1);
    assert_eq!(2 * a_evals.len(), 1 << log_bits);
    assert_eq!(2 * c_lo_evals.len(), 1 << log_bits);
    assert_eq!(2 * c_hi_evals.len(), 1 << log_bits);

    let n_vars = a_c_eval_point.len();
    assert_eq!(b_eval_point.len(), n_vars);

    // This is the eval of `a_0 * b_0` and `c_lo_0`.
    let overflow_zerocheck_eval = channel.recv_one()?;

    // Evals for the batched sumcheck: a (2^(k-1)), c_lo (2^(k-1)), c_hi (2^(k-1)) from the
    // bivariate product layer, then a_0*b_0 and c_lo_0 for the parity zerocheck, then b
    // exponent evals (2^k) for the rerandomization sumcheck.
    let evals = [
        a_evals,
        c_lo_evals,
        c_hi_evals,
        &[overflow_zerocheck_eval],
        &[overflow_zerocheck_eval],
        b_exponent_evals,
    ]
    .concat();

    let BatchSumcheckOutput {
        batch_coeff,
        mut challenges,
        eval,
    } = batch_verify(n_vars, 3, &evals, channel)?;
    challenges.reverse();

    // Read the evals of all multilinears in the bivariate product sumcheck: 2^k for `a`, 2^(k+1)
    // for `c`, 2 for `a_0` and `b_0`.
    let mut bivariate_evals: Vec<F> = channel.recv_many((3 << log_bits) + 2)?;
    // Read the single eval of the `c_lo_0` rerand sumcheck.
    let c_lo_0_eval = channel.recv_one()?;
    // Read the 2^k evals of the `b` rerand sumcheck.
    let b_exponent_evals: Vec<F> = channel.recv_many(1 << log_bits)?;

    // Compose the expected evaluation of the batched composition via
    // the prover's claimed multilinear evals extracted above.
    // For every pair (p,q) of multilinears, the verifier can be sure that
    // the MLE of p*q at `a_c_eq_eval` equals the corresponding eval in `evals`.
    // The last of these pairs implies the MLE of `a_0 * b_0` at `a_c_eq_eval` equals
    // `overflow_zerocheck_eval`.
    let a_c_eq_eval = eq_ind(a_c_eval_point, &challenges);
    let expected_bivariate_unbatched_evals = bivariate_evals
        .iter()
        .tuples()
        .map(|(left, right)| a_c_eq_eval * left * right)
        .collect::<Vec<F>>();

    // Likewise, the verifier can be sure that the MLE of `c_lo_0` at `a_c_eq_eval`
    // equals `overflow_zerocheck_eval`. Combined with the MLE of `a_0 * b_0` at `a_c_eq_eval`
    // being `overflow_zerocheck_eval`, the verifier can conclude the
    // MLE of `a_0 * b_0 - c_lo_0` at `a_c_eq_eval` equals zero. By the Schwartz-Zippel lemma,
    // the verifier concludes `a_0_i * b_0_i - c_lo_0_i = 0` for all rows `i`.
    let expected_c_lo_0_rerand_unbatched_eval = a_c_eq_eval * c_lo_0_eval;

    let b_eq_eval = eq_ind(b_eval_point, &challenges);
    let expected_b_rerand_unbatched_evals = b_exponent_evals
        .iter()
        .map(|&b_exponent_eval| b_eq_eval * b_exponent_eval)
        .collect::<Vec<F>>();

    let expected_unbatched_evals = [
        expected_bivariate_unbatched_evals,
        vec![expected_c_lo_0_rerand_unbatched_eval],
        expected_b_rerand_unbatched_evals,
    ]
    .concat();
    let expected_batched_eval = evaluate_univariate(&expected_unbatched_evals, batch_coeff);

    // Compare expected evaluation against given evaluation `eval`.
    channel.assert_zero(expected_batched_eval - eval)?;

    // Evals `b_0_eval`, `a_0_eval`, and `c_lo_0_eval` will be verified following phase 5.
    let b_0_eval = bivariate_evals
        .pop()
        .expect("non-empty scaled a_c exponent evals");
    let a_0_eval = bivariate_evals
        .pop()
        .expect("non-empty scaled a_c exponent evals");

    Ok(Phase5Output {
        eval_point: challenges,
        scaled_a_c_exponent_evals: bivariate_evals,
        b_exponent_evals,
        a_0_eval,
        b_0_eval,
        c_lo_0_eval,
    })
}

/// Verify the integer multiplication check (IntMul) protocol.
///
/// The IntMul protocol is a reduction that checks a relation on four virtual multilinear
/// polynomials: $\widetilde{a}, \widetilde{b}, \widetilde{c}_{\textsf{lo}},
/// \widetilde{c}_{\textsf{hi}}$. These multilinear polynomials are over $\mathbb{F}_2$ and have
/// $k + n$ variables. We write $a, b, c_{\textsf{lo}}, c_{\textsf{hi}} \in \mathbb{F}_2^{n \times
/// k}$ for their boolean hypercube evaluations. Let $\textsf{int}(M) \in \mathbb{N}^n$ map one of
/// the four matrices, $M$, to a vector of their interpretations as a $k$-bit unsigned integer. That
/// is, it embeds the $\mathbb{F}_2$ elements into $\mathbb{N}$ and multiplies by $(2^0, 2^1,
/// \ldots, 2^{k-1})$.
///
/// ## Protocol
///
/// The IntMul protocol reduces this relation to claims on the partial multilinear evaluations of
/// $\widetilde{a}, \widetilde{b}, \widetilde{c}_{\textsf{lo}}, \widetilde{c}_{\textsf{hi}}$ at a
/// common $n$-coordinate random evaluation point.
///
/// ### Exponentiation identity
///
/// The core technique reduces integer multiplication to field arithmetic via exponentiation. Let
/// $g$ be a generator of the multiplicative group of $\mathbb{F}_{2^{2k}}$, which has order
/// $2^{2k} - 1$. Then $\textsf{int}(a) \cdot \textsf{int}(b) = \textsf{int}(c_{\textsf{hi}})
/// \cdot 2^k + \textsf{int}(c_{\textsf{lo}})$ over the integers is equivalent to
///
/// $$\widetilde{Q}(x) = \widetilde{\textsf{LO}}(x) \cdot \widetilde{\textsf{HI}}(x) \quad
/// \forall x \in \{0, 1\}^n$$
///
/// where $\widetilde{Q}$ is obtained by exponentiating $g^{\widetilde{a}}$ by $\widetilde{b}$,
/// $\widetilde{\textsf{LO}} = g^{\widetilde{c}_{\textsf{lo}}}$, and $\widetilde{\textsf{HI}} =
/// (g^{2^k})^{\widetilde{c}_{\textsf{hi}}}$.
///
/// There is a wraparound edge case: when $a \cdot b = 0$, a malicious prover could set
/// $c_{\textsf{hi}} \| c_{\textsf{lo}} = 2^{2k} - 1$, which satisfies the exponentiation
/// identity modulo $2^{2k} - 1$ but not over the integers. A parity check on the least
/// significant bits ($a_0 \cdot b_0 = c_{\textsf{lo},0}$) rules this out.
///
/// ### Phases
///
/// - **Phase 1 — GKR step on $\widetilde{Q}$:** The verifier samples a random evaluation point $r$
///   and the prover sends the claimed evaluation $s = \widetilde{Q}(r)$. The parties run a GKR step
///   ($k$-layer balanced binary tree of bivariate products) reducing $s$ to $2^k$ leaf claims
///   $s'_{Q,i} = \widetilde{Q_i}(r')$.
///
/// - **Phase 2 — Frobenius step:** The verifier applies $\varphi^{-i}$ (inverse Frobenius) to each
///   leaf claim, reducing degree-$2^i$ expressions to degree-1. This is a local verifier
///   computation with no interaction.
///
/// - **Phase 3 — Batched Frobenius sumcheck + $\widetilde{\textsf{LO}} \cdot
///   \widetilde{\textsf{HI}}$ product sumcheck:** Two sumchecks batched into one: (a) The
///   Frobenius-twisted selector sumcheck on the $\widetilde{Q_i}$ claims, reducing to claims on
///   $\widetilde{b}$ exponent evaluations and the base $\widetilde{P}$ (i.e. $g^{\widetilde{a}}$).
///   (b) The deferred product claim $s = \sum \textsf{eq}(r, x) \cdot \widetilde{\textsf{LO}}(x)
///   \cdot \widetilde{\textsf{HI}}(x)$. This yields root claims on $\widetilde{P}$ (the
///   $\widetilde{a}$ selector), $\widetilde{\textsf{LO}}$, $\widetilde{\textsf{HI}}$, plus $2^k$
///   exponent claims on $\widetilde{b}$.
///
/// - **Phase 4 — GKR on $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
///   $\widetilde{c}_{\textsf{hi}}$ (all but last layer):** Batched GKR layers for the three
///   remaining exponentiation product trees. Each layer is a batched bivariate product sumcheck.
///   Since the bases ($g$ and $g^{2^k}$) are fixed, the Frobenius steps can be skipped — the
///   verifier locally reduces "scaled" evaluations to plain exponent evaluations.
///
/// - **Phase 5 — Final GKR layer + $\widetilde{b}$ rerandomization + parity check:** The final
///   (widest) GKR layer for $\widetilde{a}$, $\widetilde{c}_{\textsf{lo}}$,
///   $\widetilde{c}_{\textsf{hi}}$ is batched with: (a) A rerandomization sumcheck on the
///   $\widetilde{b}$ exponent evaluations from Phase 3, bringing them to the same evaluation point
///   as $\widetilde{a}$ and $\widetilde{c}$. (b) A zerocheck verifying $a_0 \cdot b_0 =
///   c_{\textsf{lo},0}$ (least significant bits), ruling out the wraparound edge case.
///
/// ### Output
///
/// The protocol outputs evaluation claims on $\widetilde{a}_i$, $\widetilde{b}_i$,
/// $\widetilde{c}_{\textsf{lo},i}$, $\widetilde{c}_{\textsf{hi},i}$ (for $i \in \{0, \ldots,
/// 2^k - 1\}$) at a common $n$-dimensional evaluation point. These are passed to the shift
/// reduction.
///
/// ### Parameters
///
/// - `log_bits`: $k$, where $2^k$ is the bit-width of the integer operands.
/// - `n_vars`: Number of variables in the row dimension (i.e., $\log_2$ of the number of
///   multiplication constraints).
pub fn verify<F, C>(
    log_bits: usize,
    n_vars: usize,
    channel: &mut C,
) -> Result<IntMulOutput<F>, Error>
where
    F: BinaryField,
    C: IPVerifierChannel<F, Elem = F>,
{
    assert!(log_bits >= 1);
    assert!((1 << (log_bits + 1)) <= F::N_BITS);

    let initial_eval_point: Vec<F> = channel.sample_many(n_vars);

    // Read the evaluation of the multilinear extension of the powers of the generator.
    let exp_eval: F = channel.recv_one()?;

    // Phase 1
    let Phase1Output {
        eval_point: phase_1_eval_point,
        b_leaves_evals,
    } = verify_phase_1(log_bits, &initial_eval_point, exp_eval, channel)?;

    assert_eq!(phase_1_eval_point.len(), n_vars);
    assert_eq!(b_leaves_evals.len(), 1 << log_bits);

    // Phase 2
    let Phase2Output {
        twisted_eval_points,
        twisted_evals,
    } = frobenius_twist(log_bits, &phase_1_eval_point, &b_leaves_evals);

    // Phase 3
    let Phase3Output {
        eval_point: phase_3_eval_point,
        b_exponent_evals,
        selector_eval,
        c_lo_root_eval,
        c_hi_root_eval,
    } = verify_phase_3(
        log_bits,
        twisted_eval_points,
        twisted_evals,
        &initial_eval_point,
        exp_eval,
        channel,
    )?;

    // Phase 4
    let Phase4Output {
        eval_point: phase_4_eval_point,
        a_evals,
        c_lo_evals,
        c_hi_evals,
    } = verify_phase_4(
        log_bits,
        &phase_3_eval_point,
        selector_eval,
        c_lo_root_eval,
        c_hi_root_eval,
        channel,
    )?;

    // Phase 5
    let Phase5Output {
        eval_point: phase_5_eval_point,
        scaled_a_c_exponent_evals,
        b_exponent_evals,
        a_0_eval,
        b_0_eval,
        c_lo_0_eval,
    } = verify_phase_5(
        log_bits,
        &phase_4_eval_point,
        &a_evals,
        &c_lo_evals,
        &c_hi_evals,
        &phase_3_eval_point,
        &b_exponent_evals,
        channel,
    )?;

    let [a_exponent_evals, c_lo_exponent_evals, c_hi_exponent_evals] =
        normalize_a_c_exponent_evals(log_bits, scaled_a_c_exponent_evals);

    assert_eq!(a_exponent_evals[0], a_0_eval);
    assert_eq!(b_exponent_evals[0], b_0_eval);
    assert_eq!(c_lo_exponent_evals[0], c_lo_0_eval);

    Ok(IntMulOutput {
        eval_point: phase_5_eval_point,
        a_evals: a_exponent_evals,
        b_evals: b_exponent_evals,
        c_lo_evals: c_lo_exponent_evals,
        c_hi_evals: c_hi_exponent_evals,
    })
}
