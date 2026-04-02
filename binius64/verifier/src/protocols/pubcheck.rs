// Copyright 2025 Irreducible Inc.
use std::iter;

use binius_field::Field;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};

use crate::{
    error::Error,
    protocols::{mlecheck, sumcheck::SumcheckOutput},
};

/// Output of [`verify`].
#[derive(Debug)]
pub struct VerifyOutput<F: Field> {
    /// Evaluation of the (w - p) polynomial at `eval_point`.
    pub eval: F,
    /// Evaluation point from the MLE-check.
    pub eval_point: Vec<F>,
}

/// Verify the public input check (pubcheck) protocol.
///
/// The pubcheck protocol argues that the witness multilinear agrees with the public input
/// multilinear on a subdomain. The witness $w$ is $\ell$-variate, and the public multilinear $p$
/// is $m$-variate, where $m \le \ell$. The interactive reduction argues that for all $v \in B_m$
///
/// $$
/// w(v_0, \ldots, v_{m-1}, 0^{\ell - m}) = p(v_0, \ldots, v_{m-1})
/// $$
///
/// The protocol is an MLE-check on the multilinear $w$, using a zero-padded challenge point. It
/// begins with an $m$-dimensional challenge point $r$ and reduces to an MLE-check that
/// $w(r || 0) = p(r)$.
///
/// ## Arguments
///
/// * `n_witness_vars` - base-2 logarithm of the number of witness words
/// * `challenge` - the $m$-dimensional challenge point
/// * `transcript` - the verifier's transcript
///
/// ## Preconditions
///
/// * `challenge.len()` is at most `n_witness_vars`
pub fn verify<F: Field, Challenger_: Challenger>(
    n_witness_vars: usize,
    public_eval: F,
    challenge: &[F],
    transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<VerifyOutput<F>, Error> {
    let n_public_vars = challenge.len();
    assert!(n_public_vars <= n_witness_vars); // precondition

    // The MLE-check verifier checks an evaluation at the zero-padded point.
    let zero_padded_eval_point = itertools::chain(challenge.iter().copied(), iter::repeat(F::ZERO))
        .take(n_witness_vars)
        .collect::<Vec<_>>();

    let SumcheckOutput {
        eval,
        mut challenges,
    } = mlecheck::verify(
        &zero_padded_eval_point,
        1, // degree 1 for w multilinear
        public_eval,
        transcript,
    )?;

    // MLE-check expects prover to bind variables high-to-low, so reverse challenge order.
    challenges.reverse();

    Ok(VerifyOutput {
        eval,
        eval_point: challenges,
    })
}
