// Copyright 2025 Irreducible Inc.

use binius_field::Field;

use super::error::Error;
use crate::{
    channel::IPVerifierChannel,
    sumcheck::{RoundCoeffs, RoundProof},
};

/// The reduced output of a sumcheck verification.
///
/// The [`verify`] function reduces a claim about the sum of a multivariate polynomial over the
/// boolean hypercube to its evaluation at a challenge point.
#[derive(Debug, Clone, PartialEq)]
pub struct SumcheckOutput<F> {
    /// The evaluation of the sumcheck multivariate at the challenge point.
    pub eval: F,
    /// The sequence of sumcheck challenges defining the evaluation point.
    pub challenges: Vec<F>,
}

/// Verify a sumcheck protocol interaction.
///
/// The sumcheck verifier reduces a claim about the sum of a multivariate polynomial over the
/// boolean hypercube to its evaluation at a challenge point.
///
/// ## Arguments
///
/// * `n_vars` - The number of variables in the multivariate polynomial
/// * `degree` - The degree of the univariate polynomial in each round
/// * `sum` - The claimed sum of the multivariate polynomial over the boolean hypercube
/// * `channel` - The channel for receiving prover messages and sampling challenges
///
/// ## Returns
///
/// Returns a `Result` containing the `SumcheckOutput` with the reduced evaluation and challenge
/// point, or an error if verification fails.
pub fn verify<F, C>(
    n_vars: usize,
    degree: usize,
    mut sum: C::Elem,
    channel: &mut C,
) -> Result<SumcheckOutput<C::Elem>, Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    let mut challenges = Vec::with_capacity(n_vars);
    for _round in 0..n_vars {
        let round_proof = RoundProof(RoundCoeffs(channel.recv_many(degree)?));
        let challenge = channel.sample();

        let round_coeffs = round_proof.recover(sum);
        sum = round_coeffs.evaluate(challenge.clone());
        challenges.push(challenge);
    }

    Ok(SumcheckOutput {
        eval: sum,
        challenges,
    })
}
