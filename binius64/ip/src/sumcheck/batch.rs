// Copyright 2025 Irreducible Inc.

use binius_field::Field;
use binius_math::univariate::evaluate_univariate;

use crate::{
    channel::IPVerifierChannel,
    mlecheck,
    sumcheck::{self, Error, SumcheckOutput},
};

/// The reduced output of a sumcheck verification.
///
/// The [`batch_verify`] function reduces a set of claims on multivariate polynomials over the
/// boolean hypercube to their evaluation at a challenge point. See the function docstring for
/// details.
pub struct BatchSumcheckOutput<F> {
    /// The challenge value of the batching variable.
    pub batch_coeff: F,
    /// The evaluation of the sumcheck multivariate at the challenge point.
    pub eval: F,
    /// Verifier challenges for each round of the sumcheck protocol.
    ///
    /// One challenge is generated per variable in the multivariate polynomial,
    /// with challenges\[i\] corresponding to the i-th round of the protocol.
    ///
    /// Note: reverse when folding high-to-low to obtain evaluation claim.
    pub challenges: Vec<F>,
}

/// Verify a batched sumcheck protocol interaction.
///
/// The batched sumcheck verifier reduces a set of claims about the sums of multivariate polynomials
/// over the boolean hypercube to their evaluation at a (shared) challenge point. This is achieved
/// by constructing an `n_vars + 1`-variate polynomial whose coefficients in the "new variable" are
/// the individual sum claims and evaluating it at a random point.
pub fn batch_verify<F, C>(
    n_vars: usize,
    degree: usize,
    sums: &[C::Elem],
    channel: &mut C,
) -> Result<BatchSumcheckOutput<C::Elem>, Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    // Random linear-combination coefficient that binds all sum claims together.
    let batch_coeff = channel.sample();
    // Combine the individual sum claims into a single scalar for sumcheck verification.
    let sum = evaluate_univariate(sums, batch_coeff.clone());

    let SumcheckOutput { eval, challenges } =
        sumcheck::verify::<F, C>(n_vars, degree, sum, channel)?;

    Ok(BatchSumcheckOutput {
        batch_coeff,
        challenges,
        eval,
    })
}

/// Verify a batched sumcheck protocol interaction for MLE-checks.
///
/// This is the MLE-check analog of [`batch_verify`]: it batches evaluation claims from multiple
/// MLE-check instances that share a common evaluation point, using a single batching coefficient
/// and shared verifier challenges to reduce all claims to one scalar verification.
pub fn batch_verify_mle<F, C>(
    point: &[C::Elem],
    degree: usize,
    evals: &[C::Elem],
    channel: &mut C,
) -> Result<BatchSumcheckOutput<C::Elem>, Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    // Random linear-combination coefficient that binds all eval claims together.
    let batch_coeff = channel.sample();
    // Combine the individual eval claims into a single scalar for MLE-check verification.
    let eval = evaluate_univariate(evals, batch_coeff.clone());

    let SumcheckOutput { eval, challenges } =
        mlecheck::verify::<F, C>(point, degree, eval, channel)?;

    Ok(BatchSumcheckOutput {
        batch_coeff,
        challenges,
        eval,
    })
}
