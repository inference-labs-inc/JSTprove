// Copyright 2025 Irreducible Inc.
//! Prover implementation for the sumcheck protocol.
//!
//! This module provides functionality for executing the sumcheck proving protocol,
//! which allows a prover to convince a verifier that a claimed sum over a multivariate
//! polynomial is correct through an interactive proof protocol.

use binius_field::Field;
use binius_ip::mlecheck;

use super::{common::SumcheckProver, error::Error};
use crate::{channel::IPProverChannel, sumcheck::common::MleCheckProver};

/// Executes the sumcheck proving protocol for a single multivariate polynomial.
///
/// This function drives the interactive sumcheck protocol, where the prover convinces
/// a verifier that a claimed sum over a multivariate polynomial is correct. The protocol
/// proceeds in rounds, with one round per variable in the polynomial.
///
/// # Arguments
///
/// * `prover` - An implementation of [`SumcheckProver`] that computes the polynomial evaluations
///   for each round. The prover must evaluate exactly one composition polynomial per round.
/// * `channel` - The channel for sending prover messages and sampling challenges.
///
/// # Returns
///
/// Returns [`ProveSingleOutput`] containing:
/// - `multilinear_evals`: Final evaluations of the multilinear polynomials at the challenge point
/// - `challenges`: The verifier challenges used in each round
///
/// # Errors
///
/// - [`Error::ArgumentError`] if the prover returns more than one composition polynomial from its
///   `execute()` method
/// - Any error propagated from the underlying prover implementation
///
/// # Protocol Flow
///
/// For each of the `n_vars` rounds:
/// 1. The prover computes univariate polynomial coefficients via `execute()`
/// 2. These coefficients are written to the channel
/// 3. A challenge is sampled from the channel
/// 4. The prover folds the polynomial with this challenge via `fold()`
///
/// After all rounds, `finish()` is called to obtain the final multilinear evaluations.
pub fn prove_single<F: Field>(
    mut prover: impl SumcheckProver<F>,
    channel: &mut impl IPProverChannel<F>,
) -> Result<ProveSingleOutput<F>, Error> {
    let n_vars = prover.n_vars();
    let mut challenges = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        let mut round_coeffs_vec = prover.execute()?;
        if round_coeffs_vec.len() != 1 {
            return Err(Error::ArgumentError(format!(
                "function expects prover to evaluate one composition, but it returned {} from \
				execute()",
                round_coeffs_vec.len()
            )));
        }
        let round_coeffs = round_coeffs_vec.pop().expect("round_coeffs_vec.len() == 1");

        channel.send_many(round_coeffs.truncate().coeffs());

        let challenge = channel.sample();
        challenges.push(challenge);
        prover.fold(challenge)?;
    }

    let multilinear_evals = prover.finish()?;
    Ok(ProveSingleOutput {
        multilinear_evals,
        challenges,
    })
}

/// Executes the MLE-check proving protocol for a single multivariate polynomial.
///
/// Analogous to [`prove_single`] for the MLE-check protocol instead of sumcheck.
pub fn prove_single_mlecheck<F: Field>(
    mut prover: impl MleCheckProver<F>,
    channel: &mut impl IPProverChannel<F>,
) -> Result<ProveSingleOutput<F>, Error> {
    let n_vars = prover.n_vars();
    let mut challenges = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        let mut round_coeffs_vec = prover.execute()?;
        if round_coeffs_vec.len() != 1 {
            return Err(Error::ArgumentError(format!(
                "function expects prover to evaluate one composition, but it returned {} from \
				execute()",
                round_coeffs_vec.len()
            )));
        }
        let round_coeffs = round_coeffs_vec.pop().expect("round_coeffs_vec.len() == 1");

        channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());

        let challenge = channel.sample();
        challenges.push(challenge);
        prover.fold(challenge)?;
    }

    let multilinear_evals = prover.finish()?;
    Ok(ProveSingleOutput {
        multilinear_evals,
        challenges,
    })
}

/// Output of the sumcheck proving protocol for a single multivariate polynomial.
///
/// Contains the final evaluations and challenges generated during the interactive
/// protocol execution.
pub struct ProveSingleOutput<F: Field> {
    /// Evaluations of the multilinear polynomials at the challenge point.
    ///
    /// After the sumcheck protocol completes, these are the values of each multilinear
    /// polynomial evaluated at the point formed by all verifier challenges.
    pub multilinear_evals: Vec<F>,
    /// Verifier challenges for each round of the sumcheck protocol.
    ///
    /// One challenge is generated per variable in the multivariate polynomial,
    /// with challenges\[i\] corresponding to the i-th round of the protocol.
    /// NB: reverse when folding high-to-low to obtain evaluation claim.
    pub challenges: Vec<F>,
}
