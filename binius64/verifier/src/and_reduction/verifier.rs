// Copyright 2025 Irreducible Inc.

use std::iter::{self};

use binius_field::{BinaryField, field::FieldOps};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{BinarySubspace, univariate::extrapolate_over_subspace};

use crate::{
    Error,
    and_reduction::utils::constants::ROWS_PER_HYPERCUBE_VERTEX,
    protocols::{mlecheck::verify, sumcheck::SumcheckOutput},
};

/// Output from the AND constraint reduction protocol verification.
#[derive(Debug, PartialEq)]
pub struct AndCheckOutput<F> {
    pub a_eval: F,
    pub b_eval: F,
    pub c_eval: F,
    pub z_challenge: F,
    pub eval_point: Vec<F>,
}

/// Verifies the AND constraint reduction protocol via univariate zerocheck.
///
/// Note: Following section 4.4 of the Binius64 writeup, Z is the bit index within a word, and X is
/// the word index
///
/// Let our oblong polynomials be A(Z, X₀, ...), B(Z, X₀, ...), and C(Z, X₀, ...)
///
/// Let our zerocheck challenges be (r₀, ...)
///
/// This protocol reduces the verification of AND constraints (A(Z,X₀,...,Xₙ₋₁)·B(Z,X₀,...,Xₙ₋₁) -
/// C(Z,X₀,...,Xₙ₋₁) = 0) over a multivariate domain to a single multilinear polynomial evaluation.
/// The key insight is that A·B-C = 0 if and only if for all Z, the multilinear extension of A·B-C
/// evaluates to zero at a random point (Z,r₀,...,rₙ₋₁), (up to some negligible error probability).
///
/// Note: This is equivalent to proving |D| multilinear zerochecks at once, all using the same
/// random zerocheck challenges
///
/// ## Phase 1: Univariate Polynomial Verification
///
/// The prover sends a univariate polynomial R₀(Z) that encodes the sum:
///
/// R₀(Z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A(Z,X₀,...,Xₙ₋₁)·B(Z,X₀,...,Xₙ₋₁) -
/// C(Z,X₀,...,Xₙ₋₁))·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
/// where:
/// - A(Z,X₀,...,Xₙ₋₁), B(Z,X₀,...,Xₙ₋₁), C(Z,X₀,...,Xₙ₋₁) are oblong multilinear polynomials
///   representing the AND constraint operands
/// - eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁) is the multilinear equality indicator partially evaluated at a
///   series of random and compile-time pre-known challenges r₀,...,rₙ₋₁ (note: Z is not included in
///   the equality check)
/// - Z ranges over a univariate domain of size 2^(SKIPPED_VARS + 1)
///
/// The equality indicator eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁) = ∏ᵢ₌₀ⁿ⁻¹(Xᵢ·rᵢ + (1-Xᵢ)·(1-rᵢ)) ensures
/// we're checking that the multilinear extension of A·B-C evaluates to zero at the random point
/// (Z, r₀,...,rₙ₋₁) for each Z in the domain.
///
/// The polynomial R₀(Z) has degree at most 2*(|D| - 1) where |D| is the domain size. The prover
/// only sends evaluations on an extension domain (the upper half) since R₀(Z) = 0 on the base
/// domain when all AND constraints are satisfied.
///
/// ## Phase 2: Multilinear Sumcheck Reduction
///
/// After the verifier samples a random challenge z for Z, the protocol continues with a standard
/// sumcheck protocol on the remaining variables X₀,...,Xₙ₋₁ to verify that:
///
/// R₀(z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A(z,X₀,...,Xₙ₋₁)·B(z,X₀,...,Xₙ₋₁) -
/// C(z,X₀,...,Xₙ₋₁))·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
///
/// This reduces to a single evaluation of the folded polynomial at the sumcheck challenge point.
///
/// ## Arguments
///
/// * `n_vars` - The number of variables in the sumcheck protocol (excluding the univariate variable
///   Z)
/// * `transcript` - The verifier's transcript for reading prover messages and sampling challenges
/// * `round_message_univariate_domain` - The univariate domain D for polynomial evaluations
///
/// ## Returns
///
/// Returns `AndCheckOutput` containing:
/// - `z_challenge`: The univariate challenge z sampled for the bit-index variable
/// - `eval_point`: The multilinear evaluation point. Prepened with the `z_challenge` this makes the
///   oblong evaluation point
/// - `a_eval`, `b_eval`, `c_eval`: The claimed evaluations of the A, B, and C at the oblong
///   evaluation point
pub fn verify_with_channel<F, C>(
    all_zerocheck_challenges: &[C::Elem],
    channel: &mut C,
    round_message_univariate_domain: &BinarySubspace<F>,
) -> Result<AndCheckOutput<C::Elem>, Error>
where
    F: BinaryField,
    C: IPVerifierChannel<F>,
    // This bound is necessary to make Barycentric evaluation constants symbolic
    C::Elem: From<F>,
{
    let univariate_message_coeffs_ext_domain = channel.recv_many(ROWS_PER_HYPERCUBE_VERTEX)?;

    let univariate_message_coeffs = iter::chain(
        iter::repeat_n(C::Elem::zero(), ROWS_PER_HYPERCUBE_VERTEX),
        univariate_message_coeffs_ext_domain,
    )
    .collect::<Vec<_>>();

    let univariate_sumcheck_challenge = channel.sample();

    let sumcheck_claim = extrapolate_over_subspace(
        round_message_univariate_domain,
        &univariate_message_coeffs,
        univariate_sumcheck_challenge.clone(),
    );

    let SumcheckOutput {
        eval,
        challenges: mut eval_point,
    } = verify(all_zerocheck_challenges, 2, sumcheck_claim, channel)?;

    let [a_eval, b_eval, c_eval] = channel.recv_array()?;

    channel.assert_zero(a_eval.clone() * &b_eval - &c_eval - &eval)?;

    eval_point.reverse();

    Ok(AndCheckOutput {
        a_eval,
        b_eval,
        c_eval,
        z_challenge: univariate_sumcheck_challenge,
        eval_point,
    })
}
