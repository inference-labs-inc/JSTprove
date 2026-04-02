// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, field::FieldOps, util::powers};
use binius_math::multilinear::eq::eq_ind_partial_eval_scalars;

use crate::{
    channel::IPVerifierChannel,
    sumcheck::{self, RoundCoeffs, SumcheckOutput},
};

/// Returns `(m_n, m_d)` dimensions for a mask polynomial buffer.
///
/// The ZK MLE-check protocol uses a separable mask polynomial with `n_vars` univariate polynomials,
/// each of degree `d`. The mask coefficients are stored in a `2^m_n × 2^m_d` matrix where:
/// - `m_n`: log of number of rows (one per variable, padded to power of two)
/// - `m_d`: log of row size (degree + 1 coefficients, padded to power of two)
///
/// The protocol imposes `n * d + 1` linear constraints on the coefficients. The `n_extra_dof`
/// parameter specifies additional degrees of freedom needed (e.g., for FRI openings), which
/// may increase `m_n` to ensure `2^(m_n + m_d) >= n * d + 1 + n_extra_dof`.
///
/// # Arguments
///
/// * `n_vars` - Number of variables (n).
/// * `degree` - Degree of each univariate polynomial (d).
/// * `n_extra_dof` - Number of additional degrees of freedom.
///
/// # Returns
///
/// A tuple `(m_n, m_d)` where the total mask buffer size is `2^(m_n + m_d)`.
pub fn mask_buffer_dimensions(n_vars: usize, degree: usize, n_extra_dof: usize) -> (usize, usize) {
    let min_buffer_size = n_vars * degree + 1 + n_extra_dof;
    let m_d = (degree + 1).next_power_of_two().ilog2() as usize;
    // m_n must be large enough to hold n_vars rows AND satisfy the DOF constraint
    let m_n_for_vars = n_vars.next_power_of_two().ilog2() as usize;
    let m_n_for_size = (min_buffer_size.next_power_of_two().ilog2() as usize).saturating_sub(m_d);
    let m_n = m_n_for_vars.max(m_n_for_size);
    (m_n, m_d)
}

/// Output of the zero-knowledge MLE-check verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyZKOutput<F> {
    /// The reduced evaluation of the main polynomial at the challenge point.
    pub eval: F,
    /// The evaluation of the mask polynomial at the challenge point.
    pub mask_eval: F,
    /// The sequence of challenge values from each round.
    pub challenges: Vec<F>,
}

/// An MLE-check protocol is an interactive protocol similar to sumcheck, but with modifications
/// introduced in [Gruen24], Section 3.
///
/// The prover in an MLE-check argues a that for some $n$-variate polynomial
/// $F(X_0, \ldots, X_{n-1})$ (which is not necessarily multilinear), for a given point
/// $(z_0, \ldots, z_{n-1})$ and claimed value $s$, that
///
/// $$
/// s = \sum_{v \in B_n} F(v) \cdot eq(v, z)
/// $$
///
/// Unless $F$ is indeed multilinear, $s \ne F(z)$ necessarily. While the prover and verifier could
/// engage in a standard sumcheck protocol to reduce this claim, it is concretely more efficient to
/// use the optimized protocol from [Gruen24], which we call an "MLE-check".
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
///
/// ## Arguments
///
/// * `point` - The evaluation point for the multilinear extension
/// * `degree` - The degree of the univariate polynomial in each round
/// * `eval` - The claimed multilinear-extension evaluation of the multivariate polynomial
/// * `channel` - The channel for receiving prover messages and sampling challenges
///
/// ## Returns
///
/// Returns a `Result` containing the `SumcheckOutput` with the reduced evaluation and challenge
/// point, or an error if verification fails.
pub fn verify<F, C>(
    point: &[C::Elem],
    degree: usize,
    mut eval: C::Elem,
    channel: &mut C,
) -> Result<SumcheckOutput<C::Elem>, sumcheck::Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    let n_vars = point.len();

    let mut challenges = Vec::with_capacity(n_vars);
    for z_i in point.iter().rev() {
        let round_proof = RoundProof(RoundCoeffs(channel.recv_many(degree)?));
        let challenge = channel.sample();

        let round_coeffs = round_proof.recover(eval, z_i.clone());
        eval = round_coeffs.evaluate(challenge.clone());
        challenges.push(challenge);
    }

    Ok(SumcheckOutput { eval, challenges })
}

/// Variation of the MLE-check protocol that provides the hiding property.
///
/// This protocol is based on the zero-knowledge sumcheck technique from [Libra], with a
/// modification. When the field has characteristic 2, the Libra ZK-sumcheck protocol is not hiding.
/// Instead, the mask polynomial $g$ is batched together with the multivariate polynomial whose MLE
/// is being evaluated, whereas Libra would batch the mask polynomial together with the MLE itself.
///
/// [Libra]: <https://dl.acm.org/doi/10.1007/978-3-030-26954-8_24>
pub fn verify_zk<F, C>(
    point: &[C::Elem],
    degree: usize,
    eval: C::Elem,
    channel: &mut C,
) -> Result<VerifyZKOutput<C::Elem>, sumcheck::Error>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    // Read the evaluation of the MLE of the mask polynomial (g).
    let mask_eval = channel.recv_one()?;

    // Randomly mix the evaluation claim with the mask evaluation claim.
    let batch_challenge = channel.sample();
    let batch_eval = eval + batch_challenge.clone() * mask_eval.clone();

    let SumcheckOutput {
        eval: batch_eval_out,
        challenges,
    } = verify(point, degree, batch_eval, channel)?;

    // Read the evaluation of the mask polynomial (g) at the sumcheck challenge point.
    let mask_eval_out = channel.recv_one()?;

    let eval_out = batch_eval_out - batch_challenge * mask_eval_out.clone();
    Ok(VerifyZKOutput {
        eval: eval_out,
        mask_eval: mask_eval_out,
        challenges,
    })
}

/// An MLE-check round proof is a univariate polynomial in monomial basis with the coefficient of
/// the lowest-degree term truncated off.
///
/// Since the verifier knows the claimed linear extrapolation of the polynomial values at the
/// points 0 and 1, the low-degree term coefficient can be easily recovered. Truncating the
/// coefficient off saves a small amount of proof data.
///
/// This is an analogous struct to [`sumcheck::RoundProof`], except that we truncate the low-degree
/// coefficient instead of the high-degree coefficient.
///
/// In a sumcheck protocol, the verifier has a claimed sum $s$ and the round polynomial $R(X)$ must
/// satisfy $R(0) + R(1) = s$. In an MLE-check protocol, the verifier has a claimed coordinate
/// $\alpha$ and extrapolated value $s$ and the round polynomial must satisfy
/// $(1 - \alpha) R(0) + \alpha R(1) = s$. This difference changes the recovery procedure and which
/// polynomial coefficient is most convenient to truncate.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoundProof<F>(pub RoundCoeffs<F>);

impl<F> RoundProof<F> {
    /// Truncates the polynomial coefficients to a round proof.
    ///
    /// Removes the first coefficient. See the struct documentation for more info.
    ///
    /// ## Pre-conditions
    ///
    /// * `coeffs` must not be empty
    pub fn truncate(mut coeffs: RoundCoeffs<F>) -> Self {
        coeffs.0.remove(0);
        Self(coeffs)
    }

    /// Recovers all univariate polynomial coefficients from the compressed round proof.
    ///
    /// The prover has sent coefficients for the purported $i$'th round polynomial
    /// $R(X) = \sum_{j=0}^d a_j * X^j$.
    ///
    /// However, the prover has not sent the lowest degree coefficient $a_0$. The verifier will
    /// need to recover this missing coefficient.
    ///
    /// Let $s$ denote the current round's claimed sum and $\alpha_i$ be the $i$'th coordinate of
    /// the evaluation point.
    ///
    /// The verifier expects the round polynomial $R_i$ to satisfy the identity
    /// $s = (1 - \alpha) R(0) + \alpha R(1)$, or equivalently, $s = R(0) + (R(1) - R(0)) \alpha$.
    ///
    /// Using
    ///     $R(0) = a_0$
    ///     $R(1) = \sum_{j=0}^d a_j$
    /// There is a unique $a_0$ that allows $R$ to satisfy the above identity. Specifically,
    /// $a_0 = s - \alpha \sum_{j=1}^d a_j$.
    pub fn recover(self, eval: F, alpha: F) -> RoundCoeffs<F>
    where
        F: FieldOps,
    {
        let Self(RoundCoeffs(mut coeffs)) = self;
        let first_coeff = eval - alpha * coeffs.iter().cloned().sum::<F>();
        coeffs.insert(0, first_coeff);
        RoundCoeffs(coeffs)
    }

    /// The truncated polynomial coefficients.
    pub fn coeffs(&self) -> &[F] {
        &self.0.0
    }
}

/// Evaluates the MLE of the libra_eval polynomial at a query point.
///
/// Computes the multilinear extension of `libra_eval_r` at the query point `(query_j, query_k)`:
///
/// ```text
/// Σⱼ Σₖ eq(j, query_j) · eq(k, query_k) · r[j]^k
/// ```
///
/// for `j < n_vars` and `k ≤ degree`, where `eq` is the equality indicator polynomial.
///
/// # Arguments
///
/// * `challenge_point` - The challenge point `r` from sumcheck (length `n_vars`)
/// * `query_j` - Query point for the variable index (length `m_n`)
/// * `query_k` - Query point for the power index (length `m_d`)
/// * `n_vars` - Number of variables in the mask polynomial
/// * `degree` - Degree of each univariate in the mask polynomial
pub fn libra_eval<F: FieldOps>(
    challenge_point: &[F],
    query_j: &[F],
    query_k: &[F],
    n_vars: usize,
    degree: usize,
) -> F {
    let eq_j = eq_ind_partial_eval_scalars(query_j);
    let eq_k = eq_ind_partial_eval_scalars(query_k);

    eq_j.iter()
        .take(n_vars)
        .zip(challenge_point)
        .map(|(eq_j_val, r_j)| {
            eq_k.iter()
                .take(degree + 1)
                .zip(powers(r_j.clone()))
                .map(|(eq_k_val, r_j_power)| eq_j_val.clone() * eq_k_val.clone() * r_j_power)
                .sum::<F>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use binius_field::{Random, arch::OptimalB128 as B128};
    use binius_math::{line::extrapolate_line_packed, test_utils::random_scalars};
    use rand::prelude::*;

    use super::*;

    fn test_recover_with_degree<F: Field>(mut rng: impl Rng, alpha: F, degree: usize) {
        let coeffs = RoundCoeffs(random_scalars(&mut rng, degree + 1));

        let v0 = coeffs.evaluate(F::ZERO);
        let v1 = coeffs.evaluate(F::ONE);
        let eval = extrapolate_line_packed(v0, v1, alpha);

        let proof = RoundProof::truncate(coeffs.clone());
        assert_eq!(proof.recover(eval, alpha), coeffs);
    }

    #[test]
    fn test_recover() {
        let mut rng = StdRng::seed_from_u64(0);
        let alpha = B128::random(&mut rng);

        for degree in 0..4 {
            // Test with random coordinate
            test_recover_with_degree(&mut rng, alpha, degree);

            // Test edge case coordinate values
            test_recover_with_degree(&mut rng, B128::ZERO, degree);
            test_recover_with_degree(&mut rng, B128::ONE, degree);
        }
    }
}
