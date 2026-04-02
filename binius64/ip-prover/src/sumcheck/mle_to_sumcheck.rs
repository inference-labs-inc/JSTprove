// Copyright 2025 Irreducible Inc.
use binius_field::Field;
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::multilinear::eq::eq_one_var;

use crate::sumcheck::{
    Error,
    common::{MleCheckProver, SumcheckProver},
    round_evals::round_coeffs_by_eq,
};

/// Adaptor that exposes a `SumcheckProver` interface for an internal `MleCheckProver`.
///
/// This struct implements the technique from [Gruen24] to convert an MLE-check protocol
/// into a standard sumcheck protocol. The key insight is that the MLE-check claim
/// $\sum_{v \in \{0,1\}^n} F(v) \cdot \text{eq}(v, z) = s$ can be rewritten as a sumcheck
/// claim by multiplying in the equality polynomial term-by-term during the protocol execution.
///
/// In each round, the adaptor multiplies the round polynomials from the inner MLE-check
/// prover by a linear polynomial term $(X - \alpha)$ where $\alpha$ is the corresponding
/// coordinate of the evaluation point. This effectively transforms the MLE-check round
/// polynomial into a sumcheck round polynomial that includes the equality check.
///
/// The `eq_prefix_eval` field accumulates the product of all previously factored equality
/// terms, ensuring the round polynomials maintain the correct scaling throughout the protocol.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[derive(Debug, Clone)]
pub struct MleToSumCheckDecorator<F: Field, InnerProver> {
    mlecheck_prover: InnerProver,
    eq_prefix_eval: F,
}

impl<F: Field, InnerProver: MleCheckProver<F>> MleToSumCheckDecorator<F, InnerProver> {
    pub fn new(mlecheck_prover: InnerProver) -> Self {
        Self {
            mlecheck_prover,
            eq_prefix_eval: F::ONE,
        }
    }
}

impl<F: Field, InnerProver: MleCheckProver<F>> SumcheckProver<F>
    for MleToSumCheckDecorator<F, InnerProver>
{
    fn n_vars(&self) -> usize {
        self.mlecheck_prover.n_vars()
    }

    fn n_claims(&self) -> usize {
        self.mlecheck_prover.n_claims()
    }

    fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
        let round_coeffs_multi = self.mlecheck_prover.execute()?;

        // Multiply the round polynomials from the inner MLE-check prover by (X - α).
        let alpha = self.mlecheck_prover.eval_point()[self.n_vars() - 1];
        let wrapped_round_coeffs = round_coeffs_multi
            .into_iter()
            .map(|round_coeffs| round_coeffs_by_eq(&round_coeffs, alpha) * self.eq_prefix_eval)
            .collect();

        Ok(wrapped_round_coeffs)
    }

    fn fold(&mut self, challenge: F) -> Result<(), Error> {
        if self.n_vars() == 0 {
            return Err(Error::ExpectedFinish);
        }

        let alpha = self.mlecheck_prover.eval_point()[self.n_vars() - 1];
        self.eq_prefix_eval *= eq_one_var(challenge, alpha);

        self.mlecheck_prover.fold(challenge)
    }

    fn finish(self) -> Result<Vec<F>, Error> {
        self.mlecheck_prover.finish()
    }
}
