use arith::Field;
use gkr_engine::Transcript;

use super::{IOPProverMessage, IOPVerifierState, SumCheckSubClaim};

impl<F: Field> IOPVerifierState<F> {
    /// Initialize the verifier's state.
    pub fn verifier_init(num_vars: usize) -> Self {
        Self {
            round: 1,
            num_vars,
            finished: false,
            polynomials_received: Vec::with_capacity(num_vars),
            challenges: Vec::with_capacity(num_vars),
        }
    }

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    pub fn verify_round_and_update_state(
        &mut self,
        prover_msg: &IOPProverMessage<F>,
        transcript: &mut impl Transcript,
    ) -> F {
        if self.finished {
            panic!("Incorrect verifier state: Verifier is already finished.")
        }

        // In an interactive protocol, the verifier should
        //
        // 1. check if the received 'P(0) + P(1) = expected`.
        // 2. set `expected` to P(r)`
        //
        // When we turn the protocol to a non-interactive one, it is sufficient to defer
        // such checks to `check_and_generate_subclaim` after the last round.

        let challenge = transcript.generate_field_element::<F>();
        self.challenges.push(challenge);
        self.polynomials_received
            .push(prover_msg.evaluations.clone());

        if self.round == self.num_vars {
            // accept and close
            self.finished = true;
        } else {
            // proceed to the next round
            self.round += 1;
        }

        challenge
    }

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    pub fn check_and_generate_subclaim(&self, asserted_sum: &F) -> (bool, SumCheckSubClaim<F>) {
        if !self.finished {
            panic!("Incorrect verifier state: Verifier has not finished.");
        }

        if self.polynomials_received.len() != self.num_vars {
            return (false, SumCheckSubClaim::default());
        }

        let mut expected = *asserted_sum;

        for i in 0..self.num_vars {
            let evals = &self.polynomials_received[i];

            // check that the sum received from last round is correct
            if expected != evals[0] + evals[1] {
                return (false, SumCheckSubClaim::default());
            }

            // the univariate polynomial f is received in its extrapolated form, i.e.,
            //   h(0) = evals[0], h(1) = evals[1], h(2) = evals[2]
            // that is, suppose h = h_0 + h_1 * x + h_2 * x^2, then
            //   h(0) = h_0
            //   h(1) = h_0 + h_1 + h_2
            //   h(2) = h_0 + 2 * h_1 + 4 * h_2
            // therefore
            //   h_0 = evals[0]
            //   h_2 = (h(2) + h(0))/2 -  h(1)
            //   h_1 = h(1) - h_0 - h_2

            let h_0 = evals[0];
            let h_2 = (evals[2] + evals[0]) * F::from(2u32).inv().unwrap() - evals[1];
            let h_1 = evals[1] - h_0 - h_2;

            // now we want to compute h(r) for the challenge r = self.challenges[i]
            // h(r) = h_0 + h_1 * r + h_2 * r^2
            expected = h_0 + h_1 * self.challenges[i] + h_2 * self.challenges[i].square();
        }

        (
            true,
            SumCheckSubClaim {
                point: self.challenges.clone(),
                // the last expected value (not checked within this function) will be included in
                // the subclaim
                expected_evaluation: expected,
            },
        )
    }
}
