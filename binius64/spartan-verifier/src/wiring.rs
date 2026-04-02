// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{Field, field::FieldOps};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
    multilinear::eq::{eq_ind, eq_ind_partial_eval_scalars, eq_one_var},
    univariate::evaluate_univariate,
};
use binius_spartan_frontend::constraint_system::{
    ConstraintSystemPadded, MulConstraint, WitnessIndex,
};
use binius_verifier::protocols::{basefold, sumcheck};

/// Claim components from the wiring check computation via IOP channel.
#[derive(Debug, Clone)]
pub struct WiringClaim<F> {
    /// Batching challenge for constraint operands.
    pub lambda: F,
    /// Coefficient for batching public input check with wiring check.
    pub batch_coeff: F,
    /// The batched sum of all claims.
    pub batched_sum: F,
}

/// Computes the wiring claim using an IOP channel interface.
///
/// Samples the batching challenges and computes the batched claim from the
/// evaluation claims and public input evaluation.
pub fn compute_claim<F, C>(
    _constraint_system: &ConstraintSystemPadded,
    _r_public: &[C::Elem],
    eval_claims: &[C::Elem],
    public_eval: C::Elem,
    channel: &mut C,
) -> WiringClaim<C::Elem>
where
    F: Field,
    C: IPVerifierChannel<F>,
{
    // \lambda is the batching challenge for the constraint operands
    let lambda = channel.sample();

    // Coefficient for batching the public input check with the wiring check.
    let batch_coeff = channel.sample();

    // Batch together the witness public input consistency claim with the
    // constraint operand evaluation claims.
    let batched_sum =
        evaluate_univariate(eval_claims, lambda.clone()) + batch_coeff.clone() * public_eval;

    WiringClaim {
        lambda,
        batch_coeff,
        batched_sum,
    }
}

/// Returns a closure that evaluates the wiring transparent polynomial at a given point.
///
/// The returned closure computes the expected evaluation of the wiring MLE batched with the
/// public input equality check, given a challenge point from the BaseFold opening.
pub fn eval_transparent<F: FieldOps + 'static>(
    constraint_system: &ConstraintSystemPadded,
    r_public: &[F],
    r_x: &[F],
    lambda: F,
    batch_coeff: F,
) -> binius_iop::channel::TransparentEvalFn<F> {
    let r_public = r_public.to_vec();
    let r_x = r_x.to_vec();
    let mul_constraints = constraint_system.mul_constraints().to_vec();

    Box::new(move |r_y: &[F]| {
        let wiring_eval = evaluate_wiring_mle(&mul_constraints, lambda.clone(), &r_x, r_y);

        // Evaluate eq(r_public || ZERO, r_y)
        let (r_y_head, r_y_tail) = r_y.split_at(r_public.len());
        let eq_head = eq_ind(&r_public, r_y_head);
        let eq_public = r_y_tail.iter().fold(eq_head, |eval, r_y_i| {
            eval * eq_one_var(r_y_i.clone(), F::zero())
        });

        wiring_eval + batch_coeff.clone() * eq_public
    })
}

pub fn evaluate_wiring_mle<F: FieldOps>(
    mul_constraints: &[MulConstraint<WitnessIndex>],
    lambda: F,
    r_x: &[F],
    r_y: &[F],
) -> F {
    let mut acc = [F::zero(), F::zero(), F::zero()];

    let r_x_tensor = eq_ind_partial_eval_scalars(r_x);
    let r_y_tensor = eq_ind_partial_eval_scalars(r_y);
    for (r_x_tensor_i, MulConstraint { a, b, c }) in iter::zip(&r_x_tensor, mul_constraints) {
        for (dst, operand) in iter::zip(&mut acc, [a, b, c]) {
            let r_y_tensor_sum = operand
                .wires()
                .iter()
                .map(|j| r_y_tensor[j.0 as usize].clone())
                .sum::<F>();
            *dst += r_x_tensor_i.clone() * r_y_tensor_sum;
        }
    }

    evaluate_univariate(&acc, lambda)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("transcript error: {0}")]
    Transcript(#[from] binius_transcript::Error),
    #[error("BaseFold error: {0}")]
    BaseFold(#[from] basefold::Error),
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] sumcheck::Error),
}
