// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, ExtensionField, PackedField};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
    FieldBuffer,
    multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
    tensor_algebra::TensorAlgebra,
};

use crate::config::B1;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("channel: {0}")]
    Channel(#[from] binius_ip::channel::Error),
}

/// Output of ring-switching verification.
pub struct RingSwitchVerifyOutput<F: BinaryField + PackedField<Scalar = F>> {
    /// The row-batching challenges (expanded via eq_ind).
    pub eq_r_double_prime: FieldBuffer<F>,
    /// The verified sumcheck claim for BaseFold.
    pub sumcheck_claim: F,
}

/// Verify the ring-switching reduction.
///
/// Takes the expected evaluation claim and eval point, and:
/// 1. Receives s_hat_v from prover via channel
/// 2. Verifies the partial evaluation matches the expected claim
/// 3. Samples row-batching challenges
/// 4. Computes the sumcheck claim for BaseFold
///
/// Returns the eq_r_double_prime (for final check) and sumcheck claim.
///
/// ## Arguments
///
/// * `evaluation_claim` - the expected evaluation of the B1 polynomial
/// * `eval_point` - the evaluation point from shift reduction
/// * `channel` - the verifier channel for receiving/sampling
///
/// ## Preconditions
///
/// * `eval_point.len()` must equal `log_witness_elems + log_packing` where log_packing is the
///   base-2 log of the extension degree of F over B1
pub fn verify<F, C>(
    evaluation_claim: F,
    eval_point: &[F],
    channel: &mut C,
) -> Result<RingSwitchVerifyOutput<F>, Error>
where
    F: BinaryField + PackedField<Scalar = F>,
    C: IPVerifierChannel<F, Elem = F>,
{
    let log_packing = <F as ExtensionField<B1>>::LOG_DEGREE;
    let (eval_point_low, _eval_point_high) = eval_point.split_at(log_packing);

    // Receive s_hat_v
    let s_hat_v = channel.recv_many(1 << log_packing)?;
    let s_hat_v_buf = FieldBuffer::from_values(&s_hat_v);

    // Verify partial eval matches expected claim
    let computed_claim = evaluate::<F, F, _>(&s_hat_v_buf, eval_point_low);
    channel.assert_zero(evaluation_claim - computed_claim)?;

    // Basis transpose
    let s_hat_u = FieldBuffer::from_values(&TensorAlgebra::<B1, F>::new(s_hat_v).transpose().elems);

    // Sample r_double_prime
    let r_double_prime = channel.sample_many(log_packing);
    let eq_r_double_prime = eq_ind_partial_eval::<F>(&r_double_prime);

    // Compute sumcheck claim
    let sumcheck_claim = evaluate::<F, F, _>(&s_hat_u, &r_double_prime);

    Ok(RingSwitchVerifyOutput {
        eq_r_double_prime,
        sumcheck_claim,
    })
}

/// Evaluate the ring switching equality indicator at a given point.
///
/// The ring switching equality indicator is the multilinear function $A$ from [DP24],
/// Construction 3.1. It is evaluated succinctly by computing the equality indicator over the
/// tensor algebra, where the first components live in the vertical subring and the later
/// components live in the vertical subring. Then we apply row-batching to the tensor algebra
/// element.
///
/// ## Arguments
///
/// * `z_vals` - the vertical evaluation point, with $\ell'$ components
/// * `query` - the horizontal evaluation point
/// * `expanded_row_batch_query` - the scaling elements for row-batching
///
/// ## Pre-conditions
///
/// * the lengths of `z_vals` and `query` are equal
/// * the length of `expanded_row_batch_query` must equal `FE::DEGREE`
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn eval_rs_eq<F>(z_vals: &[F], query: &[F], expanded_row_batch_query: &[F]) -> F
where
    F: BinaryField,
{
    assert_eq!(z_vals.len(), query.len()); // pre-condition
    assert_eq!(expanded_row_batch_query.len(), F::DEGREE); // pre-condition

    let tensor_eval = iter::zip(z_vals, query).fold(
        <TensorAlgebra<B1, F>>::from_vertical(F::ONE),
        |eval, (&vert_i, &hztl_i)| {
            // This formula is specific to characteristic 2 fields
            // Here we know that $h v + (1 - h) (1 - v) = 1 + h + v$.
            let vert_scaled = eval.clone().scale_vertical(vert_i);
            let hztl_scaled = eval.clone().scale_horizontal(hztl_i);

            eval + &vert_scaled + &hztl_scaled
        },
    );

    tensor_eval.fold_vertical(expanded_row_batch_query)
}
