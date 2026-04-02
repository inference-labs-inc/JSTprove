// Copyright 2025 Irreducible Inc.

use binius_core::word::Word;
use binius_field::{
    AESTowerField8b, BinaryField, Field, PackedField, UnderlierWithBitOps, WithUnderlier,
    util::powers,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
    FieldBuffer, inner_product::inner_product, multilinear::eq::eq_ind_partial_eval,
};
use binius_verifier::protocols::sumcheck::SumcheckOutput;

use super::{
    error::Error, key_collection::KeyCollection, phase_1::prove_phase_1, phase_2::prove_phase_2,
};

/// Holds the prover data for an operator.
///
/// Contains evaluation claims and challenge points for an operation.
///
/// Each operator (AND/MUL) has multiple operand positions, each with an oblong evaluation claim.
/// The `evals` field stores these claim evaluations. The evaluation points consist of:
/// - `r_zhat_prime`: univariate challenge point (pre-populated)
/// - `r_x_prime`: multilinear challenge point (pre-populated)
#[derive(Debug, Clone)]
pub struct OperatorData<F: Field> {
    pub evals: Vec<F>,
    pub r_zhat_prime: F,
    pub r_x_prime: Vec<F>,
}

/// Prepared operator data for proving.
///
/// Contains evaluation claims, challenge points, and precomputed values needed during proving:
/// - `evals`: evaluation claims for each operand position
/// - `r_zhat_prime`: univariate challenge point
/// - `r_x_prime_tensor`: tensor expansion of r_x_prime for efficient proving
/// - `lambda`: sampled random value for operand weighting
#[derive(Debug, Clone)]
pub struct PreparedOperatorData<F: Field> {
    pub evals: Vec<F>,
    pub r_zhat_prime: F,
    pub r_x_prime_tensor: FieldBuffer<F>,
    pub lambda_powers: Vec<F>,
}

impl<F: Field> PreparedOperatorData<F> {
    /// Creates a new prepared operator data from operator data and lambda.
    pub fn new(operator_data: OperatorData<F>, lambda: F) -> Self {
        let OperatorData {
            evals,
            r_zhat_prime,
            r_x_prime,
        } = operator_data;
        let r_x_prime_tensor = eq_ind_partial_eval::<F>(&r_x_prime);
        let lambda_powers = powers(lambda).skip(1).take(evals.len()).collect();
        Self {
            evals,
            r_zhat_prime,
            r_x_prime_tensor,
            lambda_powers,
        }
    }

    /// Returns the batched evaluation of the oblong evaluation claims.
    /// Since the univariate evaluation of the evals at lambda is
    /// further multiplied by lambda, the batched evaluation claims
    /// for different operators can soundly be added without further
    /// random scaling.
    pub fn batched_eval(&self) -> F {
        inner_product(
            self.evals.iter().copied(),
            self.lambda_powers.iter().copied(),
        )
    }
}

/// Proves the shift protocol reduction using a two-phase approach.
///
/// This function orchestrates the complete shift protocol proof, reducing bitand and intmul
/// evaluation claims to a single multilinear claim on the witness. The protocol consists
/// of two sequential sumcheck phases that progressively reduce the complexity of the claims.
///
/// # Protocol Overview
/// 1. **Lambda Sampling**: Samples random coefficients for batching operator claims
/// 2. **Phase 1**: Proves batched operator claims over shift variants and operand positions
/// 3. **Phase 2**: Reduces to witness evaluation using monster multilinear polynomial
///
/// # Parameters
/// - `log_public_words`: log2 the number of public words
/// - `key_collection`: Prover's key collection representing the constraint system
/// - `words`: The witness words (must have power-of-2 length)
/// - `bitand_data`: Operator data for bit multiplication (AND) constraints
/// - `intmul_data`: Operator data for integer multiplication (MUL) constraints
/// - `transcript`: The prover's transcript for interactive protocol
///
/// # Returns
/// Returns `SumcheckOutput` containing the final challenges and witness evaluation,
/// or an error if the proof generation fails.
///
/// # Requirements
/// - `words` must have power-of-2 length for efficient multilinear operations
pub fn prove<F, P, Channel>(
    log_public_words: usize,
    key_collection: &KeyCollection,
    words: &[Word],
    bitand_data: OperatorData<F>,
    intmul_data: OperatorData<F>,
    channel: &mut Channel,
) -> Result<SumcheckOutput<F>, Error>
where
    F: BinaryField + From<AESTowerField8b> + WithUnderlier<Underlier: UnderlierWithBitOps>,
    P: PackedField<Scalar = F> + WithUnderlier<Underlier: UnderlierWithBitOps>,
    Channel: IPProverChannel<F>,
{
    // Sample lambdas, one for each operator.
    let bitand_lambda = channel.sample();
    let intmul_lambda = channel.sample();

    // Create prepared operator data with sampled lambdas
    let expand_scope = tracing::debug_span!("Expand tensor queries").entered();
    let prepared_bitand_data = PreparedOperatorData::new(bitand_data, bitand_lambda);
    let prepared_intmul_data = PreparedOperatorData::new(intmul_data, intmul_lambda);
    drop(expand_scope);

    // Prove the first phase, receiving a `SumcheckOutput`
    // with challenges made of `r_j` and `r_s`,
    // and eval equal to `gamma` (see paper).
    let phase_1_output = prove_phase_1::<_, P, _>(
        key_collection,
        words,
        &prepared_bitand_data,
        &prepared_intmul_data,
        channel,
    )?;

    // Prove the second phase, receiving a `SumcheckOutput`
    // with challenges `r_y` and eval the evaluation of
    // the witness at oblong point had by univariate
    // variable `r_j` and multilinear variable `r_y`.
    let SumcheckOutput { challenges, eval } = prove_phase_2::<_, P, _>(
        log_public_words,
        key_collection,
        words,
        &prepared_bitand_data,
        &prepared_intmul_data,
        phase_1_output,
        channel,
    )?;

    // Return evaluation claim on the witness.
    Ok(SumcheckOutput { challenges, eval })
}
