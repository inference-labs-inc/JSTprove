// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, Field, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
    FieldBuffer,
    multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
};
use binius_verifier::{config::LOG_WORD_SIZE_BITS, protocols::sumcheck::SumcheckOutput};
use tracing::instrument;

use super::{
    error::Error, key_collection::KeyCollection, monster::build_monster_multilinear,
    prove::PreparedOperatorData,
};
use crate::{
    fold_word::fold_words,
    protocols::sumcheck::{
        ProveSingleOutput, bivariate_product::BivariateProductSumcheckProver, prove_single,
    },
};

/// Proves the second phase of the shift protocol reduction.
///
/// This function implements phase 2 of the shift protocol prover, which takes the output
/// from phase 1 and completes the shift reduction by proving the relationship between
/// the witness and the monster multilinear polynomial.
///
/// # Protocol Steps
/// 1. **Challenge Splitting**: Splits phase 1 challenges into `r_j` and `r_s` components
/// 2. **Witness Folding**: Folds the witness words using the `r_j` challenges
/// 3. **Monster Multilinear Construction**: Builds the monster multilinear from key collection and
///    operator data
/// 4. **Sumcheck Execution**: Runs bivariate product sumcheck to prove witness ×
///    monster_multilinear relationship
///
/// # Parameters
/// - `key_collection`: Prover's key collection representing the constraint system
/// - `words`: The witness words
/// - `bitand_data`: Operator data for bit multiplication constraints
/// - `intmul_data`: Operator data for integer multiplication constraints
/// - `phase_1_output`: Challenges and evaluation from the first phase
/// - `transcript`: The prover's transcript
///
/// # Returns
/// Returns `SumcheckOutput` containing the combined challenges `[r_j, r_y]` and witness evaluation,
/// or an error if the protocol fails.
#[instrument(skip_all, name = "prove_phase_2")]
pub fn prove_phase_2<F, P: PackedField<Scalar = F>, Channel>(
    inout_n_vars: usize,
    key_collection: &KeyCollection,
    words: &[Word],
    bitand_data: &PreparedOperatorData<F>,
    intmul_data: &PreparedOperatorData<F>,
    phase_1_output: SumcheckOutput<F>,
    channel: &mut Channel,
) -> Result<SumcheckOutput<F>, Error>
where
    F: BinaryField + From<AESTowerField8b>,
    Channel: IPProverChannel<F>,
{
    let SumcheckOutput {
        challenges: mut r_jr_s,
        eval: gamma,
    } = phase_1_output;
    // Split challenges as r_j,r_s where r_j is the first LOG_WORD_SIZE_BITS
    // variables and r_s is the last LOG_WORD_SIZE_BITS variables
    // Thus r_s are the more significant variables.
    let r_s = r_jr_s.split_off(LOG_WORD_SIZE_BITS);
    let r_j = r_jr_s;

    let r_j_tensor = eq_ind_partial_eval::<F>(&r_j);
    let r_j_witness = fold_words::<_, P>(words, r_j_tensor.as_ref());

    let monster_multilinear =
        build_monster_multilinear(key_collection, bitand_data, intmul_data, &r_j, &r_s)?;

    run_sumcheck(
        inout_n_vars,
        r_j_witness,
        monster_multilinear,
        r_j,
        gamma,
        channel,
    )
}

/// Evaluates the r_j_witness multilinear at the inout evaluation point.
///
/// This helper function extracts the public input chunk from the r_j_witness
/// and evaluates it at the given inout_eval_point, corresponding to evaluating
/// the witness at `(r_j, inout_eval_point || 0)`.
///
/// # Parameters
/// - `r_j_witness`: The witness folded at challenges `r_j`
/// - `inout_eval_point`: Challenge point for the public input/output variables
///
/// # Returns
/// The evaluation of the public chunk at `inout_eval_point`.
fn evaluate_public_at_inout<F: Field, P: PackedField<Scalar = F>>(
    r_j_witness: &FieldBuffer<P>,
    inout_eval_point: &[F],
) -> F {
    let public_chunk = r_j_witness.chunk(inout_eval_point.len(), 0);
    evaluate(&public_chunk, inout_eval_point)
}

/// Computes the batched polynomial m = monster + χ eq(inout_eval_point || 0),
/// where monster is the monster multilinear and χ is `batch_coeff`.
///
/// Since eq(inout_eval_point || 0) takes the value 0 on all hypercube vertices
/// except the first 2^inout_eval_point.len(), only that first chunk needs to be updated.
///
/// # Parameters
/// - `monster_multilinear`: The monster multilinear polynomial
/// - `inout_eval_point`: Challenge point for the public input/output variables
/// - `batch_coeff`: Random batching coefficient
///
/// # Returns
/// The combined polynomial `monster_multilinear + batch_coeff * eq(inout_eval_point || 0)`.
fn compute_monster_with_inout<F: Field, P: PackedField<Scalar = F>>(
    monster_multilinear: FieldBuffer<P>,
    inout_eval_point: &[F],
    batch_coeff: F,
) -> FieldBuffer<P> {
    let mut combined_monster = monster_multilinear;

    {
        let mut public_chunk = combined_monster.chunk_mut(inout_eval_point.len(), 0);
        let mut public_chunk = public_chunk.get();

        let eq_inout = eq_ind_partial_eval::<P>(inout_eval_point);

        let batch_coeff_packed = P::broadcast(batch_coeff);
        for (dst, src) in iter::zip(public_chunk.as_mut(), eq_inout.as_ref()) {
            *dst += *src * batch_coeff_packed;
        }
    }

    combined_monster
}

/// Executes the bivariate product sumcheck for the witness and monster multilinear relationship.
///
/// This helper function runs the sumcheck protocol to prove the relationship between
/// the witness and monster multilinear, batched with the public input check.
///
/// # Protocol Details
/// - Samples challenge point for public inputs and computes public evaluation
/// - Samples batching coefficient and merges public check into monster multilinear
/// - Uses single `BivariateProductSumcheckProver` for the batched relationship
/// - Extracts witness evaluation from the sumcheck output
/// - In debug mode, verifies the witness evaluation against expected value
///
/// # Parameters
/// - `inout_n_vars`: Number of variables for the public input/output point
/// - `r_j_witness`: The witness folded at challenges `r_j`
/// - `monster_multilinear`: The monster multilinear polynomial constructed from constraints
/// - `r_j`: Challenge vector from phase 1 (first `LOG_WORD_SIZE_BITS` challenges)
/// - `gamma`: The claimed evaluation from phase 1
/// - `transcript`: The prover's transcript
///
/// # Returns
/// Returns `SumcheckOutput` with concatenated challenges `[r_j, r_y]` and witness evaluation.
#[instrument(skip_all, name = "run_sumcheck")]
fn run_sumcheck<F: Field, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>>(
    inout_n_vars: usize,
    r_j_witness: FieldBuffer<P>,
    monster_multilinear: FieldBuffer<P>,
    r_j: Vec<F>,
    gamma: F,
    channel: &mut Channel,
) -> Result<SumcheckOutput<F>, Error> {
    #[cfg(debug_assertions)]
    let cloned_r_j_witness_for_debugging = r_j_witness.clone();

    // Sample inout evaluation point and compute public evaluation
    let inout_eval_point = channel.sample_many(inout_n_vars);
    let public_eval = evaluate_public_at_inout(&r_j_witness, &inout_eval_point);

    // Sample batching coefficient
    let batch_coeff = channel.sample();

    // Compute the batched polynomial m = monster + χ eq(inout_eval_point || 0)
    let combined_monster =
        compute_monster_with_inout(monster_multilinear, &inout_eval_point, batch_coeff);

    // Run sumcheck on bivariate product
    let batched_sum = gamma + batch_coeff * public_eval;
    let prover = BivariateProductSumcheckProver::new([r_j_witness, combined_monster], batched_sum)?;

    let ProveSingleOutput {
        multilinear_evals,
        challenges: mut r_y,
    } = prove_single(prover, channel)?;

    // Reverse the challenges to get the evaluation point.
    r_y.reverse();

    // Extract witness evaluation
    let [witness_eval, _monster_eval] = multilinear_evals
        .try_into()
        .expect("prover has 2 multilinear polynomials");

    channel.send_one(witness_eval);

    #[cfg(debug_assertions)]
    {
        let r_y_tensor = eq_ind_partial_eval(&r_y);
        let expected_witness_eval = binius_math::inner_product::inner_product_buffers(
            &cloned_r_j_witness_for_debugging,
            &r_y_tensor,
        );
        debug_assert_eq!(witness_eval, expected_witness_eval);
    }

    Ok(SumcheckOutput {
        challenges: [r_j, r_y].concat(),
        eval: witness_eval,
    })
}
