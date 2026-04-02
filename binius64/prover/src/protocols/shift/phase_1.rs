// Copyright 2025 Irreducible Inc.

use std::{iter, ops::Range};

use binius_core::word::Word;
use binius_field::{
    AESTowerField8b, BinaryField, Field, PackedField, UnderlierWithBitOps, WithUnderlier,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, inner_product::inner_product_buffers};
use binius_utils::rayon::prelude::*;
use binius_verifier::{
    config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS, WORD_SIZE_BYTES},
    protocols::{
        shift::SHIFT_VARIANT_COUNT,
        sumcheck::{SumcheckOutput, common::RoundCoeffs},
    },
};
use bytemuck::zeroed_vec;
use itertools::izip;
use tracing::instrument;

use super::{
    error::Error,
    key_collection::{KeyCollection, Operation},
    monster::build_h_parts,
    prove::PreparedOperatorData,
};
use crate::protocols::sumcheck::{
    bivariate_product::BivariateProductSumcheckProver, common::SumcheckProver,
};

// This is the number of variables in the g (and h) multilinears of phase 1.
const LOG_LEN: usize = LOG_WORD_SIZE_BITS + LOG_WORD_SIZE_BITS;

/// Constructs the "g" multilinear parts for both BITAND and INTMUL operations.
/// Proves the first phase of the shift reduction.
/// Computes the g and h multilinears and performs the sumcheck.
#[instrument(skip_all, name = "prover_phase_1")]
pub fn prove_phase_1<F, P, Channel>(
    key_collection: &KeyCollection,
    words: &[Word],
    bitand_data: &PreparedOperatorData<F>,
    intmul_data: &PreparedOperatorData<F>,
    channel: &mut Channel,
) -> Result<SumcheckOutput<F>, Error>
where
    F: BinaryField + From<AESTowerField8b> + WithUnderlier<Underlier: UnderlierWithBitOps>,
    P: PackedField<Scalar = F> + WithUnderlier<Underlier: UnderlierWithBitOps>,
    Channel: IPProverChannel<F>,
{
    let g_parts = build_g_parts::<_, P>(words, key_collection, bitand_data, intmul_data)?;

    // BitAnd and IntMul share the same `r_zhat_prime`.
    let h_parts = build_h_parts(bitand_data.r_zhat_prime);

    run_phase_1_sumcheck(g_parts, h_parts, channel)
}

/// Runs the phase 1 sumcheck protocol for shift constraint verification.
///
/// Executes a sumcheck over bivariate products of g and h multilinear parts for each
/// operation (BITAND, INTMUL). The protocol proves that the sum of g·h products across
/// all shift variants equals the claimed batched evaluation.
///
/// # Protocol Structure
///
/// For each operation, creates 3 bivariate product sumcheck provers (one per shift variant):
/// - g_sll · h_sll with claim `sll_sum`
/// - g_srl · h_srl with claim `srl_sum`
/// - g_sra · h_sra with claim `sar_sum = total_sum - sll_sum - srl_sum`
///
/// The g parts incorporate batching randomness (lambda weighting), while h parts
/// encode the shift operation behavior at the univariate challenge points.
///
/// # Parameters
///
/// - `g_parts`: g multilinear parts for each operation (witness-dependent)
/// - `h_parts`: h multilinear parts for each operation (challenge-dependent)
/// - `sums`: Expected total sums for each operation from lambda-weighted evaluation claims
///
/// # Returns
///
/// `SumcheckOutput` containing the challenge vector and final evaluation `gamma`
#[instrument(skip_all, name = "run_sumcheck")]
fn run_phase_1_sumcheck<F: Field, P: PackedField<Scalar = F>, Channel: IPProverChannel<F>>(
    g_parts: [FieldBuffer<P>; SHIFT_VARIANT_COUNT],
    h_parts: [FieldBuffer<P>; SHIFT_VARIANT_COUNT],
    channel: &mut Channel,
) -> Result<SumcheckOutput<F>, Error> {
    // Build `BivariateProductSumcheckProver` provers.
    let mut provers = iter::zip(g_parts, h_parts)
        .map(|(g_part, h_part)| {
            let sum = inner_product_buffers(&g_part, &h_part);
            BivariateProductSumcheckProver::new([g_part, h_part], sum)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Perform the sumcheck rounds, collecting challenges.
    let n_vars = 2 * LOG_WORD_SIZE_BITS;
    let mut challenges = Vec::with_capacity(n_vars);
    for _ in 0..n_vars {
        let mut all_round_coeffs = Vec::new();
        for prover in &mut provers {
            all_round_coeffs.extend(prover.execute()?);
        }

        let summed_round_coeffs = all_round_coeffs
            .into_iter()
            .rfold(RoundCoeffs::default(), |acc, coeffs| acc + &coeffs);

        let round_proof = summed_round_coeffs.truncate();

        channel.send_many(round_proof.coeffs());

        let challenge = channel.sample();
        challenges.push(challenge);

        for prover in &mut provers {
            prover.fold(challenge)?;
        }
    }
    challenges.reverse();

    let multilinear_evals = provers
        .into_iter()
        .map(|prover| prover.finish())
        .collect::<Result<Vec<Vec<F>>, _>>()?;

    // Evaluate the composition polynomial to compute `gamma`.
    let gamma = multilinear_evals
        .into_iter()
        .map(|prover_evals| {
            assert_eq!(prover_evals.len(), 2);
            let h_eval = prover_evals[0];
            let g_eval = prover_evals[1];
            h_eval * g_eval
        })
        .sum();

    Ok(SumcheckOutput {
        challenges,
        eval: gamma,
    })
}

/// Constructs the "g" multilinear parts for both BITAND and INTMUL operations.
///
/// This function builds the g multilinear polynomials used in phase 1 of the shift protocol.
/// For each operation (BITAND and INTMUL), it constructs three multilinear polynomials
/// corresponding to the three shift variants (SLL, SRL, SRA).
///
/// # Construction Process
///
/// 1. **Parallel Processing**: Words are processed in parallel chunks for efficiency
/// 2. **Key Processing**: For each word, iterate through its associated keys from the key
///    collection
/// 3. **Accumulation**: For each key, accumulate its contribution weighted by the r_x' tensor
/// 4. **Word Expansion**: Expand each witness word bitwise to populate the g multilinears
/// 5. **Lambda Weighting**: Apply lambda powers to weight different operand positions
///
/// # Returns
///
/// An array of multilinear extensions of each shift variant part.
///
/// # Usage
///
/// Used in phase 1 to construct the constant size g multilinears
/// that will participate in the phase 1 sumcheck protocol.
#[instrument(skip_all, name = "build_g_parts")]
fn build_g_parts<
    F: BinaryField + WithUnderlier<Underlier: UnderlierWithBitOps>,
    P: PackedField<Scalar = F> + WithUnderlier<Underlier: UnderlierWithBitOps>,
>(
    words: &[Word],
    key_collection: &KeyCollection,
    bitand_operator_data: &PreparedOperatorData<F>,
    intmul_operator_data: &PreparedOperatorData<F>,
) -> Result<[FieldBuffer<P>; SHIFT_VARIANT_COUNT], Error> {
    let acc_size: usize = SHIFT_VARIANT_COUNT << (LOG_LEN.saturating_sub(P::LOG_WIDTH));

    assert!(
        P::WIDTH <= 8,
        "the optimizations below work only when the width of `P` is less than 8 (which is true for all packed 128b fields we use for now)"
    );

    // Field element that is represented by all ones
    let all_ones_f = F::from_underlier(UnderlierWithBitOps::fill_with_bit(1));
    // Map from u8 with `P::WIDTH` meaningful bits to the `P` where each element has
    // all zeroes or all ones depending on the corresponding bit value.
    let packed_masks_map = (0..1 << P::WIDTH)
        .map(|i| {
            let mut mask = P::zero();
            for bit_index in 0..P::WIDTH {
                if (i >> bit_index) & 1 == 1 {
                    mask.set(bit_index, all_ones_f);
                }
            }
            mask.to_underlier()
        })
        .collect::<Vec<_>>();
    // A mask for low `P::WIDTH` bits.
    let low_bits_mask = (1u8 << P::WIDTH) - 1;

    let multilinears = words
        .par_iter()
        .zip(key_collection.key_ranges.par_iter())
        .fold(
            || zeroed_vec::<P>(acc_size).into_boxed_slice(),
            |mut multilinears, (word, Range { start, end })| {
                let keys = &key_collection.keys[*start as usize..*end as usize];

                for key in keys {
                    let operator_data = match key.operation {
                        Operation::BitwiseAnd => bitand_operator_data,
                        Operation::IntegerMul => intmul_operator_data,
                    };

                    let acc = key.accumulate(&key_collection.constraint_indices, operator_data);
                    let acc_underlier = P::broadcast(acc).to_underlier();

                    // The following loop is an optimized version of the following
                    // for i in 0..WORD_SIZE_BITS {
                    //     if get_bit(word, i) {
                    //         values[start + i] += acc;
                    //     }
                    // }
                    let start = key.id as usize * (WORD_SIZE_BITS >> P::LOG_WIDTH);
                    let word_bytes = word.0.to_le_bytes();
                    for (&byte, values) in word_bytes.iter().zip(
                        multilinears[start..start + (WORD_SIZE_BITS >> P::LOG_WIDTH)]
                            .chunks_exact_mut(WORD_SIZE_BYTES >> P::LOG_WIDTH),
                    ) {
                        for value_index in 0..(8 >> P::LOG_WIDTH) {
                            unsafe {
                                let packed_mask_index =
                                    ((byte >> (value_index * P::WIDTH)) & low_bits_mask) as usize;

                                // Safety:
                                // - `packed_masks_map` is guaranteed to have enough elements to be
                                //   indexed with a `P::WIDTH`-bits value.
                                let packed_mask =
                                    *packed_masks_map.get_unchecked(packed_mask_index);

                                // Safety:
                                // - `values` is guaranteed to be (8 >> P::LOG_WIDTH) elements long
                                //   due to the chunking
                                // - `value_index` is always in bounds because we iterate over 0..(8
                                //   >> P::LOG_WIDTH)
                                *values.get_unchecked_mut(value_index) +=
                                    P::from_underlier(packed_mask & acc_underlier);
                            }
                        }
                    }
                }

                multilinears
            },
        )
        .reduce(
            || zeroed_vec::<P>(acc_size).into_boxed_slice(),
            |mut acc, local| {
                izip!(acc.iter_mut(), local.iter()).for_each(|(acc, local)| {
                    *acc += *local;
                });
                acc
            },
        );

    let parts = build_multilinear_parts(&multilinears)?;
    Ok(parts)
}

/// Builds the multilinear parts for a single operation by combining its operand multilinears.
///
/// Takes the raw multilinears for all operands and shift variants of an operation,
/// applies lambda weighting to each operand, and combines them into parts.
/// Each operand of index `i` gets weighted by λ^(i+1).
#[instrument(skip_all, name = "build_multilinear_parts")]
fn build_multilinear_parts<P: PackedField>(
    multilinears: &[P],
) -> Result<[FieldBuffer<P>; SHIFT_VARIANT_COUNT], Error> {
    assert!(
        P::LOG_WIDTH < LOG_LEN,
        "P::WIDTH is not supposed to exceed 8, so this statement must hold"
    );

    let parts = multilinears
        .chunks(1 << (LOG_LEN - P::LOG_WIDTH))
        .map(|chunk| FieldBuffer::new(LOG_LEN, chunk.to_vec().into_boxed_slice()))
        .collect::<Vec<_>>()
        .try_into()
        .expect("chunk has SHIFT_VARIANT_COUNT parts of size 1 << LOG_LEN");

    Ok(parts)
}
