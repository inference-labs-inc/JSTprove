// Copyright 2025 Irreducible Inc.

use std::{iter, ops::Range};

use binius_field::{AESTowerField8b, BinaryField, Field, PackedField};
use binius_math::{
    BinarySubspace, FieldBuffer, multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};
use binius_utils::{checked_arithmetics::strict_log_2, rayon::prelude::*};
use binius_verifier::{
    config::{LOG_WORD_SIZE_BITS, WORD_SIZE_BITS},
    protocols::shift::{BITAND_ARITY, INTMUL_ARITY, evaluate_h_op},
};
use bytemuck::zeroed_vec;
use tracing::instrument;

use super::{
    SHIFT_VARIANT_COUNT,
    error::Error,
    key_collection::{KeyCollection, Operation},
    prove::PreparedOperatorData,
};

/// Constructs the three "h" multilinear polynomials for shift operations at a
/// univariate challenge point. See the paper for definition of h polynomials.
///
/// There is one h multilinear for each shift variant. For each operation, there is one univariate
/// challenge `r_zhat_prime` at which to construct the h parts.
///
/// # Usage in Protocol
///
/// Used in phase 1, thus returning an array of multilinear evaluations.
#[instrument(skip_all, name = "build_h_parts")]
pub fn build_h_parts<F, P: PackedField<Scalar = F>>(
    r_zhat_prime: F,
) -> [FieldBuffer<P>; SHIFT_VARIANT_COUNT]
where
    F: BinaryField + From<AESTowerField8b>,
{
    let subspace = BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
    let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
    let l_tilde = l_tilde.as_ref();

    fn build_part<F: Field, P: PackedField<Scalar = F>>(
        fill: impl Fn(usize, &mut [F; WORD_SIZE_BITS]),
    ) -> FieldBuffer<P> {
        let mut data = zeroed_vec::<[F; WORD_SIZE_BITS]>(WORD_SIZE_BITS);
        for (s, chunk) in data.iter_mut().enumerate() {
            fill(s, chunk);
        }
        FieldBuffer::from_values(&data.into_flattened())
    }

    let sll = build_part(|s, sll_s| {
        sll_s[..WORD_SIZE_BITS - s].copy_from_slice(&l_tilde[s..]);
    });

    let srl = build_part(|s, srl_s| {
        srl_s[s..].copy_from_slice(&l_tilde[..WORD_SIZE_BITS - s]);
    });

    let sra = build_part(|s, sra_s| {
        sra_s[s..].copy_from_slice(&l_tilde[..WORD_SIZE_BITS - s]);
        sra_s[WORD_SIZE_BITS - 1] += l_tilde[WORD_SIZE_BITS - s..].iter().sum::<F>();
    });

    let rotr = build_part(|s, rotr_s| {
        rotr_s[..s].copy_from_slice(&l_tilde[WORD_SIZE_BITS - s..]);
        rotr_s[s..].copy_from_slice(&l_tilde[..WORD_SIZE_BITS - s]);
    });

    let sll32 = build_part(|s, sll32_s| {
        let s = s % 32;
        for (l_tilde_i, sll_s_i) in iter::zip(l_tilde.chunks(32), sll32_s.chunks_mut(32)) {
            sll_s_i[..32 - s].copy_from_slice(&l_tilde_i[s..]);
        }
    });

    let srl32 = build_part(|s, srl32_s| {
        let s = s % 32;
        for (l_tilde_i, srl32_s_i) in iter::zip(l_tilde.chunks(32), srl32_s.chunks_mut(32)) {
            srl32_s_i[s..].copy_from_slice(&l_tilde_i[..32 - s]);
        }
    });

    let sra32 = build_part(|s, sra32_s| {
        let s = s % 32;
        for (l_tilde_i, sra32_s_i) in iter::zip(l_tilde.chunks(32), sra32_s.chunks_mut(32)) {
            sra32_s_i[s..].copy_from_slice(&l_tilde_i[..32 - s]);
            sra32_s_i[32 - 1] += l_tilde_i[32 - s..].iter().sum::<F>();
        }
    });

    let rotr32 = build_part(|s, rotr32_s| {
        let s = s % 32;
        for (l_tilde_i, rotr32_s_i) in iter::zip(l_tilde.chunks(32), rotr32_s.chunks_mut(32)) {
            rotr32_s_i[..s].copy_from_slice(&l_tilde_i[32 - s..]);
            rotr32_s_i[s..].copy_from_slice(&l_tilde_i[..32 - s]);
        }
    });

    [sll, srl, sra, rotr, sll32, srl32, sra32, rotr32]
}

/// Constructs the "monster multilinear" that combines all shift operations into a single
/// multilinear.
///
/// This function builds a comprehensive multilinear polynomial that encapsulates both AND and MUL
/// constraints with their associated shift operations. For each witness word, it computes the
/// contribution from all constraints involving that word, weighted by the appropriate h-polynomial
/// evaluations and lambda powers.
///
/// # Construction Process
///
/// 1. **Compute lambda powers**: Powers λ^(i+1) for each operand index in both operations
/// 2. **Evaluate h-polynomials**: Compute h_op evaluations for SLL, SRL, SRA at challenge points
/// 3. **Build scalar matrix**: Create scalars combining lambda powers, h-evaluations, and r_s
///    tensor
/// 4. **Process keys in parallel**: For each word, accumulate contributions from all its
///    constraints
///
/// # Formula
///
/// For each word w, computes:
/// ```text
/// ∑_{key ∈ keys[w]} key.accumulate(constraint_indices, tensor) × scalars[key.id]
/// ```
/// where the scalars encode `λ^(operand_idx+1) × h_op[shift_variant] × r_s_tensor[shift_amount]`
/// for operand index `operand_idx` and `shift_variant` in {SLL, SRL, SRA} and `shift_amount` in
/// [0, WORD_SIZE_BITS).
///
/// # Usage
///
/// Used in phase 2 of the shift protocol where the prover needs a single multilinear combining
/// all shift-related constraints for efficient sumcheck computation.
#[instrument(skip_all, name = "build_monster_multilinear")]
pub fn build_monster_multilinear<F, P: PackedField<Scalar = F>>(
    key_collection: &KeyCollection,
    bitand_operator_data: &PreparedOperatorData<F>,
    intmul_operator_data: &PreparedOperatorData<F>,
    r_j: &[F],
    r_s: &[F],
) -> Result<FieldBuffer<P>, Error>
where
    F: BinaryField + From<AESTowerField8b>,
{
    // Compute h evaluations
    let subspace = BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
    let [bitand_h_ops, intmul_h_ops] = [
        bitand_operator_data.r_zhat_prime,
        intmul_operator_data.r_zhat_prime,
    ]
    .map(|r_zhat_prime| {
        let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
        evaluate_h_op(l_tilde.to_ref(), r_j, r_s)
    });

    let r_s_tensor = eq_ind_partial_eval::<F>(r_s);

    // Allocate and populate the scalars
    let mut bitand_scalars = vec![F::ZERO; BITAND_ARITY * SHIFT_VARIANT_COUNT * WORD_SIZE_BITS];
    let mut intmul_scalars = vec![F::ZERO; INTMUL_ARITY * SHIFT_VARIANT_COUNT * WORD_SIZE_BITS];

    let populate_scalars = |scalars: &mut [F], arity: usize, lambda_powers: &[F], h_ops: &[F]| {
        for operand_idx in 0..arity {
            for op in 0..SHIFT_VARIANT_COUNT {
                let operand_op_idx = operand_idx * SHIFT_VARIANT_COUNT + op;
                let operand_op_scalar = lambda_powers[operand_idx] * h_ops[op];
                for s in 0..WORD_SIZE_BITS {
                    let operand_op_s_idx = operand_op_idx * WORD_SIZE_BITS + s;
                    scalars[operand_op_s_idx] = operand_op_scalar * r_s_tensor.as_ref()[s];
                }
            }
        }
    };

    populate_scalars(
        &mut bitand_scalars,
        BITAND_ARITY,
        &bitand_operator_data.lambda_powers,
        &bitand_h_ops,
    );
    populate_scalars(
        &mut intmul_scalars,
        INTMUL_ARITY,
        &intmul_operator_data.lambda_powers,
        &intmul_h_ops,
    );

    let monster_multilinear = key_collection
        .key_ranges
        .par_chunks(P::WIDTH)
        .map(|chunk| {
            P::from_scalars(chunk.iter().map(|Range { start, end }| {
                key_collection.keys[*start as usize..*end as usize]
                    .iter()
                    .map(|key| {
                        let (operator_data, scalars) = match key.operation {
                            Operation::BitwiseAnd => (bitand_operator_data, &bitand_scalars),
                            Operation::IntegerMul => (intmul_operator_data, &intmul_scalars),
                        };
                        key.accumulate_by_operand(&key_collection.constraint_indices, operator_data)
                            .map(|(operand_index, acc)| {
                                let index = key.id as usize
                                    + operand_index * SHIFT_VARIANT_COUNT * WORD_SIZE_BITS;
                                acc * scalars[index]
                            })
                            .sum::<F>()
                    })
                    .sum()
            }))
        })
        .collect::<Box<[_]>>();

    let log_len = strict_log_2(key_collection.key_ranges.len())
        .expect("same length as constraint system's `key_ranges`");
    Ok(FieldBuffer::new(log_len, monster_multilinear))
}

#[cfg(test)]
mod tests {
    use binius_field::{BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
    use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
    use binius_verifier::protocols::shift::evaluate_h_op;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    /// Test consistency between direct multilinear evaluation of h
    /// multilinears and the succinct `evaluate_h_op` implementation
    #[test]
    fn h_op_consistency() {
        type F = BinaryField128bGhash;
        type P = PackedBinaryGhash2x128b;

        let mut rng = StdRng::seed_from_u64(0);

        let num_random_tests = 10;

        for test_case in 0..num_random_tests {
            let r_zhat_prime = F::random(&mut rng);

            let r_j: Vec<F> = (0..6).map(|_| F::random(&mut rng)).collect();
            let r_s: Vec<F> = (0..6).map(|_| F::random(&mut rng)).collect();

            // Method 1: Succinct evaluation using `evaluate_h_op`
            let subspace =
                BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
            let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
            let succinct_evaluations = evaluate_h_op(l_tilde.to_ref(), &r_j, &r_s);

            // Method 2: Direct evaluation via multilinear part
            let h_parts = build_h_parts(r_zhat_prime);
            let evaluation_point: Vec<F> = [r_j.clone(), r_s.clone()].concat();
            let tensor = eq_ind_partial_eval::<P>(&evaluation_point);
            let direct_evaluations = h_parts.map(|buf| inner_product_buffers(&buf, &tensor));

            assert_eq!(
                succinct_evaluations, direct_evaluations,
                "H-op evaluation mismatch (test_case={test_case}): succinct != direct",
            );
        }
    }
}
