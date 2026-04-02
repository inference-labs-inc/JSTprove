// Copyright 2025 Irreducible Inc.
use std::array;

use binius_field::{
    BinaryField, Field, PackedBinaryField128x1b, PackedExtension, packed::get_packed_slice,
};
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval};
use binius_utils::rayon::prelude::*;
use binius_verifier::{and_reduction::utils::constants::ROWS_PER_HYPERCUBE_VERTEX, config::B1};
use bytemuck::must_cast_ref;
use itertools::izip;

use super::{ntt_lookup::NTTLookup, utils::multivariate::OneBitOblongMultilinear};

/// Generates a univariate polynomial for the sumcheck protocol in AND constraint reduction.
///
/// Let our oblong polynomials be A(Z, X₀, ...), B(Z, X₀, ...), and C(Z, X₀, ...)
///
/// Let our zerocheck challenges be (r₀, ...)
///
/// It turns out that the first k zerocheck challenges can actually be deterministic, since our
/// polynomials have 1-bit coefficients as long as their tensor product expansion is an
/// F2-linearly-independent set.
///
/// Note: Deterministic here means that the first k zerocheck challenges are a compile-time
/// agreed-upon parameter to the proof, and not sampled randomly by the verifier
///
///
/// We choose k=3 because we want them to be in a field isomorphic to the 8-bit NTT domain field
///
/// Computes a univariate polynomial:
/// R₀(Z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A·B - C)·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
/// This is zero at every point on the hypercube IFF A*B-C evaluates to zero at (r₀,...,rₙ₋₁)
/// for every Z on the univariate domain. Since R₀(Z) is 0 on the univariate domain, the prover
/// sends only enough values such that the verifier learns a domain of evaluations of size >
/// deg(R₀(Z))
///
/// # Arguments
///
/// * `first_col` - First multiplicand (a) as a one-bit oblong multilinear polynomial
/// * `second_col` - Second multiplicand (b) as a one-bit oblong multilinear polynomial
/// * `third_col` - Product constraint (c) as a one-bit oblong multilinear polynomial
/// * `eq_ind_big_field_challenges` - Partial equality indicator evaluations for big field variables
/// * `prover_message_domain_with_ntt_lookup` - Domain and NTT lookup tables for efficient
///   computation
/// * `small_field_zerocheck_challenges` - Zerocheck challenges for the small field variables (3
///   vars). These are a parameter to the proof, agreed on ahead of time by prover and verifier.
///   Their tensor product expansion must be an F2-basis of PNTTDomain::Scalar
/// * `univariate_zerocheck_challenge` - Zerocheck challenge for the univariate variable
///
/// # Returns
///
/// The evaluations of R₀(Z), a univariate polynomial of degree at most 2*(|D| - 1) where |D| is the
/// domain size, on another, disjoint |D|-sized domain. This allows the verifier to construct R₀(Z),
/// since it must equal zero on D.
///
/// # Type Parameters
///
/// * `FChallenge` - The challenge field type (must be a binary field)
/// * `PNTTDomain` - The packed extension field type for NTT operations (width must be 16)
pub fn univariate_round_message_extension_domain<FChallenge, PNTTDomain>(
    first_col: &OneBitOblongMultilinear,
    second_col: &OneBitOblongMultilinear,
    third_col: &OneBitOblongMultilinear,
    eq_ind_big_field_challenges: &FieldBuffer<FChallenge>,
    ntt_lookup: &NTTLookup<PNTTDomain>,
    small_field_zerocheck_challenges: &[PNTTDomain::Scalar],
) -> [FChallenge; ROWS_PER_HYPERCUBE_VERTEX]
where
    FChallenge: Field + From<PNTTDomain::Scalar> + BinaryField,
    PNTTDomain: PackedExtension<B1, PackedSubfield = PackedBinaryField128x1b>,
    PNTTDomain::Scalar: BinaryField,
    u8: From<PNTTDomain::Scalar>,
{
    // This assertion is used as a workaround for Rust's limited support for const-generics,
    // ideally we would just use PNTTDomain::WIDTH everywhere instead, but since this function only
    // supports 128b underliers, this is set to a constant. We would need to rethink for support of
    // multiple underlier sizes
    assert_eq!(PNTTDomain::WIDTH, 16);

    let eq_ind_small: Vec<PNTTDomain> = eq_ind_partial_eval(small_field_zerocheck_challenges)
        .as_ref()
        .iter()
        .map(|&item| PNTTDomain::broadcast(item))
        .collect();

    // Accumulate resulting polynomial evals by iterating over each hypercube vertex
    let chunk_size = eq_ind_small.len();
    (
        first_col.packed_evals.par_chunks(chunk_size),
        second_col.packed_evals.par_chunks(chunk_size),
        third_col.packed_evals.par_chunks(chunk_size),
        eq_ind_big_field_challenges.as_ref(),
    )
        .into_par_iter()
        .map(|(a_chunk, b_chunk, c_chunk, &eq_weight)| {
            let mut summed_ntt = [PNTTDomain::zero(); ROWS_PER_HYPERCUBE_VERTEX / 16];
            let lookup = ntt_lookup.get_lookup();

            for (a_i, b_i, c_i, &weight) in izip!(a_chunk, b_chunk, c_chunk, &eq_ind_small) {
                let col_1_bytes = must_cast_ref::<_, [u8; 8]>(&a_i.0);
                let col_2_bytes = must_cast_ref::<_, [u8; 8]>(&b_i.0);
                let col_3_bytes = must_cast_ref::<_, [u8; 8]>(&c_i.0);

                // In this cycle, we compute the NTT for each column using the lookup table.
                // We are not using the `NTTLookup::ntt` method directly for performance reasons.
                for (summed_ntt, lookup) in izip!(&mut summed_ntt, lookup) {
                    let mut first_col_ntt = PNTTDomain::zero();
                    let mut second_col_ntt = PNTTDomain::zero();
                    let mut third_col_ntt = PNTTDomain::zero();

                    for byte_index in 0..8 {
                        first_col_ntt += lookup[col_1_bytes[byte_index] as usize][byte_index];
                        second_col_ntt += lookup[col_2_bytes[byte_index] as usize][byte_index];
                        third_col_ntt += lookup[col_3_bytes[byte_index] as usize][byte_index];
                    }

                    *summed_ntt += (first_col_ntt * second_col_ntt - third_col_ntt) * weight;
                }
            }

            array::from_fn::<_, ROWS_PER_HYPERCUBE_VERTEX, _>(|i| {
                eq_weight * FChallenge::from(get_packed_slice(&summed_ntt, i))
            })
        })
        .reduce(
            || [FChallenge::ZERO; ROWS_PER_HYPERCUBE_VERTEX],
            |mut acc, array_to_add| {
                for (i, val) in array_to_add.into_iter().enumerate() {
                    acc[i] += val;
                }
                acc
            },
        )
}

#[cfg(test)]
mod test {
    use std::iter::repeat_with;

    use binius_core::word::Word;
    use binius_field::{AESTowerField8b, Field, PackedAESBinaryField16x8b, Random};
    use binius_math::{
        BinarySubspace, FieldBuffer, multilinear::eq::eq_ind_partial_eval,
        univariate::extrapolate_over_subspace,
    };
    use binius_verifier::{
        and_reduction::utils::constants::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS},
        config::{B128, LOG_WORD_SIZE_BITS},
    };
    use itertools::izip;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::univariate_round_message_extension_domain;
    use crate::and_reduction::{
        fold_lookup::FoldLookup, prover_setup::ntt_lookup_from_prover_message_domain,
        utils::multivariate::OneBitOblongMultilinear,
    };

    fn random_one_bit_multivariate(
        log_num_rows: usize,
        mut rng: impl Rng,
    ) -> OneBitOblongMultilinear {
        OneBitOblongMultilinear {
            log_num_rows,
            packed_evals: repeat_with(|| Word(rng.random()))
                .take(1 << (log_num_rows - LOG_WORD_SIZE_BITS))
                .collect(),
        }
    }

    // Sends the sum claim from first multilinear round (second overall round)
    pub fn sum_claim<BF: Field + From<B128>>(
        first_col: &FieldBuffer<BF>,
        second_col: &FieldBuffer<BF>,
        third_col: &FieldBuffer<BF>,
        eq_ind: &FieldBuffer<BF>,
    ) -> BF {
        izip!(
            first_col.as_ref(),
            second_col.as_ref(),
            third_col.as_ref(),
            eq_ind.as_ref()
        )
        .map(|(a, b, c, eq)| (*a * *b - *c) * *eq)
        .sum()
    }

    #[test]
    fn test_first_round_message_matches_next_round_sum_claim() {
        // Setup
        let log_num_rows = 10;
        let mut rng = StdRng::from_seed([0; 32]);

        let big_field_zerocheck_challenges =
            vec![B128::random(&mut rng); (log_num_rows - SKIPPED_VARS - 3) + 1];

        let small_field_zerocheck_challenges = [
            AESTowerField8b::new(2),
            AESTowerField8b::new(4),
            AESTowerField8b::new(16),
        ];

        let mlv_1 = random_one_bit_multivariate(log_num_rows, &mut rng);
        let mlv_2 = random_one_bit_multivariate(log_num_rows, &mut rng);
        let mlv_3 = OneBitOblongMultilinear {
            log_num_rows,
            packed_evals: (0..1 << (log_num_rows - LOG_WORD_SIZE_BITS))
                .map(|i| mlv_1.packed_evals[i] & mlv_2.packed_evals[i])
                .collect(),
        };

        let eq_ind_only_big = eq_ind_partial_eval(&big_field_zerocheck_challenges);

        // Agreed-upon proof parameter

        let prover_message_domain = BinarySubspace::with_dim(SKIPPED_VARS + 1);
        let ntt_lookup = ntt_lookup_from_prover_message_domain::<PackedAESBinaryField16x8b>(
            prover_message_domain.clone(),
        );

        let verifier_message_domain = prover_message_domain.isomorphic::<B128>();

        // Prover generates first round message
        let first_round_message_on_ext_domain = univariate_round_message_extension_domain::<B128, _>(
            &mlv_1,
            &mlv_2,
            &mlv_3,
            &eq_ind_only_big,
            &ntt_lookup,
            &small_field_zerocheck_challenges,
        );

        let mut first_round_message_coeffs = vec![B128::ZERO; 2 * ROWS_PER_HYPERCUBE_VERTEX];

        first_round_message_coeffs[ROWS_PER_HYPERCUBE_VERTEX..2 * ROWS_PER_HYPERCUBE_VERTEX]
            .copy_from_slice(&first_round_message_on_ext_domain);

        // Verifier checks the accuracy of the message by challenging the prover and folding
        // polynomials transparently

        let verifier_input_domain: BinarySubspace<B128> =
            verifier_message_domain.reduce_dim(verifier_message_domain.dim() - 1);

        let first_sumcheck_challenge = B128::random(&mut rng);
        let expected_next_round_sum = extrapolate_over_subspace(
            &verifier_message_domain,
            &first_round_message_coeffs,
            first_sumcheck_challenge,
        );

        let lookup =
            FoldLookup::<_, SKIPPED_VARS>::new(&verifier_input_domain, first_sumcheck_challenge);

        let folded_first_mle = mlv_1.fold(&lookup);
        let folded_second_mle = mlv_2.fold(&lookup);
        let folded_third_mle = mlv_3.fold(&lookup);

        let upcasted_small_field_challenges: Vec<_> = small_field_zerocheck_challenges
            .into_iter()
            .map(B128::from)
            .collect();

        let verifier_field_zerocheck_challenges: Vec<_> = upcasted_small_field_challenges
            .iter()
            .chain(big_field_zerocheck_challenges.iter())
            .copied()
            .collect();

        let verifier_field_eq = eq_ind_partial_eval(&verifier_field_zerocheck_challenges);
        let actual_next_round_sum = sum_claim(
            &folded_first_mle,
            &folded_second_mle,
            &folded_third_mle,
            &verifier_field_eq,
        );

        assert_eq!(expected_next_round_sum, actual_next_round_sum);
    }
}
