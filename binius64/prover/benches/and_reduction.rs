// Copyright 2025 Irreducible Inc.
use std::{iter, iter::repeat_with};

use binius_core::word::Word;
use binius_field::{AESTowerField8b, Field, PackedAESBinaryField16x8b, Random};
use binius_math::{
    BinarySubspace, multilinear::eq::eq_ind_partial_eval, univariate::extrapolate_over_subspace,
};
use binius_prover::{
    and_reduction::{
        fold_lookup::FoldLookup, prover_setup::ntt_lookup_from_prover_message_domain,
        sumcheck_round_messages::univariate_round_message_extension_domain,
        utils::multivariate::OneBitOblongMultilinear,
    },
    protocols::sumcheck::{common::SumcheckProver, quadratic_mle::QuadraticMleCheckProver},
};
use binius_verifier::{
    and_reduction::utils::constants::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS},
    config::B128,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn bench(c: &mut Criterion) {
    let log_num_rows = 27;
    let mut rng = StdRng::seed_from_u64(0);
    let big_field_zerocheck_challenges =
        vec![B128::random(&mut rng); log_num_rows - SKIPPED_VARS - 3];
    let small_field_zerocheck_challenges = [
        AESTowerField8b::new(2),
        AESTowerField8b::new(4),
        AESTowerField8b::new(16),
    ];
    let first_mlv = OneBitOblongMultilinear {
        log_num_rows,
        packed_evals: repeat_with(|| Word(rng.random()))
            .take(1 << (log_num_rows - SKIPPED_VARS))
            .collect(),
    };

    let second_mlv = OneBitOblongMultilinear {
        log_num_rows,
        packed_evals: repeat_with(|| Word(rng.random()))
            .take(1 << (log_num_rows - SKIPPED_VARS))
            .collect(),
    };

    let third_mlv = OneBitOblongMultilinear {
        log_num_rows,
        packed_evals: iter::zip(&first_mlv.packed_evals, &second_mlv.packed_evals)
            .map(|(&a, &b)| a & b)
            .collect(),
    };

    let prover_message_domain = BinarySubspace::with_dim(SKIPPED_VARS + 1);

    let univariate_domain = prover_message_domain
        .reduce_dim(prover_message_domain.dim() - 1)
        .isomorphic();

    let ntt_lookup = ntt_lookup_from_prover_message_domain::<PackedAESBinaryField16x8b>(
        prover_message_domain.clone(),
    );

    let mut group = c.benchmark_group("evaluate");
    group.throughput(Throughput::Elements(1 << (log_num_rows - SKIPPED_VARS)));

    group.bench_function("NTT lookup precompute", |bench| {
        bench.iter(|| {
            ntt_lookup_from_prover_message_domain::<PackedAESBinaryField16x8b>(
                prover_message_domain.clone(),
            )
        });
    });

    group.bench_function(
        format!("univariate_round_message 2^{log_num_rows}"),
        |bench| {
            bench.iter(|| {
                let eq_ind_mle = eq_ind_partial_eval(&big_field_zerocheck_challenges);

                let urm: [B128; _] = univariate_round_message_extension_domain(
                    &first_mlv,
                    &second_mlv,
                    &third_mlv,
                    &eq_ind_mle,
                    &ntt_lookup,
                    &small_field_zerocheck_challenges,
                );

                urm
            });
        },
    );

    group.bench_function(format!("full_univariate_round 2^{log_num_rows}"), |bench| {
        bench.iter(|| {
            let eq_ind_mle = eq_ind_partial_eval(&big_field_zerocheck_challenges);

            let urm: [B128; _] = univariate_round_message_extension_domain(
                &first_mlv,
                &second_mlv,
                &third_mlv,
                &eq_ind_mle,
                &ntt_lookup,
                &small_field_zerocheck_challenges,
            );

            let fold_lookup =
                FoldLookup::<_, SKIPPED_VARS>::new(&univariate_domain, B128::random(&mut rng));

            (
                urm,
                first_mlv.fold(&fold_lookup),
                second_mlv.fold(&fold_lookup),
                third_mlv.fold(&fold_lookup),
            )
        });
    });

    group.bench_function(format!("full zerocheck 2^{log_num_rows}"), |bench| {
        bench.iter(|| {
            let eq_ind_only_big = eq_ind_partial_eval(&big_field_zerocheck_challenges);

            let urm = univariate_round_message_extension_domain(
                &first_mlv,
                &second_mlv,
                &third_mlv,
                &eq_ind_only_big,
                &ntt_lookup,
                &small_field_zerocheck_challenges,
            );

            let first_sumcheck_challenge = B128::random(&mut rng);

            let fold_lookup =
                FoldLookup::<_, SKIPPED_VARS>::new(&univariate_domain, first_sumcheck_challenge);

            let mut univariate_message_coeffs = vec![B128::ZERO; 2 * ROWS_PER_HYPERCUBE_VERTEX];

            univariate_message_coeffs[ROWS_PER_HYPERCUBE_VERTEX..2 * ROWS_PER_HYPERCUBE_VERTEX]
                .copy_from_slice(&urm);

            let next_round_claim = extrapolate_over_subspace(
                &prover_message_domain.clone().isomorphic::<B128>(),
                &univariate_message_coeffs,
                first_sumcheck_challenge,
            );

            let upcasted_small_field_challenges: Vec<_> = small_field_zerocheck_challenges
                .into_iter()
                .map(B128::from)
                .collect();

            let multilinear_zerocheck_challenges: Vec<_> = upcasted_small_field_challenges
                .iter()
                .chain(big_field_zerocheck_challenges.iter())
                .copied()
                .collect();

            let proving_polys = [
                first_mlv.fold(&fold_lookup),
                second_mlv.fold(&fold_lookup),
                third_mlv.fold(&fold_lookup),
            ];

            let mut prover = QuadraticMleCheckProver::new(
                proving_polys,
                |[a, b, c]| a * b - c,
                |[a, b, _]| a * b,
                multilinear_zerocheck_challenges.clone(),
                next_round_claim,
            )
            .expect("multilinears should have consistent dimensions");

            for _ in multilinear_zerocheck_challenges {
                let _ = prover.execute().unwrap();
                prover.fold(B128::random(&mut rng)).unwrap();
            }

            prover.finish().unwrap()
        });
    });
}

criterion_group!(univariate_round, bench);
criterion_main!(univariate_round);
