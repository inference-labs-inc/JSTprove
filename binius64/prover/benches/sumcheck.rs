// Copyright 2025 Irreducible Inc.

use binius_field::{arch::OptimalPackedB128, packed::PackedField};
use binius_math::{
    inner_product::inner_product_par,
    test_utils::{random_field_buffer, random_scalars},
};
use binius_prover::protocols::sumcheck::{
    bivariate_product::BivariateProductSumcheckProver, prove_single, prove_single_mlecheck,
    quadratic_mle::QuadraticMleCheckProver,
};
use binius_transcript::ProverTranscript;
use binius_utils::rayon::prelude::*;
use binius_verifier::config::StdChallenger;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, prelude::StdRng};

type P = OptimalPackedB128;

fn bench_sumcheck_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck/bivariate_product");

    // Test different sizes of multilinear polynomials
    for n_vars in [12, 16, 20] {
        // Consider each element to be one hypercube vertex.
        group.throughput(Throughput::Elements(1 << n_vars));
        group.bench_function(format!("n_vars={n_vars}"), |b| {
            // Setup phase - prepare the multilinears and compute the sum
            let mut rng = StdRng::seed_from_u64(0);
            let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
            let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);
            let sum = inner_product_par(&multilinear_a, &multilinear_b);

            let mut transcript = ProverTranscript::new(StdChallenger::default());

            // Benchmark only the proving phase
            b.iter_batched(
                || [multilinear_a.clone(), multilinear_b.clone()],
                |multilinears| {
                    let prover = BivariateProductSumcheckProver::new(multilinears, sum).unwrap();
                    prove_single(prover, &mut transcript).unwrap()
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_mlecheck_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlecheck");

    let mut rng = StdRng::seed_from_u64(0);

    // Test different sizes of multilinear polynomials
    for n_vars in [12, 16, 20] {
        // Consider each element to be one hypercube vertex.
        group.throughput(Throughput::Elements(1 << n_vars));
        group.bench_function(format!("A*B/n_vars={n_vars}"), |b| {
            let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
            let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

            let eval_point = random_scalars(&mut rng, n_vars);
            let eval_claim = inner_product_par(&multilinear_a, &multilinear_b);

            let mut transcript = ProverTranscript::new(StdChallenger::default());

            // Benchmark only the proving phase
            b.iter_batched(
                || [multilinear_a.clone(), multilinear_b.clone()],
                |multilinears| {
                    let prover = QuadraticMleCheckProver::new(
                        multilinears,
                        |[a, b]| a * b,
                        |[a, b]| a * b,
                        eval_point.clone(),
                        eval_claim,
                    )
                    .unwrap();

                    prove_single_mlecheck(prover, &mut transcript).unwrap()
                },
                BatchSize::SmallInput,
            );
        });

        // Benchmark mul gate composition: a * b - c
        group.bench_function(format!("A*B-C/n_vars={n_vars}"), |b| {
            let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
            let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);
            let multilinear_c = random_field_buffer::<P>(&mut rng, n_vars);

            let eval_point = random_scalars(&mut rng, n_vars);
            let eval_claim = (
                multilinear_a.as_ref(),
                multilinear_b.as_ref(),
                multilinear_c.as_ref(),
            )
                .into_par_iter()
                .map(|(&a_i, &b_i, &c_i)| a_i * b_i - c_i)
                .sum::<P>()
                .into_iter()
                .take(1 << n_vars)
                .sum();

            let mut transcript = ProverTranscript::new(StdChallenger::default());

            // Benchmark only the proving phase
            b.iter_batched(
                || {
                    [
                        multilinear_a.clone(),
                        multilinear_b.clone(),
                        multilinear_c.clone(),
                    ]
                },
                |multilinears| {
                    let prover = QuadraticMleCheckProver::new(
                        multilinears,
                        |[a, b, c]| a * b - c,
                        |[a, b, _c]| a * b,
                        eval_point.clone(),
                        eval_claim,
                    )
                    .unwrap();

                    prove_single_mlecheck(prover, &mut transcript).unwrap()
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(sumcheck, bench_sumcheck_prove, bench_mlecheck_prove);
criterion_main!(sumcheck);
