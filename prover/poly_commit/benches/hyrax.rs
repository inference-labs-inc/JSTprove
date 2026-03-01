use std::hint::black_box;

use arith::{Field, Fr};
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gkr_engine::Transcript;
use gkr_hashers::Keccak256hasher;
use halo2curves::bn256::G1Affine;
use poly_commit::{HyraxPCS, PolynomialCommitmentScheme};
use polynomials::MultiLinearPoly;
use transcript::BytesHashTranscript;

fn hyrax_committing_benchmark_helper(
    c: &mut Criterion,
    lowest_num_vars: usize,
    highest_num_vars: usize,
) {
    let mut group = c.benchmark_group("Hyrax PCS committing");

    let mut rng = test_rng();
    let mut scratch_pad = ();

    for num_vars in lowest_num_vars..=highest_num_vars {
        let poly = MultiLinearPoly::<Fr>::random(num_vars, &mut rng);

        let (srs, _) = HyraxPCS::<G1Affine>::gen_srs_for_testing(&num_vars, &mut rng);

        group
            .bench_function(
                BenchmarkId::new(format!("{num_vars} variables"), num_vars),
                |b| {
                    b.iter(|| {
                        _ = black_box(HyraxPCS::<G1Affine>::commit(
                            &num_vars,
                            &srs,
                            &poly,
                            &mut scratch_pad,
                        ))
                    })
                },
            )
            .sample_size(10);
    }
}

fn hyrax_committing_benchmark(c: &mut Criterion) {
    hyrax_committing_benchmark_helper(c, 8, 15)
}

fn hyrax_opening_benchmark_helper(
    c: &mut Criterion,
    lowest_num_vars: usize,
    highest_num_vars: usize,
) {
    let mut group = c.benchmark_group("Hyrax PCS opening");

    let mut rng = test_rng();
    let mut transcript = BytesHashTranscript::<Keccak256hasher>::new();
    let mut scratch_pad = ();

    for num_vars in lowest_num_vars..=highest_num_vars {
        let poly = MultiLinearPoly::<Fr>::random(num_vars, &mut rng);

        let (srs, _) = HyraxPCS::<G1Affine>::gen_srs_for_testing(&num_vars, &mut rng);
        let eval_point: Vec<_> = (0..num_vars).map(|_| Fr::random_unsafe(&mut rng)).collect();

        let _ = HyraxPCS::<G1Affine>::commit(&num_vars, &srs, &poly, &mut scratch_pad);

        group
            .bench_function(
                BenchmarkId::new(format!("{num_vars} variables"), num_vars),
                |b| {
                    b.iter(|| {
                        _ = black_box(HyraxPCS::<G1Affine>::open(
                            &num_vars,
                            &srs,
                            &poly,
                            &eval_point,
                            &scratch_pad,
                            &mut transcript,
                        ))
                    })
                },
            )
            .sample_size(10);
    }
}

fn hyrax_opening_benchmark(c: &mut Criterion) {
    hyrax_opening_benchmark_helper(c, 8, 15)
}

criterion_group!(bench, hyrax_committing_benchmark, hyrax_opening_benchmark);
criterion_main!(bench);
