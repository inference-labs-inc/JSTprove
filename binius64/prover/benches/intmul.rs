// Copyright 2025 Irreducible Inc.
use binius_field::PackedBinaryGhash1x128b;
use binius_prover::protocols::intmul::{prove::IntMulProver, witness::Witness};
use binius_transcript::ProverTranscript;
use binius_utils::rayon::ThreadPoolBuilder;
use binius_verifier::config::StdChallenger;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

type P = PackedBinaryGhash1x128b;

const LOG_BITS: usize = 6;

fn generate_test_data(log_num: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let num_exponents = 1 << log_num;
    let mut rng = StdRng::seed_from_u64(0);

    let mut a = Vec::with_capacity(num_exponents);
    let mut b = Vec::with_capacity(num_exponents);
    let mut c_lo = Vec::with_capacity(num_exponents);
    let mut c_hi = Vec::with_capacity(num_exponents);

    for _ in 0..num_exponents {
        let a_i = rng.random_range(1..u64::MAX);
        let b_i = rng.random_range(1..u64::MAX);

        let full_result = (a_i as u128) * (b_i as u128);

        let c_lo_i = full_result as u64;
        let c_hi_i = (full_result >> 64) as u64;

        a.push(a_i);
        b.push(b_i);
        c_lo.push(c_lo_i);
        c_hi.push(c_hi_i);
    }

    (a, b, c_lo, c_hi)
}

fn bench_intmul_prove(c: &mut Criterion) {
    ThreadPoolBuilder::new().num_threads(1).build_global().ok();

    let mut group = c.benchmark_group("intmul_phases");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));

    let log_num = 14;
    let num_exponents = 1 << log_num;
    let (a, b, c_lo, c_hi) = generate_test_data(log_num);

    group.bench_with_input(
        BenchmarkId::new("witness", num_exponents),
        &num_exponents,
        |bencher, _| {
            bencher.iter(|| Witness::<P, _, _>::new(LOG_BITS, &a, &b, &c_lo, &c_hi).unwrap())
        },
    );

    // prove
    let witness = Witness::<P, _, _>::new(LOG_BITS, &a, &b, &c_lo, &c_hi).unwrap();

    group.bench_with_input(
        BenchmarkId::new("prove", num_exponents),
        &witness,
        |bencher, witness| {
            bencher.iter_with_setup(
                || {
                    let prover_transcript = ProverTranscript::<StdChallenger>::default();
                    (witness.clone(), prover_transcript)
                },
                |(witness, mut prover_transcript)| {
                    let mut intmul_prover = IntMulProver::new(0, &mut prover_transcript);
                    intmul_prover.prove(witness).unwrap();
                },
            )
        },
    );

    // execute + prove
    group.bench_with_input(
        BenchmarkId::new("combined", num_exponents),
        &num_exponents,
        |bencher, _| {
            bencher.iter(|| {
                let witness = Witness::<P, _, _>::new(LOG_BITS, &a, &b, &c_lo, &c_hi).unwrap();

                let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
                let mut intmul_prover = IntMulProver::new(0, &mut prover_transcript);
                intmul_prover.prove(witness).unwrap();
            })
        },
    );

    group.finish();
}

criterion_group!(intmul_benches, bench_intmul_prove);
criterion_main!(intmul_benches);
