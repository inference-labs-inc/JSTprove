// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField128bGhash as Ghash, Random, arch::OptimalPackedB128};
use binius_math::batch_invert::BatchInversion;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn bench_batch_inversion_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("BatchInversion/Scalar");
    let mut rng = rand::rng();

    for n in [1, 4, 6, 64, 96, 256, 384] {
        group.throughput(Throughput::Elements(n as u64));
        let mut elements: Vec<Ghash> = (0..n)
            .map(|_| <Ghash as Random>::random(&mut rng))
            .collect();
        let mut inverter = BatchInversion::<Ghash>::new(n);
        group.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                inverter.invert_or_zero(&mut elements);
            })
        });
    }

    group.finish();
}

fn bench_batch_inversion_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("BatchInversion/OptimalPackedB128");
    let mut rng = rand::rng();

    // n is the number of packed elements; total scalars = n * WIDTH
    for n in [1, 4, 6, 64, 96, 256, 384] {
        let total_scalars = n * OptimalPackedB128::WIDTH;
        group.throughput(Throughput::Elements(total_scalars as u64));
        let mut elements: Vec<OptimalPackedB128> = (0..n)
            .map(|_| <OptimalPackedB128 as Random>::random(&mut rng))
            .collect();
        let mut inverter = BatchInversion::<OptimalPackedB128>::new(n);
        group.bench_function(format!("{n}x{}", OptimalPackedB128::WIDTH), |b| {
            b.iter(|| inverter.invert_nonzero(&mut elements))
        });
    }

    group.finish();
}

criterion_group!(
    batch_invert_bench,
    bench_batch_inversion_scalar,
    bench_batch_inversion_packed
);
criterion_main!(batch_invert_bench);
