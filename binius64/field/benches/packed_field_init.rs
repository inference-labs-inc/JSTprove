// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{
    PackedField, Random,
    arch::{packed_128::*, packed_256::*, packed_512::*},
};
use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};

const BATCH_SIZE: usize = 32;

fn benchmark_get_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
    let mut rng = rand::rng();
    let values = array::from_fn::<_, BATCH_SIZE, _>(|_| {
        (0..P::WIDTH)
            .map(|_| P::Scalar::random(&mut rng))
            .collect::<Vec<_>>()
    });

    group.throughput(Throughput::Elements(P::WIDTH as _));
    group.bench_function(id, |b| {
        b.iter(|| {
            array::from_fn::<_, BATCH_SIZE, _>(|j| {
                let values = &values[j];
                P::from_fn(|i| values[i])
            })
        })
    });
}

macro_rules! benchmark_from_fn {
    ($field:ty, $g:ident) => {
        benchmark_get_impl::<$field>(&mut $g, &format!("{}/from_fn", stringify!($field)));
    };
}

fn packed_128(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_128");

    benchmark_from_fn!(PackedBinaryField128x1b, group);
}

fn packed_256(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_256");

    benchmark_from_fn!(PackedBinaryField256x1b, group);
}

fn packed_512(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_512");

    benchmark_from_fn!(PackedBinaryField512x1b, group);
}

criterion_group!(initialization, packed_128, packed_256, packed_512);
criterion_main!(initialization);
