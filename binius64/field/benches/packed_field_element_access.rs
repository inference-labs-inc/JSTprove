// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{
    PackedField, Random,
    arch::{packed_128::*, packed_256::*, packed_512::*},
};
use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use rand::distr::{Distribution, Uniform};

const BATCH_SIZE: usize = 32;

fn benchmark_get_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
    let mut rng = rand::rng();
    let value = P::random(&mut rng);
    let distr = Uniform::<usize>::new(0, P::WIDTH).expect("Failed to create uniform distribution");
    let indices = array::from_fn::<_, BATCH_SIZE, _>(|_| distr.sample(&mut rng));

    group.throughput(Throughput::Elements(BATCH_SIZE as _));
    group.bench_function(id, |b| b.iter(|| indices.map(|i| value.get(i))));
}

fn benchmark_set_impl<P: PackedField>(group: &mut BenchmarkGroup<'_, WallTime>, id: &str) {
    let mut rng = rand::rng();
    let mut value = P::random(&mut rng);
    let distr = Uniform::<usize>::new(0, P::WIDTH).expect("Failed to create uniform distribution");
    let indices_values = array::from_fn::<_, BATCH_SIZE, _>(|_| {
        (distr.sample(&mut rng), P::Scalar::random(&mut rng))
    });

    group.throughput(Throughput::Elements(BATCH_SIZE as _));
    group.bench_function(id, |b| {
        b.iter(|| {
            indices_values
                .iter()
                .for_each(|(i, val)| value.set(*i, *val));

            value
        })
    });
}

macro_rules! benchmark_get_set {
    ($field:ty, $g:ident) => {
        benchmark_get_impl::<$field>(&mut $g, &format!("{}/get", stringify!($field)));
        benchmark_set_impl::<$field>(&mut $g, &format!("{}/set", stringify!($field)));
    };
}

fn packed_128(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_128");

    benchmark_get_set!(PackedBinaryField128x1b, group);
}

fn packed_256(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_256");

    benchmark_get_set!(PackedBinaryField256x1b, group);
}

fn packed_512(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_512");

    benchmark_get_set!(PackedBinaryField512x1b, group);
}

criterion_group!(get_set, packed_128, packed_256, packed_512,);
criterion_main!(get_set);
