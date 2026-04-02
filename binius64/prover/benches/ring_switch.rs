// Copyright 2025 Irreducible Inc.

use std::mem::size_of;

use binius_field::{BinaryField, ExtensionField, arch::OptimalPackedB128};
use binius_math::test_utils::random_field_buffer;
use binius_prover::ring_switch::{
    fold_1b_rows_for_b128, fold_b128_elems_inplace, fold_elems_inplace,
};
use binius_utils::checked_arithmetics::log2_strict_usize;
use binius_verifier::config::{B1, B128};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

fn bench_fold_1b_rows_for_b128(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold_1b_rows_for_b128");

    let log_bits = log2_strict_usize(B128::N_BITS);
    for log_len in [12, 16] {
        const LOG_BITS_PER_BYTE: usize = 3;
        group.throughput(Throughput::Bytes(
            (1 << (log_len + log_bits - LOG_BITS_PER_BYTE)) as u64,
        ));
        group.bench_function(format!("log_len={log_len}"), |b| {
            let mut rng = rand::rng();

            let mat = random_field_buffer::<B128>(&mut rng, log_len);
            let vec = random_field_buffer::<B128>(&mut rng, log_len);

            b.iter(|| fold_1b_rows_for_b128(&mat, &vec));
        });
    }

    group.finish();
}

fn bench_fold_elems_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_switch/fold_elems_inplace");

    type P = OptimalPackedB128;

    for log_len in [12, 16] {
        // Calculate throughput based on the size of elems buffer in bytes
        let elem_bytes = (1 << log_len) * size_of::<P>();
        group.throughput(Throughput::Bytes(elem_bytes as u64));
        group.bench_function(format!("log_len={log_len}"), |b| {
            let mut rng = rand::rng();

            let elems = random_field_buffer::<P>(&mut rng, log_len);
            let vec =
                random_field_buffer::<B128>(&mut rng, <B128 as ExtensionField<B1>>::LOG_DEGREE);

            b.iter_batched(
                || elems.clone(),
                |elems| fold_elems_inplace(elems, &vec),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_fold_b128_elems_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_switch/fold_b128_elems_inplace");

    type P = OptimalPackedB128;

    for log_len in [12, 16] {
        // Calculate throughput based on the size of elems buffer in bytes
        let elem_bytes = (1 << log_len) * size_of::<P>();
        group.throughput(Throughput::Bytes(elem_bytes as u64));
        group.bench_function(format!("log_len={log_len}"), |b| {
            let mut rng = rand::rng();

            let elems = random_field_buffer::<P>(&mut rng, log_len);
            let vec =
                random_field_buffer::<B128>(&mut rng, <B128 as ExtensionField<B1>>::LOG_DEGREE);

            b.iter_batched(
                || elems.clone(),
                |elems| fold_b128_elems_inplace(elems, &vec),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    ring_switch,
    bench_fold_1b_rows_for_b128,
    bench_fold_elems_inplace,
    bench_fold_b128_elems_inplace
);
criterion_main!(ring_switch);
