// this module benchmarks the performance of different field operations

use arith::bench_field;
use criterion::{criterion_group, criterion_main, Criterion};
use halo2curves::bn256::Fr;

fn criterion_benchmark(c: &mut Criterion) {
    bench_field::<Fr>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
