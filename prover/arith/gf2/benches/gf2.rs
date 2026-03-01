use arith::bench_field;
use criterion::{criterion_group, criterion_main, Criterion};
use gf2::GF2;

fn field_benchmark(c: &mut Criterion) {
    bench_field::<GF2>(c);
}

criterion_group!(bench, field_benchmark);
criterion_main!(bench);
