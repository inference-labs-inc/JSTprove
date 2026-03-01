use arith::{bench_ext_field, bench_field};
use criterion::{criterion_group, criterion_main, Criterion};
use gf2_128::{GF2_128x8, GF2_128};

fn ext_by_base_benchmark(c: &mut Criterion) {
    bench_ext_field::<GF2_128>(c);
    bench_ext_field::<GF2_128x8>(c);
}

fn field_benchmark(c: &mut Criterion) {
    bench_field::<GF2_128>(c);
    bench_field::<GF2_128x8>(c);
}

criterion_group!(bench, ext_by_base_benchmark, field_benchmark);
criterion_main!(bench);
