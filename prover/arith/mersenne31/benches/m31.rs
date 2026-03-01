use arith::{bench_ext_field, bench_fft, bench_field};
use criterion::{criterion_group, criterion_main, Criterion};
use mersenne31::{M31Ext3, M31Ext3x16, M31Ext6, M31x16, M31};

fn ext_by_base_benchmark(c: &mut Criterion) {
    bench_ext_field::<M31Ext3>(c);
    bench_ext_field::<M31Ext3x16>(c);
}

fn field_benchmark(c: &mut Criterion) {
    bench_field::<M31>(c);
    bench_field::<M31x16>(c);
    bench_field::<M31Ext3>(c);
    bench_field::<M31Ext3x16>(c);
    bench_fft::<M31Ext6>(c)
}

criterion_group!(bench, ext_by_base_benchmark, field_benchmark);
criterion_main!(bench);
