use arith::{bench_ext_field, bench_field};
use criterion::{criterion_group, criterion_main, Criterion};
use goldilocks::{Goldilocks, GoldilocksExt2, GoldilocksExt2x8, Goldilocksx8};

fn ext_by_base_benchmark(c: &mut Criterion) {
    bench_ext_field::<GoldilocksExt2>(c);
    bench_ext_field::<GoldilocksExt2x8>(c);
}

fn field_benchmark(c: &mut Criterion) {
    bench_field::<Goldilocks>(c);
    bench_field::<Goldilocksx8>(c);
    bench_field::<GoldilocksExt2>(c);
    bench_field::<GoldilocksExt2x8>(c);
}

criterion_group!(bench, ext_by_base_benchmark, field_benchmark);
criterion_main!(bench);
