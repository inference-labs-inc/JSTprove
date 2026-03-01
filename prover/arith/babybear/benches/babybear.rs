use arith::{bench_ext_field, bench_field};
use babybear::{BabyBear, BabyBearExt3, BabyBearExt3x16, BabyBearx16};
use criterion::{criterion_group, criterion_main, Criterion};

fn ext_by_base_benchmark(c: &mut Criterion) {
    bench_ext_field::<BabyBearExt3>(c);
    bench_ext_field::<BabyBearExt3x16>(c);
}

fn field_benchmark(c: &mut Criterion) {
    bench_field::<BabyBear>(c);
    bench_field::<BabyBearx16>(c);
    bench_field::<BabyBearExt3>(c);
    bench_field::<BabyBearExt3x16>(c);
}

criterion_group!(bench, ext_by_base_benchmark, field_benchmark);
criterion_main!(bench);
