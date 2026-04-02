// Copyright 2025 Irreducible Inc.

use binius_field::{
    BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b,
    PackedField,
};
use binius_math::{bit_reverse::bit_reverse_packed, test_utils::random_field_buffer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn bench_bit_reverse_helper<F: BinaryField, P: PackedField<Scalar = F>>(
    c: &mut Criterion,
    group_name: &str,
) {
    let mut group = c.benchmark_group(group_name);

    for log_d in [16, 20, 24] {
        let mut rng = rand::rng();

        let parameter = format!("log_d={log_d}");
        let throughput = Throughput::Bytes(((F::N_BITS / 8) << log_d) as u64);
        group.throughput(throughput);

        group.bench_function(BenchmarkId::new("bit_reverse_packed", &parameter), |b| {
            let mut data = random_field_buffer::<P>(&mut rng, log_d);
            b.iter(|| bit_reverse_packed(data.to_mut()))
        });
    }

    group.finish();
}

fn bench_bit_reverse(c: &mut Criterion) {
    bench_bit_reverse_helper::<_, PackedBinaryGhash1x128b>(c, "1xGhash");
    bench_bit_reverse_helper::<_, PackedBinaryGhash2x128b>(c, "2xGhash");
    bench_bit_reverse_helper::<_, PackedBinaryGhash4x128b>(c, "4xGhash");
}

criterion_group! {
    name = default;
    config = Criterion::default().sample_size(20).significance_level(0.01);
    targets = bench_bit_reverse
}
criterion_main!(default);
