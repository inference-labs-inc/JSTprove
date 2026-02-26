use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jstprove_remainder::runner::witness::WitnessData;
use serde::{Deserialize, Serialize};

fn make_witness_data(num_shreds: usize, shred_size: usize) -> WitnessData {
    let mut shreds = HashMap::new();
    let mut observed_n_bits = HashMap::new();
    for i in 0..num_shreds {
        let name = format!("layer_{i}");
        shreds.insert(name.clone(), vec![42i64; shred_size]);
        observed_n_bits.insert(name, 48);
    }
    WitnessData {
        shreds,
        observed_n_bits,
    }
}

#[derive(Serialize, Deserialize)]
struct BlobPayload {
    data: Vec<u8>,
    metadata: HashMap<String, usize>,
}

fn make_blob_payload(size: usize) -> BlobPayload {
    let mut metadata = HashMap::new();
    for i in 0..16 {
        metadata.insert(format!("key_{i}"), i * 100);
    }
    BlobPayload {
        data: vec![0xAB; size],
        metadata,
    }
}

fn bench_roundtrip<T: Serialize + for<'de> Deserialize<'de>>(
    c: &mut Criterion,
    name: &str,
    value: &T,
) {
    let bincode_bytes = bincode::serialize(value).unwrap();
    let msgpack_bytes = rmp_serde::to_vec_named(value).unwrap();

    let group_name = format!(
        "{name} (bincode={}, msgpack={})",
        bincode_bytes.len(),
        msgpack_bytes.len()
    );
    let mut group = c.benchmark_group(&group_name);

    group.bench_function("bincode_serialize", |b| {
        b.iter(|| bincode::serialize(black_box(value)).unwrap());
    });
    group.bench_function("msgpack_serialize", |b| {
        b.iter(|| rmp_serde::to_vec_named(black_box(value)).unwrap());
    });
    group.bench_function("bincode_deserialize", |b| {
        b.iter(|| bincode::deserialize::<T>(black_box(&bincode_bytes)).unwrap());
    });
    group.bench_function("msgpack_deserialize", |b| {
        b.iter(|| rmp_serde::from_slice::<T>(black_box(&msgpack_bytes)).unwrap());
    });

    group.finish();
}

fn serialization_benchmarks(c: &mut Criterion) {
    let small_witness = make_witness_data(8, 1024);
    let large_witness = make_witness_data(32, 256 * 1024);
    let small_blob = make_blob_payload(64 * 1024);
    let large_blob = make_blob_payload(4 * 1024 * 1024);

    bench_roundtrip(c, "WitnessData_8x1K", &small_witness);
    bench_roundtrip(c, "WitnessData_32x256K", &large_witness);
    bench_roundtrip(c, "BlobPayload_64KB", &small_blob);
    bench_roundtrip(c, "BlobPayload_4MB", &large_blob);
}

criterion_group!(benches, serialization_benchmarks);
criterion_main!(benches);
