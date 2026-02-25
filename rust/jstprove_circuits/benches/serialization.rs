use criterion::{Criterion, black_box, criterion_group, criterion_main};
use jstprove_circuits::runner::schema::{CompiledCircuit, ProofBundle, WitnessBundle};
use serde::{Deserialize, Serialize};

fn make_compiled_circuit(size: usize) -> CompiledCircuit {
    CompiledCircuit {
        circuit: vec![0xAB; size],
        witness_solver: vec![0xCD; size / 2],
        metadata: None,
        version: None,
    }
}

fn make_witness_bundle(size: usize) -> WitnessBundle {
    WitnessBundle {
        witness: vec![0xEF; size],
        output_data: Some((0..256).collect()),
        version: None,
    }
}

fn make_proof_bundle(size: usize) -> ProofBundle {
    ProofBundle {
        proof: vec![0x42; size],
        version: None,
    }
}

fn bench_roundtrip<T: Serialize + for<'de> Deserialize<'de>>(
    c: &mut Criterion,
    name: &str,
    value: &T,
) {
    let json_bytes = serde_json::to_vec(value).unwrap();
    let msgpack_bytes = rmp_serde::to_vec_named(value).unwrap();

    let group_name = format!(
        "{name} (json={}, msgpack={})",
        json_bytes.len(),
        msgpack_bytes.len()
    );
    let mut group = c.benchmark_group(&group_name);

    group.bench_function("json_serialize", |b| {
        b.iter(|| serde_json::to_vec(black_box(value)).unwrap());
    });
    group.bench_function("msgpack_serialize", |b| {
        b.iter(|| rmp_serde::to_vec_named(black_box(value)).unwrap());
    });
    group.bench_function("json_deserialize", |b| {
        b.iter(|| serde_json::from_slice::<T>(black_box(&json_bytes)).unwrap());
    });
    group.bench_function("msgpack_deserialize", |b| {
        b.iter(|| rmp_serde::from_slice::<T>(black_box(&msgpack_bytes)).unwrap());
    });

    group.finish();
}

fn serialization_benchmarks(c: &mut Criterion) {
    let small_circuit = make_compiled_circuit(1024);
    let large_circuit = make_compiled_circuit(4 * 1024 * 1024);
    let witness = make_witness_bundle(2 * 1024 * 1024);
    let proof = make_proof_bundle(512 * 1024);

    bench_roundtrip(c, "CompiledCircuit_1KB", &small_circuit);
    bench_roundtrip(c, "CompiledCircuit_4MB", &large_circuit);
    bench_roundtrip(c, "WitnessBundle_2MB", &witness);
    bench_roundtrip(c, "ProofBundle_512KB", &proof);
}

criterion_group!(benches, serialization_benchmarks);
criterion_main!(benches);
