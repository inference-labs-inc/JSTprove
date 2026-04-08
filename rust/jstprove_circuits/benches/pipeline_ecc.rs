use std::path::Path;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_bn254, compile_goldilocks, compile_goldilocks_basefold, prove_bn254, prove_goldilocks,
    prove_goldilocks_basefold, witness_bn254_from_f64, witness_goldilocks_basefold_from_f64,
    witness_goldilocks_from_f64,
};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use jstprove_onnx::quantizer::N_BITS_GOLDILOCKS;

fn model_name() -> String {
    std::env::var("MODEL").unwrap_or_else(|_| "lenet".to_string())
}

fn model_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jstprove_remainder/models")
        .join(format!("{}.onnx", model_name()))
}

fn bench_bn254(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        return;
    }

    let metadata = expander_metadata::generate_from_onnx(&path).unwrap();
    let params = metadata.circuit_params.clone();
    let arch = metadata.architecture.clone();
    let wandb = metadata.wandb.clone();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));

    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

    let mut group = c.benchmark_group(format!("pipeline/bn254/{}", model_name()));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function("compile", |b| {
        b.iter_batched(
            || {
                OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
                tempfile::TempDir::new().unwrap()
            },
            |tmp| {
                let p = tmp
                    .path()
                    .join("circuit.bundle")
                    .to_str()
                    .unwrap()
                    .to_string();
                compile_bn254(&p, false, Some(params.clone())).unwrap();
            },
            BatchSize::PerIteration,
        );
    });

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    compile_bn254(circuit_path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path.to_str().unwrap()).unwrap();

    group.bench_function("witness", |b| {
        b.iter(|| {
            witness_bn254_from_f64(
                &bundle.circuit,
                &bundle.witness_solver,
                &params,
                &activations,
                &[],
                false,
            )
            .unwrap()
        });
    });

    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    group.bench_function("prove", |b| {
        b.iter(|| prove_bn254(&bundle.circuit, &wb.witness, false).unwrap());
    });

    group.finish();
}

fn bench_goldilocks(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        return;
    }

    let metadata =
        expander_metadata::generate_from_onnx_for_field(&path, N_BITS_GOLDILOCKS, None).unwrap();
    let params = metadata.circuit_params.clone();
    let arch = metadata.architecture.clone();
    let wandb = metadata.wandb.clone();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));

    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

    let mut group = c.benchmark_group(format!("pipeline/goldilocks/{}", model_name()));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function("compile", |b| {
        b.iter_batched(
            || {
                OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
                tempfile::TempDir::new().unwrap()
            },
            |tmp| {
                let p = tmp
                    .path()
                    .join("circuit.bundle")
                    .to_str()
                    .unwrap()
                    .to_string();
                compile_goldilocks(&p, false, Some(params.clone())).unwrap();
            },
            BatchSize::PerIteration,
        );
    });

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    compile_goldilocks(circuit_path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path.to_str().unwrap()).unwrap();

    group.bench_function("witness", |b| {
        b.iter(|| {
            witness_goldilocks_from_f64(
                &bundle.circuit,
                &bundle.witness_solver,
                &params,
                &activations,
                &[],
                false,
            )
            .unwrap()
        });
    });

    let wb = witness_goldilocks_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    group.bench_function("prove", |b| {
        b.iter(|| prove_goldilocks(&bundle.circuit, &wb.witness, false).unwrap());
    });

    group.finish();
}

fn bench_goldilocks_basefold(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        return;
    }

    let metadata =
        expander_metadata::generate_from_onnx_for_field(&path, N_BITS_GOLDILOCKS, None).unwrap();
    let params = metadata.circuit_params.clone();
    let arch = metadata.architecture.clone();
    let wandb = metadata.wandb.clone();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));

    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

    let mut group = c.benchmark_group(format!("pipeline/goldilocks_basefold/{}", model_name()));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function("compile", |b| {
        b.iter_batched(
            || {
                OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
                tempfile::TempDir::new().unwrap()
            },
            |tmp| {
                let p = tmp
                    .path()
                    .join("circuit.bundle")
                    .to_str()
                    .unwrap()
                    .to_string();
                compile_goldilocks_basefold(&p, false, Some(params.clone())).unwrap();
            },
            BatchSize::PerIteration,
        );
    });

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    compile_goldilocks_basefold(circuit_path.to_str().unwrap(), false, Some(params.clone()))
        .unwrap();
    let bundle = read_circuit_msgpack(circuit_path.to_str().unwrap()).unwrap();

    group.bench_function("witness", |b| {
        b.iter(|| {
            witness_goldilocks_basefold_from_f64(
                &bundle.circuit,
                &bundle.witness_solver,
                &params,
                &activations,
                &[],
                false,
            )
            .unwrap()
        });
    });

    let wb = witness_goldilocks_basefold_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    group.bench_function("prove", |b| {
        b.iter(|| prove_goldilocks_basefold(&bundle.circuit, &wb.witness, false).unwrap());
    });

    group.finish();
}

criterion_group!(
    pipeline_benches,
    bench_bn254,
    bench_goldilocks,
    bench_goldilocks_basefold
);
criterion_main!(pipeline_benches);
