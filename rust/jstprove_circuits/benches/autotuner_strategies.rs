use std::path::Path;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::compile_bn254;

fn clear_cache() {
    if let Some(dir) = std::env::var("HOME").ok().map(|h| {
        let base = if cfg!(target_os = "macos") {
            std::path::PathBuf::from(&h).join("Library/Caches")
        } else {
            std::env::var("XDG_CACHE_HOME")
                .ok()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from(&h).join(".cache"))
        };
        base.join("jstprove/chunk_width")
    }) {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn model_name() -> String {
    std::env::var("MODEL").unwrap_or_else(|_| "lenet".to_string())
}

fn model_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jstprove_remainder/models")
        .join(format!("{}.onnx", model_name()))
}

/// Benchmark cold compile: the autotuner cache is cleared before every iteration,
/// forcing a full sweep each time.
fn bench_cold_compile(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        return;
    }

    let metadata = expander_metadata::generate_from_onnx(&path).unwrap();
    let params = metadata.circuit_params.clone();
    let arch = metadata.architecture.clone();
    let wandb = metadata.wandb.clone();

    let mut group = c.benchmark_group("autotuner/cold_compile");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function(model_name(), |b| {
        b.iter_batched(
            || {
                clear_cache();
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

    group.finish();
}

/// Benchmark warm compile: a single cold compile pre-populates the cache, then
/// every subsequent iteration hits the cache and skips the sweep.
fn bench_warm_compile(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        return;
    }

    // Pre-populate the cache with one cold compile.
    {
        let metadata = expander_metadata::generate_from_onnx(&path).unwrap();
        let mut params = metadata.circuit_params.clone();
        params.logup_chunk_bits = None;
        clear_cache();
        OnnxContext::set_all(metadata.architecture, params.clone(), Some(metadata.wandb));
        let tmp = tempfile::TempDir::new().unwrap();
        compile_bn254(
            tmp.path().join("circuit.bundle").to_str().unwrap(),
            false,
            Some(params),
        )
        .unwrap();
    }

    let metadata = expander_metadata::generate_from_onnx(&path).unwrap();
    let params = metadata.circuit_params.clone();
    let arch = metadata.architecture.clone();
    let wandb = metadata.wandb.clone();

    let mut group = c.benchmark_group("autotuner/warm_compile");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function(model_name(), |b| {
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

    group.finish();
}

criterion_group!(autotuner_benches, bench_cold_compile, bench_warm_compile);
criterion_main!(autotuner_benches);
