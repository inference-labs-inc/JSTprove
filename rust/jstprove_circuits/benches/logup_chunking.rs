use std::path::Path;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::compile_bn254;

fn bench_compile_chunk_bits(c: &mut Criterion) {
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "lenet".to_string());
    let model_file = format!("{model_name}.onnx");
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jstprove_remainder/models")
        .join(&model_file);
    if !model_path.exists() {
        return;
    }

    let metadata = expander_metadata::generate_from_onnx(&model_path).unwrap();
    let params_base = metadata.circuit_params.clone();
    let arch = metadata.architecture.clone();
    let wandb = metadata.wandb.clone();

    let mut group = c.benchmark_group("logup_chunking");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    for chunk_bits in [Some(10usize), Some(11), Some(12), Some(13), Some(14), None] {
        let label = match chunk_bits {
            Some(b) => format!("chunk_bits_{b}"),
            None => "adaptive".to_string(),
        };

        let mut params_it = params_base.clone();
        params_it.logup_chunk_bits = chunk_bits;
        let arch_it = arch.clone();
        let wandb_it = wandb.clone();

        group.bench_with_input(BenchmarkId::new("compile", &label), &label, |b, _| {
            b.iter_batched(
                || {
                    OnnxContext::set_all(
                        arch_it.clone(),
                        params_it.clone(),
                        Some(wandb_it.clone()),
                    );
                    tempfile::TempDir::new().unwrap()
                },
                |tmp| {
                    let path = tmp
                        .path()
                        .join("circuit.bundle")
                        .to_str()
                        .unwrap()
                        .to_string();
                    compile_bn254(&path, false, Some(params_it.clone())).unwrap();
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group!(chunking_benches, bench_compile_chunk_bits);
criterion_main!(chunking_benches);
