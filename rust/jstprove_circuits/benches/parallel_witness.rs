use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, witness_bn254_from_f64};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use rayon::prelude::*;

fn fmt(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn main() {
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "lenet".to_string());
    let batch_size: usize = std::env::var("BATCH")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let model_file = format!("{model_name}.onnx");
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jstprove_remainder/models")
        .join(&model_file);
    assert!(
        model_path.exists(),
        "{model_file} not found at {}",
        model_path.display()
    );

    let metadata = expander_metadata::generate_from_onnx(&model_path).unwrap();
    let params = metadata.circuit_params.clone();
    OnnxContext::set_all(
        metadata.architecture,
        metadata.circuit_params,
        Some(metadata.wandb),
    );

    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();

    let activation_sets: Vec<Vec<f64>> = (0..batch_size)
        .map(|b| {
            (0..num_act)
                .map(|i| (i as f64 + b as f64) / num_act as f64)
                .collect()
        })
        .collect();

    println!(
        "model: {model_file}  batch_size: {batch_size}  threads: {}",
        rayon::current_num_threads()
    );
    println!("{}", "=".repeat(60));

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    println!("compile: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let bundle = Arc::new(read_circuit_msgpack(circuit_path_str).unwrap());

    let t = Instant::now();
    let wb_single = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activation_sets[0],
        &[],
        false,
    )
    .unwrap();
    let single_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("single witness: {:>10}", fmt(single_ms));
    drop(wb_single);

    let t = Instant::now();
    for acts in &activation_sets {
        let _wb = witness_bn254_from_f64(
            &bundle.circuit,
            &bundle.witness_solver,
            &params,
            acts,
            &[],
            false,
        )
        .unwrap();
    }
    let sequential_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "sequential batch ({batch_size}): {:>10}  ({:.1}ms/witness)",
        fmt(sequential_ms),
        sequential_ms / batch_size as f64
    );

    let t = Instant::now();
    activation_sets.par_iter().for_each(|acts| {
        let _wb = witness_bn254_from_f64(
            &bundle.circuit,
            &bundle.witness_solver,
            &params,
            acts,
            &[],
            false,
        )
        .unwrap();
    });
    let parallel_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "parallel batch   ({batch_size}): {:>10}  ({:.1}ms/witness)",
        fmt(parallel_ms),
        parallel_ms / batch_size as f64
    );

    let speedup = sequential_ms / parallel_ms;
    println!("\nspeedup: {speedup:.2}x");
    println!(
        "parallel efficiency: {:.0}% (ideal = {batch_size}x on {} threads)",
        (speedup / rayon::current_num_threads().min(batch_size) as f64) * 100.0,
        rayon::current_num_threads()
    );
}
