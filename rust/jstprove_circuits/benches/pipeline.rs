use std::path::Path;
use std::time::Instant;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, prove_bn254, verify_bn254, witness_bn254_from_f64};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;

fn fmt_bytes(n: u64) -> String {
    if n >= 1_073_741_824 {
        format!("{:.1} GiB", n as f64 / 1_073_741_824.0)
    } else if n >= 1_048_576 {
        format!("{:.1} MiB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn fmt_duration(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn main() {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
    assert!(
        model_path.exists(),
        "lenet.onnx not found at {}",
        model_path.display()
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    println!("model:   lenet.onnx (12 layers, input [1,3,32,32])");
    println!("backend: expander (BN254)");
    println!("{}", "-".repeat(55));

    let t = Instant::now();
    let metadata = expander_metadata::generate_from_onnx(&model_path).unwrap();
    let meta_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("metadata:{:>10}", fmt_duration(meta_ms));

    let params = metadata.circuit_params.clone();
    OnnxContext::set_all(
        metadata.architecture,
        metadata.circuit_params,
        Some(metadata.wandb),
    );

    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    let compiled_size = std::fs::metadata(&circuit_path).unwrap().len();
    println!(
        "compile: {:>10}  artifact: {}",
        fmt_duration(compile_ms),
        fmt_bytes(compiled_size)
    );

    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();

    let num_activations: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let activations: Vec<f64> = (0..num_activations)
        .map(|i| i as f64 / num_activations as f64)
        .collect();

    let t = Instant::now();
    let witness_bundle = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;
    let witness_size = witness_bundle.witness.len() as u64;
    println!(
        "witness: {:>10}  artifact: {}",
        fmt_duration(witness_ms),
        fmt_bytes(witness_size)
    );

    let t = Instant::now();
    let proof = prove_bn254(&bundle.circuit, &witness_bundle.witness, false).unwrap();
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    let proof_size = proof.len() as u64;
    println!(
        "prove:   {:>10}  artifact: {}",
        fmt_duration(prove_ms),
        fmt_bytes(proof_size)
    );

    let t = Instant::now();
    let ok = verify_bn254(&bundle.circuit, &witness_bundle.witness, &proof).unwrap();
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(ok, "verification failed");
    println!("verify:  {:>10}", fmt_duration(verify_ms));

    let total_ms = meta_ms + compile_ms + witness_ms + prove_ms + verify_ms;
    println!("{}", "-".repeat(55));
    println!("total:   {:>10}", fmt_duration(total_ms));
}
