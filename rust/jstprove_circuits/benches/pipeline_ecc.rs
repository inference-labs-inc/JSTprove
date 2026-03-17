use std::path::Path;
use std::time::Instant;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_bn254, compile_goldilocks, compile_goldilocks_orion, prove_bn254, prove_goldilocks,
    prove_goldilocks_orion, verify_bn254, verify_goldilocks, verify_goldilocks_orion,
    witness_bn254_from_f64, witness_goldilocks_from_f64, witness_goldilocks_orion_from_f64,
};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;

fn fmt(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn rss_bytes() -> u64 {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) };
    if ret != 0 {
        return 0;
    }
    let usage = unsafe { u.assume_init() };
    let maxrss = usage.ru_maxrss as u64;
    if cfg!(target_os = "linux") {
        maxrss * 1024
    } else {
        maxrss
    }
}

fn main() {
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "lenet".to_string());
    let model_file = format!("{model_name}.onnx");
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jstprove_remainder/models")
        .join(&model_file);
    assert!(
        model_path.exists(),
        "{model_file} not found at {}",
        model_path.display()
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

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
    let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

    println!("model: {model_file}\n{}", "=".repeat(55));

    println!("\n--- BN254 pipeline ---");
    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    println!("compile: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let t = Instant::now();
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    println!("witness: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let t = Instant::now();
    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    println!("prove:   {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));
    println!(
        "proof:   {:>10}",
        format!("{:.1} KiB", proof.len() as f64 / 1024.0)
    );

    let t = Instant::now();
    assert!(verify_bn254(&bundle.circuit, &wb.witness, &proof).unwrap());
    println!("verify:  {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let rss_bn254 = rss_bytes();
    println!("peak RSS: {:.1} MiB", rss_bn254 as f64 / 1048576.0);

    let gl_circuit_path = tmp.path().join("circuit_gl.bundle");
    let gl_circuit_path_str = gl_circuit_path.to_str().unwrap();

    println!("\n--- Goldilocks pipeline ---");
    let t = Instant::now();
    compile_goldilocks(gl_circuit_path_str, false, Some(params.clone())).unwrap();
    println!("compile: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let gl_bundle = read_circuit_msgpack(gl_circuit_path_str).unwrap();
    let t = Instant::now();
    let gl_wb = witness_goldilocks_from_f64(
        &gl_bundle.circuit,
        &gl_bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    println!("witness: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let t = Instant::now();
    let gl_proof = prove_goldilocks(&gl_bundle.circuit, &gl_wb.witness, false).unwrap();
    println!("prove:   {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));
    println!(
        "proof:   {:>10}",
        format!("{:.1} KiB", gl_proof.len() as f64 / 1024.0)
    );

    let t = Instant::now();
    assert!(verify_goldilocks(&gl_bundle.circuit, &gl_wb.witness, &gl_proof).unwrap());
    println!("verify:  {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let rss_gl = rss_bytes();
    println!("peak RSS: {:.1} MiB", rss_gl as f64 / 1048576.0);

    let orion_circuit_path = tmp.path().join("circuit_orion.bundle");
    let orion_circuit_path_str = orion_circuit_path.to_str().unwrap();

    println!("\n--- Goldilocks Orion PCS pipeline ---");
    let t = Instant::now();
    compile_goldilocks_orion(orion_circuit_path_str, false, Some(params.clone())).unwrap();
    println!("compile: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let orion_bundle = read_circuit_msgpack(orion_circuit_path_str).unwrap();
    let t = Instant::now();
    let orion_wb = witness_goldilocks_orion_from_f64(
        &orion_bundle.circuit,
        &orion_bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    println!("witness: {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let t = Instant::now();
    let orion_proof =
        prove_goldilocks_orion(&orion_bundle.circuit, &orion_wb.witness, false).unwrap();
    println!("prove:   {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));
    println!(
        "proof:   {:>10}",
        format!("{:.1} KiB", orion_proof.len() as f64 / 1024.0)
    );

    let t = Instant::now();
    assert!(
        verify_goldilocks_orion(&orion_bundle.circuit, &orion_wb.witness, &orion_proof).unwrap()
    );
    println!("verify:  {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let rss_orion = rss_bytes();
    println!("peak RSS: {:.1} MiB", rss_orion as f64 / 1048576.0);
}
