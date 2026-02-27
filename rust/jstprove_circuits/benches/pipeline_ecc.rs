use std::path::Path;
use std::time::Instant;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_and_witness_bn254_direct, compile_bn254, prove_bn254, prove_bn254_direct, verify_bn254,
    verify_bn254_direct, witness_bn254_from_f64,
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

    println!("model: lenet.onnx\n{}", "=".repeat(55));

    println!("\n--- DirectBuilder ---");
    let _rss0 = rss_bytes();
    let t = Instant::now();
    let (mut circ, wit) = compile_and_witness_bn254_direct(&params, &activations, &[]).unwrap();
    let dt = t.elapsed().as_secs_f64() * 1000.0;
    let rss1 = rss_bytes();
    println!(
        "compile+witness: {:>10}  peak RSS: {:.1} MiB",
        fmt(dt),
        rss1 as f64 / 1048576.0
    );

    let t = Instant::now();
    let proof2 = prove_bn254_direct(&mut circ, &wit).unwrap();
    println!("prove:   {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let t = Instant::now();
    assert!(verify_bn254_direct(&mut circ, &wit, &proof2).unwrap());
    println!("verify:  {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    println!("\n--- ECC IR pipeline ---");
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

    let rss2 = rss_bytes();
    println!("peak RSS after ECC: {:.1} MiB", rss2 as f64 / 1048576.0);

    let t = Instant::now();
    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    println!("prove:   {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));

    let t = Instant::now();
    assert!(verify_bn254(&bundle.circuit, &wb.witness, &proof).unwrap());
    println!("verify:  {:>10}", fmt(t.elapsed().as_secs_f64() * 1000.0));
}
