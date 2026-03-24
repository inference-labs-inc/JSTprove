use std::path::Path;
use std::time::Instant;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, prove_bn254, verify_bn254, witness_bn254_from_f64};
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

fn run_pipeline(
    model_path: &Path,
    label: &str,
    activations: &[f64],
    logup_chunk_bits: Option<usize>,
) {
    let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
    let mut params = metadata.circuit_params.clone();
    params.logup_chunk_bits = logup_chunk_bits;
    OnnxContext::set_all(metadata.architecture, params.clone(), Some(metadata.wandb));

    let chunk_label = match logup_chunk_bits {
        Some(b) => format!("chunk_bits={b}"),
        None => "adaptive".to_string(),
    };
    println!("\n--- {label} ({chunk_label}) ---");

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("compile: {:>10}", fmt(compile_ms));

    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    println!(
        "circuit: {:>10}",
        format!("{:.1} KiB", bundle.circuit.len() as f64 / 1024.0)
    );

    let t = Instant::now();
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        activations,
        &[],
        false,
    )
    .unwrap();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("witness: {:>10}", fmt(witness_ms));

    let t = Instant::now();
    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("prove:   {:>10}", fmt(prove_ms));
    println!(
        "proof:   {:>10}",
        format!("{:.1} KiB", proof.len() as f64 / 1024.0)
    );

    let t = Instant::now();
    assert!(verify_bn254(&bundle.circuit, &wb.witness, &proof).unwrap());
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("verify:  {:>10}", fmt(verify_ms));

    println!("peak RSS: {:.1} MiB", rss_bytes() as f64 / 1048576.0);
    println!(
        "total:   {:>10}",
        fmt(compile_ms + witness_ms + prove_ms + verify_ms)
    );
}

fn bench_model(model_name: &str) {
    let model_file = format!("{model_name}.onnx");
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jstprove_remainder/models")
        .join(&model_file);
    if !model_path.exists() {
        println!("SKIP {model_file}: not found at {}", model_path.display());
        return;
    }

    let metadata = expander_metadata::generate_from_onnx(&model_path).unwrap();
    let params = metadata.circuit_params.clone();
    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

    println!("\n{}", "=".repeat(60));
    println!("MODEL: {model_file}");
    println!("{}", "=".repeat(60));

    run_pipeline(&model_path, model_name, &activations, Some(10));
    run_pipeline(&model_path, model_name, &activations, None);
    for c in [11, 12, 13, 14] {
        run_pipeline(&model_path, model_name, &activations, Some(c));
    }
}

fn main() {
    let models: Vec<String> = if let Ok(m) = std::env::var("MODEL") {
        m.split(',').map(str::to_string).collect()
    } else {
        vec!["lenet".to_string(), "mini_resnet".to_string()]
    };

    for model in &models {
        bench_model(model);
    }
}
