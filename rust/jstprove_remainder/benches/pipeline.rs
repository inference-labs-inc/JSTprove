use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const LENET_INPUT_ELEMENTS: usize = 3072; // [1, 3, 32, 32]

fn generate_input_file(path: &Path, num_elements: usize) {
    let input: Vec<f64> = (0..num_elements)
        .map(|i| (i as f64 / num_elements as f64))
        .collect();
    let mut map = HashMap::new();
    map.insert("input", input);
    let bytes = rmp_serde::to_vec_named(&map).unwrap();
    std::fs::write(path, bytes).unwrap();
}

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
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/lenet.onnx");
    assert!(
        model_path.exists(),
        "lenet.onnx not found at {}",
        model_path.display()
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let compiled_path = tmp.path().join("compiled.bin");
    let input_path = tmp.path().join("input.msgpack");
    let witness_path = tmp.path().join("witness.bin");
    let proof_path = tmp.path().join("proof.bin");

    generate_input_file(&input_path, LENET_INPUT_ELEMENTS);

    println!("model:   lenet.onnx (12 layers, input [1,3,32,32])");
    println!("backend: remainder");
    println!("{}", "-".repeat(50));

    let t = Instant::now();
    jstprove_remainder::runner::compile::run(&model_path, &compiled_path, false).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    let compiled_size = std::fs::metadata(&compiled_path).unwrap().len();
    println!(
        "compile: {:>10}  artifact: {}",
        fmt_duration(compile_ms),
        fmt_bytes(compiled_size)
    );

    let t = Instant::now();
    jstprove_remainder::runner::witness::run(&compiled_path, &input_path, &witness_path, false)
        .unwrap();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;
    let witness_size = std::fs::metadata(&witness_path).unwrap().len();
    println!(
        "witness: {:>10}  artifact: {}",
        fmt_duration(witness_ms),
        fmt_bytes(witness_size)
    );

    let t = Instant::now();
    jstprove_remainder::runner::prove::run(&compiled_path, &witness_path, &proof_path, false)
        .unwrap();
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    let proof_size = std::fs::metadata(&proof_path).unwrap().len();
    println!(
        "prove:   {:>10}  artifact: {}",
        fmt_duration(prove_ms),
        fmt_bytes(proof_size)
    );

    let t = Instant::now();
    jstprove_remainder::runner::verify::run(&compiled_path, &proof_path, &input_path).unwrap();
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("verify:  {:>10}", fmt_duration(verify_ms));

    let total_ms = compile_ms + witness_ms + prove_ms + verify_ms;
    println!("{}", "-".repeat(50));
    println!("total:   {:>10}", fmt_duration(total_ms));
}
