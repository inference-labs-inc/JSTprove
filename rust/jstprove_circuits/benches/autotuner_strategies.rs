use std::path::Path;
use std::time::Instant;

use jstprove_circuits::circuit_functions::gadgets::autotuner;
use jstprove_circuits::circuit_functions::utils::build_layers::default_n_bits_for_config;
use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, prove_bn254, verify_bn254, witness_bn254_from_f64};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;

use expander_compiler::frontend::BN254Config;

fn fmt(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

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

struct PipelineResult {
    compile_ms: f64,
    witness_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    circuit_kib: f64,
    chunk_bits: Option<usize>,
}

fn run_pipeline(
    model_path: &Path,
    activations: &[f64],
    chunk_bits: Option<usize>,
) -> PipelineResult {
    let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
    let mut params = metadata.circuit_params.clone();
    params.logup_chunk_bits = chunk_bits;
    OnnxContext::set_all(metadata.architecture, params.clone(), Some(metadata.wandb));

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

    let resolved_chunk = OnnxContext::get_params().unwrap().logup_chunk_bits;

    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let circuit_kib = bundle.circuit.len() as f64 / 1024.0;

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

    let t = Instant::now();
    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    assert!(verify_bn254(&bundle.circuit, &wb.witness, &proof).unwrap());
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    PipelineResult {
        compile_ms,
        witness_ms,
        prove_ms,
        verify_ms,
        circuit_kib,
        chunk_bits: resolved_chunk,
    }
}

fn print_result(label: &str, r: &PipelineResult) {
    let total = r.compile_ms + r.witness_ms + r.prove_ms + r.verify_ms;
    let chunk_str = r
        .chunk_bits
        .map_or("default".to_string(), |b| b.to_string());
    println!(
        "  {label:<30} chunk={chunk_str:<4} compile={:<10} prove={:<10} circuit={:.1} KiB  total={}",
        fmt(r.compile_ms),
        fmt(r.prove_ms),
        r.circuit_kib,
        fmt(total),
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
    let n_bits = default_n_bits_for_config::<BN254Config>();

    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

    println!("\n{}", "=".repeat(100));
    println!("MODEL: {model_file}");

    let circuit_key = autotuner::circuit_cache_key(&params, &metadata.architecture);
    let operator_key = autotuner::operator_cache_key(&params, &metadata.architecture, n_bits);
    println!("  circuit_key:  {circuit_key}");
    println!("  operator_key: {operator_key}");
    println!("{}", "=".repeat(100));

    println!("\n[baseline: chunk_bits=12 (hardcoded)]");
    let baseline = run_pipeline(&model_path, &activations, Some(12));
    print_result("hardcoded-12", &baseline);

    println!("\n[strategy A: circuit-level cache]");
    clear_cache();
    let t = Instant::now();
    let cold_a = run_pipeline(&model_path, &activations, None);
    let _cold_a_total_ms = t.elapsed().as_secs_f64() * 1000.0;
    print_result("cold (sweep + compile)", &cold_a);
    println!(
        "    sweep overhead: {}",
        fmt(_cold_a_total_ms
            - cold_a.compile_ms
            - cold_a.witness_ms
            - cold_a.prove_ms
            - cold_a.verify_ms)
    );

    let t = Instant::now();
    let warm_a = run_pipeline(&model_path, &activations, None);
    let _warm_a_total_ms = t.elapsed().as_secs_f64() * 1000.0;
    print_result("warm (cache hit)", &warm_a);

    println!("\n[strategy B: operator-profile cache]");
    clear_cache();
    let t = Instant::now();
    let cold_b = run_pipeline(&model_path, &activations, None);
    let _cold_b_total_ms = t.elapsed().as_secs_f64() * 1000.0;
    print_result("cold (sweep + compile)", &cold_b);

    let warm_b = run_pipeline(&model_path, &activations, None);
    print_result("warm (cache hit)", &warm_b);

    println!("\n[cross-model operator cache transfer]");
    println!("  (operator key shared if same op_type profile)");
    println!(
        "  circuit_key == operator_key: {}",
        circuit_key == operator_key
    );
}

fn main() {
    let models: Vec<String> = if let Ok(m) = std::env::var("MODEL") {
        m.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        vec!["lenet".to_string(), "mini_resnet".to_string()]
    };

    for model in &models {
        bench_model(model);
    }

    println!("\n{}", "=".repeat(100));
    println!("CROSS-MODEL TRANSFER TEST");
    println!("{}", "=".repeat(100));
    println!("Running lenet (populates cache), then mini_resnet (checks for operator cache hit)");

    clear_cache();

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models");
    let lenet_path = models_dir.join("lenet.onnx");
    let mini_resnet_path = models_dir.join("mini_resnet.onnx");

    if lenet_path.exists() && mini_resnet_path.exists() {
        let meta_l = expander_metadata::generate_from_onnx(&lenet_path).unwrap();
        let meta_r = expander_metadata::generate_from_onnx(&mini_resnet_path).unwrap();
        let n_bits = default_n_bits_for_config::<BN254Config>();
        let key_l =
            autotuner::operator_cache_key(&meta_l.circuit_params, &meta_l.architecture, n_bits);
        let key_r =
            autotuner::operator_cache_key(&meta_r.circuit_params, &meta_r.architecture, n_bits);
        println!("  lenet operator_key:       {key_l}");
        println!("  mini_resnet operator_key: {key_r}");
        println!("  keys match: {}", key_l == key_r);
    }
}
