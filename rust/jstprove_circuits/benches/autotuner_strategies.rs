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

fn run_compile(model_path: &Path, chunk_bits: Option<usize>) -> (f64, f64, Option<usize>) {
    let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
    let mut params = metadata.circuit_params.clone();
    params.logup_chunk_bits = chunk_bits;
    OnnxContext::set_all(metadata.architecture, params.clone(), Some(metadata.wandb));

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params)).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

    let resolved_chunk = OnnxContext::get_params().unwrap().logup_chunk_bits;
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let circuit_kib = bundle.circuit.len() as f64 / 1024.0;

    (compile_ms, circuit_kib, resolved_chunk)
}

fn run_full_pipeline(model_path: &Path, activations: &[f64], chunk_bits: Option<usize>) -> f64 {
    let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
    let mut params = metadata.circuit_params.clone();
    params.logup_chunk_bits = chunk_bits;
    OnnxContext::set_all(metadata.architecture, params.clone(), Some(metadata.wandb));

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    let t = Instant::now();
    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        activations,
        &[],
        false,
    )
    .unwrap();
    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    assert!(verify_bn254(&bundle.circuit, &wb.witness, &proof).unwrap());
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models");
    let lenet_path = models_dir.join("lenet.onnx");
    let mini_resnet_path = models_dir.join("mini_resnet.onnx");

    if !lenet_path.exists() || !mini_resnet_path.exists() {
        println!("SKIP: requires both lenet.onnx and mini_resnet.onnx");
        return;
    }

    let n_bits = default_n_bits_for_config::<BN254Config>();

    let meta_l = expander_metadata::generate_from_onnx(&lenet_path).unwrap();
    let meta_r = expander_metadata::generate_from_onnx(&mini_resnet_path).unwrap();

    let circuit_key_l = autotuner::circuit_cache_key(&meta_l.circuit_params, &meta_l.architecture);
    let circuit_key_r = autotuner::circuit_cache_key(&meta_r.circuit_params, &meta_r.architecture);
    let op_key_l =
        autotuner::operator_cache_key(&meta_l.circuit_params, &meta_l.architecture, n_bits);
    let op_key_r =
        autotuner::operator_cache_key(&meta_r.circuit_params, &meta_r.architecture, n_bits);

    println!("{}", "=".repeat(90));
    println!("CACHE KEY ANALYSIS");
    println!("{}", "=".repeat(90));
    println!("lenet       circuit_key={circuit_key_l}  operator_key={op_key_l}");
    println!("mini_resnet circuit_key={circuit_key_r}  operator_key={op_key_r}");
    println!(
        "circuit keys match: {}   operator keys match: {}",
        circuit_key_l == circuit_key_r,
        op_key_l == op_key_r
    );
    println!();
    println!("Whole-model keys differ for both strategies because lenet and mini_resnet");
    println!("have different operator mixes. In production (dsperse), models are sliced");
    println!("into single-operator segments. Conv+ReLU slices from different parent");
    println!("models with matching (kappa, n_bits, element_counts) share operator keys,");
    println!("enabling cross-model cache transfer without a cold sweep.");

    println!("\n{}", "=".repeat(90));
    println!("AUTOTUNER OVERHEAD: COLD vs WARM");
    println!("{}", "=".repeat(90));

    for (label, model_path) in [("lenet", &lenet_path), ("mini_resnet", &mini_resnet_path)] {
        println!("\n--- {label} ---");

        clear_cache();
        let (cold_ms, cold_kib, cold_chunk) = run_compile(model_path, None);
        println!(
            "  cold (sweep):  compile={:<10} circuit={:.1} KiB  chunk={:?}",
            fmt(cold_ms),
            cold_kib,
            cold_chunk
        );

        let (warm_ms, warm_kib, warm_chunk) = run_compile(model_path, None);
        println!(
            "  warm (cached): compile={:<10} circuit={:.1} KiB  chunk={:?}",
            fmt(warm_ms),
            warm_kib,
            warm_chunk
        );

        let (base_ms, base_kib, _) = run_compile(model_path, Some(12));
        println!(
            "  hardcoded-12:  compile={:<10} circuit={:.1} KiB",
            fmt(base_ms),
            base_kib,
        );
    }

    println!("\n{}", "=".repeat(90));
    println!("STRATEGY B: OPERATOR CACHE TRANSFER SIMULATION");
    println!("{}", "=".repeat(90));
    println!("Simulating dsperse slice scenario: compile lenet, then check if operator");
    println!("cache helps mini_resnet (different circuit key, possibly shared operator key).");

    clear_cache();

    println!("\n[step 1] compile lenet (cold sweep, populates both caches)");
    let (ms, _, chunk) = run_compile(&lenet_path, None);
    println!("  compile={:<10}  chunk={:?}", fmt(ms), chunk);

    println!("\n[step 2] compile mini_resnet (circuit key miss, check operator key)");
    let (ms, _, chunk) = run_compile(&mini_resnet_path, None);
    let was_cold = ms > 20_000.0;
    println!("  compile={:<10}  chunk={:?}", fmt(ms), chunk);
    if was_cold {
        println!("  -> operator key MISS (different profile, cold sweep required)");
    } else {
        println!("  -> operator key HIT (shared profile, no sweep)");
    }

    println!("\n{}", "=".repeat(90));
    println!("FULL PIPELINE (compile + witness + prove + verify)");
    println!("{}", "=".repeat(90));

    for (label, model_path) in [("lenet", &lenet_path), ("mini_resnet", &mini_resnet_path)] {
        let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
        let num_act: usize = metadata
            .circuit_params
            .inputs
            .iter()
            .map(|io| io.shape.iter().product::<usize>())
            .sum();
        let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

        let total_warm = run_full_pipeline(model_path, &activations, None);
        let total_base = run_full_pipeline(model_path, &activations, Some(12));
        println!(
            "  {label:<15} autotuned(warm)={:<10} hardcoded-12={:<10}",
            fmt(total_warm),
            fmt(total_base),
        );
    }
}
