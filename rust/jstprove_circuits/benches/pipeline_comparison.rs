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

fn generate_activations(num_elements: usize) -> Vec<f64> {
    (0..num_elements)
        .map(|i| (i as f64 / num_elements as f64))
        .collect()
}

struct Timings {
    compile_ms: f64,
    build_ms: f64,
    gen_provable_ms: f64,
    witness_ms: f64,
    prove_ms: f64,
    gen_verifiable_ms: f64,
    verify_ms: f64,
}

impl Timings {
    fn total(&self) -> f64 {
        self.compile_ms
            + self.build_ms
            + self.gen_provable_ms
            + self.witness_ms
            + self.prove_ms
            + self.gen_verifiable_ms
            + self.verify_ms
    }
}

fn run_remainder(model_path: &Path, input_elements: usize) -> Timings {
    use jstprove_remainder::onnx::graph::LayerGraph;
    use jstprove_remainder::onnx::parser;
    use jstprove_remainder::onnx::quantizer::{self, ScaleConfig};
    use jstprove_remainder::runner::circuit_builder;
    use jstprove_remainder::runner::prove::{prepare_for_proving, prove_prepared};
    use jstprove_remainder::runner::verify::{prepare_for_verifying, verify_prepared};
    use jstprove_remainder::runner::witness::{compute_witness, prepare_public_shreds};

    let alpha = ScaleConfig::default().alpha;
    let quantized_input: Vec<i64> = generate_activations(input_elements)
        .iter()
        .map(|&v| (v * alpha as f64).round() as i64)
        .collect();

    let t = Instant::now();
    let parsed = parser::parse_onnx(model_path).unwrap();
    let graph = LayerGraph::from_parsed(&parsed).unwrap();
    let config = ScaleConfig::default();
    let model = quantizer::quantize_model(graph, &config).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let witness_data = compute_witness(&model, &quantized_input).unwrap();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut build_model = model.clone();
    for layer in &mut build_model.graph.layers {
        if let Some(&obs) = witness_data.observed_n_bits.get(&layer.name) {
            layer.n_bits = Some(obs);
            build_model.n_bits_config.insert(layer.name.clone(), obs);
        }
    }
    let input_padded_size = jstprove_remainder::padding::next_power_of_two(quantized_input.len());
    let build_result = circuit_builder::build_circuit(&build_model, input_padded_size).unwrap();
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let prepared_provable = prepare_for_proving(
        build_result.circuit.clone(),
        &build_result.manifest,
        &witness_data.shreds,
    )
    .unwrap();
    let gen_provable_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut proof = prove_prepared(prepared_provable).unwrap();
    proof.observed_n_bits = witness_data.observed_n_bits;
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let public_shreds = prepare_public_shreds(
        &build_model,
        &quantized_input,
        &proof.expected_output,
        &proof.observed_n_bits,
    )
    .unwrap();

    let t = Instant::now();
    let prepared_verifiable = prepare_for_verifying(
        build_result.circuit.clone(),
        &build_result.manifest,
        &public_shreds,
    )
    .unwrap();
    let gen_verifiable_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    verify_prepared(prepared_verifiable, &proof).unwrap();
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    Timings {
        compile_ms,
        build_ms,
        gen_provable_ms,
        witness_ms,
        prove_ms,
        gen_verifiable_ms,
        verify_ms,
    }
}

fn run_expander(model_path: &Path, input_elements: usize) -> Timings {
    let activations = generate_activations(input_elements);
    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    let t = Instant::now();
    let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
    let params = metadata.circuit_params.clone();
    OnnxContext::set_all(
        metadata.architecture,
        metadata.circuit_params,
        Some(metadata.wandb),
    );
    compile_bn254(circuit_path_str, false, Some(params.clone()), false).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

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
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let proof_bytes = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let ok = verify_bn254(&bundle.circuit, &wb.witness, &proof_bytes).unwrap();
    assert!(ok, "expander verification failed");
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    Timings {
        compile_ms,
        build_ms: 0.0,
        gen_provable_ms: 0.0,
        witness_ms,
        prove_ms,
        gen_verifiable_ms: 0.0,
        verify_ms,
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

    const INPUT_ELEMENTS: usize = 3072;

    println!("model: lenet.onnx (12 layers, input [1,3,32,32])");
    println!("{}", "=".repeat(62));

    let rem = run_remainder(&model_path, INPUT_ELEMENTS);
    let exp = run_expander(&model_path, INPUT_ELEMENTS);

    println!("\n{:<16} {:>15} {:>15}", "", "Remainder", "Expander (ECC)");
    println!("{}", "-".repeat(62));
    println!(
        "{:<16} {:>15} {:>15}",
        "compile:",
        fmt(rem.compile_ms),
        fmt(exp.compile_ms)
    );
    println!(
        "{:<16} {:>15} {:>15}",
        "witness:",
        fmt(rem.witness_ms),
        fmt(exp.witness_ms)
    );
    println!(
        "{:<16} {:>15} {:>15}",
        "build:",
        fmt(rem.build_ms),
        fmt(exp.build_ms)
    );
    println!(
        "{:<16} {:>15} {:>15}",
        "gen_provable:",
        fmt(rem.gen_provable_ms),
        fmt(exp.gen_provable_ms)
    );
    println!(
        "{:<16} {:>15} {:>15}",
        "prove:",
        fmt(rem.prove_ms),
        fmt(exp.prove_ms)
    );
    println!(
        "{:<16} {:>15} {:>15}",
        "gen_verifiable:",
        fmt(rem.gen_verifiable_ms),
        fmt(exp.gen_verifiable_ms)
    );
    println!(
        "{:<16} {:>15} {:>15}",
        "verify:",
        fmt(rem.verify_ms),
        fmt(exp.verify_ms)
    );
    println!("{}", "-".repeat(62));
    println!(
        "{:<16} {:>15} {:>15}",
        "total:",
        fmt(rem.total()),
        fmt(exp.total())
    );
}
