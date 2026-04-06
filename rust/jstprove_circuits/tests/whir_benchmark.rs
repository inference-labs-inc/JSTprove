use std::time::Instant;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_goldilocks_basefold, compile_goldilocks_whir, prove_goldilocks_basefold,
    prove_goldilocks_whir, verify_goldilocks_basefold, verify_goldilocks_whir,
    witness_goldilocks_basefold_from_f64, witness_goldilocks_whir_from_f64,
};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use jstprove_onnx::quantizer::N_BITS_GOLDILOCKS;

fn lenet_model_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("jstprove_remainder/models/lenet.onnx")
}

fn setup_onnx_context_goldilocks()
-> jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams {
    let model_path = lenet_model_path();
    let metadata =
        expander_metadata::generate_from_onnx_for_field(&model_path, N_BITS_GOLDILOCKS, None)
            .unwrap();
    OnnxContext::set_all(
        metadata.architecture,
        metadata.circuit_params,
        Some(metadata.wandb),
    );
    OnnxContext::get_params().unwrap()
}

fn dummy_activations(
    params: &jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams,
) -> Vec<f64> {
    let total: usize = params
        .inputs
        .iter()
        .map(|inp| inp.shape.iter().product::<usize>())
        .sum();
    vec![0.0_f64; total]
}

#[test]
fn whir_vs_basefold_lenet_benchmark() {
    let params = setup_onnx_context_goldilocks();
    let activations = dummy_activations(&params);

    eprintln!("\n============================================================");
    eprintln!("WHIR vs Basefold benchmark — LeNet");
    eprintln!("============================================================\n");

    let tmp = tempfile::TempDir::new().unwrap();

    let bf_circuit_path = tmp.path().join("circuit_bf.msgpack");
    let whir_circuit_path = tmp.path().join("circuit_whir.msgpack");

    let t = Instant::now();
    compile_goldilocks_basefold(
        bf_circuit_path.to_str().unwrap(),
        false,
        Some(params.clone()),
    )
    .unwrap();
    let bf_compile = t.elapsed();

    let t = Instant::now();
    compile_goldilocks_whir(
        whir_circuit_path.to_str().unwrap(),
        false,
        Some(params.clone()),
    )
    .unwrap();
    let whir_compile = t.elapsed();

    let bf_bundle = read_circuit_msgpack(bf_circuit_path.to_str().unwrap()).unwrap();
    let whir_bundle = read_circuit_msgpack(whir_circuit_path.to_str().unwrap()).unwrap();

    let t = Instant::now();
    let bf_wb = witness_goldilocks_basefold_from_f64(
        &bf_bundle.circuit,
        &bf_bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    let bf_witness = t.elapsed();

    let t = Instant::now();
    let whir_wb = witness_goldilocks_whir_from_f64(
        &whir_bundle.circuit,
        &whir_bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    let whir_witness = t.elapsed();

    let t = Instant::now();
    let bf_proof = prove_goldilocks_basefold(&bf_bundle.circuit, &bf_wb.witness, false).unwrap();
    let bf_prove = t.elapsed();

    let t = Instant::now();
    let whir_proof = prove_goldilocks_whir(&whir_bundle.circuit, &whir_wb.witness, false).unwrap();
    let whir_prove = t.elapsed();

    let t = Instant::now();
    let bf_ok = verify_goldilocks_basefold(&bf_bundle.circuit, &bf_wb.witness, &bf_proof).unwrap();
    let bf_verify = t.elapsed();

    let t = Instant::now();
    let whir_ok =
        verify_goldilocks_whir(&whir_bundle.circuit, &whir_wb.witness, &whir_proof).unwrap();
    let whir_verify = t.elapsed();

    assert!(bf_ok);
    assert!(whir_ok);

    eprintln!("| PCS | Compile | Witness | Prove | Verify | Proof Size |");
    eprintln!("|-----|---------|---------|-------|--------|------------|");
    eprintln!(
        "| Basefold | {:.2}s | {}ms | {}ms | {}ms | {} KiB |",
        bf_compile.as_secs_f64(),
        bf_witness.as_millis(),
        bf_prove.as_millis(),
        bf_verify.as_millis(),
        bf_proof.len() / 1024,
    );
    eprintln!(
        "| **WHIR** | **{:.2}s** | **{}ms** | **{}ms** | **{}ms** | **{} KiB** |",
        whir_compile.as_secs_f64(),
        whir_witness.as_millis(),
        whir_prove.as_millis(),
        whir_verify.as_millis(),
        whir_proof.len() / 1024,
    );
}
