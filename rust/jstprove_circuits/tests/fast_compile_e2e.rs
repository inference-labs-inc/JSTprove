use std::collections::HashMap;
use std::path::Path;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_and_witness_bn254_direct, fast_compile_prove, fast_compile_verify, prove_bn254_direct,
    verify_bn254_direct,
};

fn lenet_model_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx")
}

fn setup_onnx_context() -> jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams {
    let model_path = lenet_model_path();
    assert!(
        model_path.exists(),
        "lenet.onnx not found at {}",
        model_path.display()
    );
    let metadata = expander_metadata::generate_from_onnx(&model_path).unwrap();
    let params = metadata.circuit_params.clone();
    OnnxContext::set_all(
        metadata.architecture,
        metadata.circuit_params,
        Some(metadata.wandb),
    );
    params
}

fn dummy_activations(
    params: &jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams,
) -> Vec<f64> {
    let num_act: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    (0..num_act).map(|i| i as f64 / num_act as f64).collect()
}

#[test]
fn direct_builder_lenet_prove_verify() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let (mut circuit, witness) =
        compile_and_witness_bn254_direct(&params, &activations, &[]).unwrap();
    let proof = prove_bn254_direct(&mut circuit, &witness).unwrap();
    assert!(verify_bn254_direct(&mut circuit, &witness, &proof).unwrap());
}

#[test]
fn fast_compile_prove_verify_file_roundtrip() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let alpha = (f64::from(params.scale_base)).powi(params.scale_exponent as i32);
    let quantized: Vec<i64> = activations.iter().map(|&v| (v * alpha) as i64).collect();

    let tmp = tempfile::TempDir::new().unwrap();
    let input_path = tmp.path().join("input.msgpack");
    let proof_path = tmp.path().join("proof.msgpack");

    let mut map = HashMap::new();
    map.insert("input", quantized);
    let bytes = rmp_serde::to_vec_named(&map).unwrap();
    std::fs::write(&input_path, bytes).unwrap();

    fast_compile_prove(
        input_path.to_str().unwrap(),
        proof_path.to_str().unwrap(),
        false,
    )
    .unwrap();
    assert!(proof_path.exists());

    let valid =
        fast_compile_verify(input_path.to_str().unwrap(), proof_path.to_str().unwrap()).unwrap();
    assert!(valid, "fast-compile verify failed on LeNet");
}
