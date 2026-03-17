use std::path::Path;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_bn254, compile_goldilocks, compile_goldilocks_basefold, deserialize_circuit_bn254,
    flatten_circuit_bn254, prove_bn254, prove_goldilocks, prove_goldilocks_basefold,
    verify_and_extract_bn254_with_flat_ref, verify_and_extract_bn254_with_layered, verify_bn254,
    verify_goldilocks, verify_goldilocks_basefold, witness_bn254_from_f64,
    witness_goldilocks_basefold_from_f64, witness_goldilocks_from_f64,
};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use jstprove_onnx::quantizer::N_BITS_GOLDILOCKS;

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
fn ecc_pipeline_lenet_prove_verify() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();

    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    assert!(verify_bn254(&bundle.circuit, &wb.witness, &proof).unwrap());
}

#[test]
fn layered_circuit_handle_reused_across_verify_calls() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();

    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    let layered = deserialize_circuit_bn254(&bundle.circuit).unwrap();
    let num_inputs = params.effective_input_dims();

    let result1 =
        verify_and_extract_bn254_with_layered(&layered, &wb.witness, &proof, num_inputs, None)
            .unwrap();
    assert!(result1.valid);

    let result2 =
        verify_and_extract_bn254_with_layered(&layered, &wb.witness, &proof, num_inputs, None)
            .unwrap();
    assert!(result2.valid);

    assert_eq!(result1.outputs, result2.outputs);
    assert_eq!(result1.scale_base, result2.scale_base);
    assert_eq!(result1.scale_exponent, result2.scale_exponent);

    let activations2: Vec<f64> = activations.iter().map(|&v| v + 0.01).collect();
    let wb2 = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations2,
        &[],
        false,
    )
    .unwrap();
    let proof2 = prove_bn254(&bundle.circuit, &wb2.witness, false).unwrap();

    let result3 =
        verify_and_extract_bn254_with_layered(&layered, &wb2.witness, &proof2, num_inputs, None)
            .unwrap();
    assert!(result3.valid);
    assert_ne!(result3.outputs, result1.outputs);
}

#[test]
fn flat_ref_verify_matches_layered_verify() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();

    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    let layered = deserialize_circuit_bn254(&bundle.circuit).unwrap();
    let num_inputs = params.effective_input_dims();

    let layered_result =
        verify_and_extract_bn254_with_layered(&layered, &wb.witness, &proof, num_inputs, None)
            .unwrap();
    assert!(layered_result.valid);

    let flat = flatten_circuit_bn254(&layered);
    let ref_result =
        verify_and_extract_bn254_with_flat_ref(&flat, &wb.witness, &proof, num_inputs, None)
            .unwrap();
    assert!(ref_result.valid);

    assert_eq!(layered_result.outputs, ref_result.outputs);
    assert_eq!(layered_result.scale_base, ref_result.scale_base);
    assert_eq!(layered_result.scale_exponent, ref_result.scale_exponent);
}

#[test]
fn flat_ref_verify_reused_across_calls() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let layered = deserialize_circuit_bn254(&bundle.circuit).unwrap();
    let flat = flatten_circuit_bn254(&layered);
    let num_inputs = params.effective_input_dims();

    let wb1 = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();
    let proof1 = prove_bn254(&bundle.circuit, &wb1.witness, false).unwrap();

    let r1 = verify_and_extract_bn254_with_flat_ref(&flat, &wb1.witness, &proof1, num_inputs, None)
        .unwrap();
    assert!(r1.valid);

    let r2 = verify_and_extract_bn254_with_flat_ref(&flat, &wb1.witness, &proof1, num_inputs, None)
        .unwrap();
    assert!(r2.valid);
    assert_eq!(r1.outputs, r2.outputs);

    let activations2: Vec<f64> = activations.iter().map(|&v| v + 0.01).collect();
    let wb2 = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations2,
        &[],
        false,
    )
    .unwrap();
    let proof2 = prove_bn254(&bundle.circuit, &wb2.witness, false).unwrap();

    let r3 = verify_and_extract_bn254_with_flat_ref(&flat, &wb2.witness, &proof2, num_inputs, None)
        .unwrap();
    assert!(r3.valid);
    assert_ne!(r1.outputs, r3.outputs);
}

#[test]
fn flat_ref_verify_compressed_witness() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();
    let layered = deserialize_circuit_bn254(&bundle.circuit).unwrap();
    let flat = flatten_circuit_bn254(&layered);
    let num_inputs = params.effective_input_dims();

    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        true,
    )
    .unwrap();
    let proof = prove_bn254(&bundle.circuit, &wb.witness, true).unwrap();

    let ref_result =
        verify_and_extract_bn254_with_flat_ref(&flat, &wb.witness, &proof, num_inputs, None)
            .unwrap();
    assert!(ref_result.valid);

    let layered_result =
        verify_and_extract_bn254_with_layered(&layered, &wb.witness, &proof, num_inputs, None)
            .unwrap();
    assert!(layered_result.valid);
    assert_eq!(ref_result.outputs, layered_result.outputs);
}

fn setup_onnx_context_goldilocks()
-> jstprove_circuits::circuit_functions::utils::onnx_model::CircuitParams {
    let model_path = lenet_model_path();
    assert!(
        model_path.exists(),
        "lenet.onnx not found at {}",
        model_path.display()
    );
    let metadata =
        expander_metadata::generate_from_onnx_for_field(&model_path, N_BITS_GOLDILOCKS).unwrap();
    let params = metadata.circuit_params.clone();
    OnnxContext::set_all(
        metadata.architecture,
        metadata.circuit_params,
        Some(metadata.wandb),
    );
    params
}

#[test]
fn goldilocks_pipeline_lenet_prove_verify() {
    let params = setup_onnx_context_goldilocks();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit_gl.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_goldilocks(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();

    let wb = witness_goldilocks_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let proof = prove_goldilocks(&bundle.circuit, &wb.witness, false).unwrap();
    assert!(verify_goldilocks(&bundle.circuit, &wb.witness, &proof).unwrap());
}


#[test]
fn goldilocks_basefold_pipeline_lenet_prove_verify() {
    let params = setup_onnx_context_goldilocks();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit_bf.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_goldilocks_basefold(circuit_path_str, false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(circuit_path_str).unwrap();

    let wb = witness_goldilocks_basefold_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let proof = prove_goldilocks_basefold(&bundle.circuit, &wb.witness, false).unwrap();
    assert!(verify_goldilocks_basefold(&bundle.circuit, &wb.witness, &proof).unwrap());
}
