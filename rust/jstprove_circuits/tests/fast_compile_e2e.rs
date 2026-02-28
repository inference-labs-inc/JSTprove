use std::collections::HashMap;
use std::path::Path;

use arith::Field;
use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_and_witness_bn254_direct, compile_bn254, fast_compile_prove, fast_compile_verify,
    prove_bn254, prove_bn254_direct, verify_bn254, verify_bn254_direct, witness_bn254_from_f64,
};
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;

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
fn ecc_pipeline_lenet_prove_verify() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone()), false).unwrap();
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
fn cross_validate_ecc_vs_direct_builder() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let (mut direct_circuit, direct_witness) =
        compile_and_witness_bn254_direct(&params, &activations, &[]).unwrap();
    let direct_proof = prove_bn254_direct(&mut direct_circuit, &direct_witness).unwrap();
    assert!(
        verify_bn254_direct(&mut direct_circuit, &direct_witness, &direct_proof).unwrap(),
        "DirectBuilder proof failed verification"
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let circuit_path = tmp.path().join("circuit.msgpack");
    let circuit_path_str = circuit_path.to_str().unwrap();

    compile_bn254(circuit_path_str, false, Some(params.clone()), false).unwrap();
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

    let ecc_proof = prove_bn254(&bundle.circuit, &wb.witness, false).unwrap();
    assert!(
        verify_bn254(&bundle.circuit, &wb.witness, &ecc_proof).unwrap(),
        "ECC pipeline proof failed verification"
    );
}

#[test]
fn direct_builder_wrong_witness_rejected() {
    let params = setup_onnx_context();
    let activations = dummy_activations(&params);

    let (mut circuit, witness) =
        compile_and_witness_bn254_direct(&params, &activations, &[]).unwrap();

    let mut bad_witness = witness.clone();
    for v in bad_witness.iter_mut().take(10) {
        *v = expander_compiler::frontend::CircuitField::<expander_compiler::frontend::BN254Config>::from(999u32);
    }

    use expander_compiler::gkr_engine::{FieldEngine, GKREngine};
    type Cfg = expander_compiler::frontend::BN254Config;
    type Simd = <<Cfg as GKREngine>::FieldConfig as FieldEngine>::SimdCircuitField;
    use arith::SimdField;
    let ps = <Simd as SimdField>::PACK_SIZE;

    circuit.layers[0].input_vals = bad_witness
        .iter()
        .map(|&v| Simd::pack(&vec![v; ps]))
        .collect();
    circuit.evaluate();

    let output = &circuit.layers.last().unwrap().output_vals;
    let n_zeros = circuit.expected_num_output_zeros;
    let has_nonzero = output[..n_zeros].iter().any(|v| !v.is_zero());
    assert!(
        has_nonzero,
        "corrupted witness should produce non-zero outputs in circuit evaluation"
    );
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
