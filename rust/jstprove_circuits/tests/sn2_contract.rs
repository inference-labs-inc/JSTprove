#![allow(unused)]

use std::collections::HashMap;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{
    compile_bn254, prove_bn254, verify_and_extract_bn254, verify_bn254, witness_bn254,
    witness_bn254_from_f64,
};
use jstprove_circuits::runner::errors::RunError;
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;
use jstprove_circuits::runner::schema::{CompiledCircuit, WitnessRequest};
use jstprove_circuits::runner::verify_extract::VerifiedOutput;

fn sample_circuit_params() -> CircuitParams {
    serde_json::from_value(serde_json::json!({
        "scale_base": 10,
        "scale_exponent": 3,
        "rescale_config": {"layer_0": true},
        "inputs": [{"name": "input_0", "elem_type": 1, "shape": [1, 3]}],
        "outputs": [{"name": "output_0", "elem_type": 1, "shape": [1, 2]}],
        "freivalds_reps": 1,
        "n_bits_config": {},
        "weights_as_inputs": false,
        "backend": "expander"
    }))
    .expect("sample CircuitParams must parse")
}

#[test]
fn verified_output_json_roundtrip() {
    let original = VerifiedOutput {
        valid: true,
        inputs: vec![1.0, 2.5, -3.14],
        outputs: vec![0.99, -0.01],
        scale_base: 10,
        scale_exponent: 3,
    };

    let json = serde_json::to_string(&original).expect("serialize VerifiedOutput");
    let restored: VerifiedOutput = serde_json::from_str(&json).expect("deserialize VerifiedOutput");

    assert_eq!(restored.valid, original.valid);
    assert_eq!(restored.inputs, original.inputs);
    assert_eq!(restored.outputs, original.outputs);
    assert_eq!(restored.scale_base, original.scale_base);
    assert_eq!(restored.scale_exponent, original.scale_exponent);

    let map: serde_json::Value = serde_json::from_str(&json).expect("parse as Value");
    assert!(map.get("valid").is_some());
    assert!(map.get("inputs").is_some());
    assert!(map.get("outputs").is_some());
    assert!(map.get("scale_base").is_some());
    assert!(map.get("scale_exponent").is_some());
}

#[test]
fn verify_and_extract_bn254_rejects_garbage() {
    let garbage_circuit = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let garbage_witness = vec![0x01, 0x02, 0x03];
    let garbage_proof = vec![0xFF; 16];

    let result =
        verify_and_extract_bn254(&garbage_circuit, &garbage_witness, &garbage_proof, 2, None);

    assert!(
        result.is_err(),
        "garbage data must produce RunError, not Ok"
    );
}

#[test]
fn verify_and_extract_bn254_rejects_empty() {
    let result = verify_and_extract_bn254(&[], &[], &[], 0, None);
    assert!(result.is_err(), "empty data must produce RunError, not Ok");
}

#[test]
fn prove_bn254_rejects_garbage() {
    let result = prove_bn254(&[0xDE, 0xAD], &[0xBE, 0xEF], false);
    assert!(result.is_err(), "prove_bn254 must reject garbage bytes");
}

#[test]
fn verify_bn254_rejects_garbage() {
    let result = verify_bn254(&[0xDE, 0xAD], &[0xBE, 0xEF], &[0xCA, 0xFE]);
    assert!(result.is_err(), "verify_bn254 must reject garbage bytes");
}

#[test]
fn witness_request_msgpack_roundtrip() {
    let original = WitnessRequest {
        circuit: vec![1, 2, 3, 4],
        witness_solver: vec![5, 6, 7, 8],
        inputs: serde_json::to_vec(&serde_json::json!({"input": [[1.0, 2.0, 3.0]]}))
            .expect("serialize inputs"),
        outputs: serde_json::to_vec(&serde_json::json!({"output": [[0.5, 0.6]]}))
            .expect("serialize outputs"),
        metadata: Some(sample_circuit_params()),
    };

    let packed = rmp_serde::to_vec(&original).expect("msgpack serialize WitnessRequest");
    let restored: WitnessRequest =
        rmp_serde::from_slice(&packed).expect("msgpack deserialize WitnessRequest");

    assert_eq!(restored.circuit, original.circuit);
    assert_eq!(restored.witness_solver, original.witness_solver);
    assert_eq!(restored.inputs, original.inputs);
    assert_eq!(restored.outputs, original.outputs);
    assert!(restored.metadata.is_some());
}

#[test]
fn compiled_circuit_msgpack_roundtrip() {
    let original = CompiledCircuit {
        circuit: vec![10, 20, 30],
        witness_solver: vec![40, 50, 60],
        metadata: Some(sample_circuit_params()),
    };

    let packed = rmp_serde::to_vec(&original).expect("msgpack serialize CompiledCircuit");
    let restored: CompiledCircuit =
        rmp_serde::from_slice(&packed).expect("msgpack deserialize CompiledCircuit");

    assert_eq!(restored.circuit, original.circuit);
    assert_eq!(restored.witness_solver, original.witness_solver);
    assert!(restored.metadata.is_some());

    let meta = restored.metadata.unwrap();
    assert_eq!(meta.scale_base, 10);
    assert_eq!(meta.scale_exponent, 3);
    assert_eq!(meta.inputs.len(), 1);
    assert_eq!(meta.outputs.len(), 1);
    assert_eq!(meta.inputs[0].name, "input_0");
    assert_eq!(meta.inputs[0].shape, vec![1, 3]);
    assert_eq!(meta.outputs[0].name, "output_0");
    assert_eq!(meta.outputs[0].shape, vec![1, 2]);
}

#[test]
fn circuit_params_json_roundtrip() {
    let original = sample_circuit_params();
    let json = serde_json::to_string(&original).expect("serialize CircuitParams");
    let restored: CircuitParams = serde_json::from_str(&json).expect("deserialize CircuitParams");

    assert_eq!(restored.scale_base, original.scale_base);
    assert_eq!(restored.scale_exponent, original.scale_exponent);
    assert_eq!(restored.inputs.len(), original.inputs.len());
    assert_eq!(restored.outputs.len(), original.outputs.len());
    assert_eq!(restored.freivalds_reps, original.freivalds_reps);
    assert_eq!(restored.weights_as_inputs, original.weights_as_inputs);
    assert_eq!(
        restored.rescale_config.get("layer_0"),
        original.rescale_config.get("layer_0")
    );
}

#[test]
fn circuit_params_dimension_helpers() {
    let params = sample_circuit_params();
    assert_eq!(params.total_input_dims(), 3);
    assert_eq!(params.total_output_dims(), 2);
    assert_eq!(params.effective_input_dims(), 3);
    assert_eq!(params.effective_output_dims(), 2);
}

#[test]
fn architecture_deserializes_from_json() {
    let json = serde_json::json!({
        "architecture": [{
            "id": 0,
            "name": "matmul_0",
            "op_type": "MatMul",
            "inputs": ["input_0", "weight_0"],
            "outputs": ["output_0"],
            "shape": {"output_0": [1, 2]},
            "tensor": null,
            "params": null,
            "opset_version_number": 13
        }]
    });

    let arch: Architecture =
        serde_json::from_value(json).expect("Architecture must deserialize from valid JSON");
    assert_eq!(arch.architecture.len(), 1);
    assert_eq!(arch.architecture[0].op_type, "MatMul");
    assert_eq!(arch.architecture[0].inputs, vec!["input_0", "weight_0"]);
}

#[test]
fn wandb_deserializes_from_json() {
    let json = serde_json::json!({
        "w_and_b": [{
            "id": 0,
            "name": "weight_0",
            "op_type": "Initializer",
            "inputs": [],
            "outputs": ["weight_0"],
            "shape": {"weight_0": [3, 2]},
            "tensor": {"value": [[1, 2], [3, 4], [5, 6]]},
            "params": null,
            "opset_version_number": 0
        }]
    });

    let wandb: WANDB =
        serde_json::from_value(json).expect("WANDB must deserialize from valid JSON");
    assert_eq!(wandb.w_and_b.len(), 1);
    assert_eq!(wandb.w_and_b[0].name, "weight_0");
    assert!(wandb.w_and_b[0].tensor.is_some());
}

#[test]
fn onnx_context_set_and_get() {
    let params = sample_circuit_params();
    let arch: Architecture = serde_json::from_value(serde_json::json!({
        "architecture": []
    }))
    .unwrap();
    let wandb: WANDB = serde_json::from_value(serde_json::json!({
        "w_and_b": []
    }))
    .unwrap();

    OnnxContext::set_all(arch, params.clone(), Some(wandb));

    let retrieved_params = OnnxContext::get_params().expect("params must be retrievable after set");
    assert_eq!(retrieved_params.scale_base, params.scale_base);

    let retrieved_arch =
        OnnxContext::get_architecture().expect("architecture must be retrievable after set");
    assert!(retrieved_arch.architecture.is_empty());
}

#[test]
fn read_circuit_msgpack_rejects_missing_file() {
    let result = read_circuit_msgpack("/nonexistent/path/to/circuit.msgpack");
    assert!(result.is_err());
}

#[test]
fn witness_bn254_rejects_garbage() {
    let req = WitnessRequest {
        circuit: vec![0xFF; 8],
        witness_solver: vec![0xFF; 8],
        inputs: serde_json::to_vec(&serde_json::json!({"input": [[1.0]]})).unwrap(),
        outputs: serde_json::to_vec(&serde_json::json!({"output": [[1.0]]})).unwrap(),
        metadata: None,
    };

    let result = witness_bn254(&req, false);
    assert!(
        result.is_err(),
        "witness_bn254 must reject garbage circuit bytes"
    );
}

#[test]
fn witness_bn254_from_f64_rejects_garbage() {
    let params = sample_circuit_params();
    let result = witness_bn254_from_f64(
        &[0xFF; 8],
        &[0xFF; 8],
        &params,
        &[1.0, 2.0, 3.0],
        &[],
        false,
    );
    assert!(
        result.is_err(),
        "witness_bn254_from_f64 must reject garbage circuit bytes"
    );
}

#[test]
fn compile_bn254_rejects_nonexistent_path() {
    let result = compile_bn254("/nonexistent/circuit/path", false, None);
    assert!(
        result.is_err(),
        "compile_bn254 must reject nonexistent path"
    );
}

#[test]
fn verified_output_field_names_stable() {
    let vo = VerifiedOutput {
        valid: false,
        inputs: vec![],
        outputs: vec![],
        scale_base: 0,
        scale_exponent: 0,
    };

    let val = serde_json::to_value(&vo).expect("serialize to Value");
    let obj = val
        .as_object()
        .expect("VerifiedOutput must serialize as JSON object");

    let expected_fields = ["valid", "inputs", "outputs", "scale_base", "scale_exponent"];
    for field in &expected_fields {
        assert!(
            obj.contains_key(*field),
            "VerifiedOutput missing expected field: {field}"
        );
    }
    assert_eq!(
        obj.len(),
        expected_fields.len(),
        "VerifiedOutput has unexpected extra fields"
    );
}

#[test]
fn witness_request_field_names_stable() {
    let req = WitnessRequest {
        circuit: vec![],
        witness_solver: vec![],
        inputs: vec![],
        outputs: vec![],
        metadata: None,
    };

    let packed = rmp_serde::to_vec_named(&req).expect("named msgpack serialize");
    let val: serde_json::Value = serde_json::to_value(&req).expect("json serialize");
    let obj = val.as_object().expect("must be object");

    let expected_fields = ["circuit", "witness_solver", "inputs", "outputs", "metadata"];
    for field in &expected_fields {
        assert!(
            obj.contains_key(*field),
            "WitnessRequest missing expected field: {field}"
        );
    }
}

#[test]
fn compiled_circuit_field_names_stable() {
    let cc = CompiledCircuit {
        circuit: vec![],
        witness_solver: vec![],
        metadata: None,
    };

    let val = serde_json::to_value(&cc).expect("json serialize");
    let obj = val.as_object().expect("must be object");

    let expected_fields = ["circuit", "witness_solver", "metadata"];
    for field in &expected_fields {
        assert!(
            obj.contains_key(*field),
            "CompiledCircuit missing expected field: {field}"
        );
    }
    assert_eq!(obj.len(), expected_fields.len());
}
