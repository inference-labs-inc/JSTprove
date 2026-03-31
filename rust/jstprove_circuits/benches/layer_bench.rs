use std::collections::HashMap;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rmpv::Value;

use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use jstprove_circuits::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::{compile_bn254, prove_bn254, witness_bn254_from_f64};
use jstprove_circuits::proof_system::ProofSystem;
use jstprove_circuits::runner::main_runner::read_circuit_msgpack;

/// α = 2^18, the fixed-point scale factor.
const ALPHA: i64 = 262144;

/// Build a 4-D nested `rmpv::Value` with shape [oc, ic, kh, kw], all elements set to `val`.
fn make_weight_4d(oc: usize, ic: usize, kh: usize, kw: usize, val: i64) -> Value {
    Value::Array(
        (0..oc)
            .map(|_| {
                Value::Array(
                    (0..ic)
                        .map(|_| {
                            Value::Array(
                                (0..kh)
                                    .map(|_| {
                                        Value::Array(
                                            (0..kw).map(|_| Value::from(val)).collect(),
                                        )
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// Construct a minimal single-Conv-layer `ExpanderMetadata`-equivalent triple without
/// touching any ONNX file.
///
/// Spec:
///   input  x: [1,1,8,8]
///   kernel W: [4,1,3,3]  (weights scaled by α¹ = 262144)
///   bias   B: [4]        (zero biases)
///   output y: [1,4,6,6]
///   stride 1, no padding
fn make_conv_metadata() -> (CircuitParams, Architecture, WANDB) {
    let params = CircuitParams {
        scale_base: 2,
        scale_exponent: 18,
        rescale_config: [("conv_0".to_string(), true)].into_iter().collect(),
        inputs: vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![1, 1, 8, 8],
        }],
        outputs: vec![ONNXIO {
            name: "y".into(),
            elem_type: 1,
            shape: vec![1, 4, 6, 6],
        }],
        freivalds_reps: 1,
        // Empty map → falls back to DEFAULT_N_BITS_BN254 (64 bits) inside the circuit builder.
        n_bits_config: HashMap::new(),
        weights_as_inputs: false,
        proof_system: ProofSystem::Expander,
        curve: None,
        // Pin chunk width to skip the autotuner sweep, keeping the bench deterministic.
        logup_chunk_bits: Some(12),
    };

    let conv_layer = ONNXLayer {
        id: 0,
        name: "conv_0".into(),
        op_type: "Conv".into(),
        inputs: vec!["x".into(), "W".into(), "B".into()],
        outputs: vec!["y".into()],
        shape: [
            ("x", vec![1usize, 1, 8, 8]),
            ("W", vec![4, 1, 3, 3]),
            ("B", vec![4]),
            ("y", vec![1, 4, 6, 6]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect(),
        tensor: None,
        params: Some(Value::Map(vec![
            (
                Value::String("kernel_shape".into()),
                Value::Array(vec![Value::from(3i64), Value::from(3i64)]),
            ),
            (
                Value::String("strides".into()),
                Value::Array(vec![Value::from(1i64), Value::from(1i64)]),
            ),
            (
                Value::String("pads".into()),
                Value::Array(vec![Value::from(0i64); 4]),
            ),
            (
                Value::String("dilations".into()),
                Value::Array(vec![Value::from(1i64), Value::from(1i64)]),
            ),
            (Value::String("group".into()), Value::from(1i64)),
        ])),
        opset_version_number: 17,
    };

    let arch = Architecture {
        architecture: vec![conv_layer],
    };

    // W: all weights = 1 * α (scaled by α¹ as required for Conv weights)
    let w_tensor = make_weight_4d(4, 1, 3, 3, ALPHA);
    // B: zero bias (0 * α² = 0)
    let b_tensor = Value::Array(vec![Value::from(0i64); 4]);

    let w_layer = ONNXLayer {
        id: 0,
        name: "W".into(),
        op_type: "Const".into(),
        inputs: vec![],
        outputs: vec![],
        shape: [("W".to_string(), vec![4usize, 1, 3, 3])]
            .into_iter()
            .collect(),
        tensor: Some(w_tensor),
        params: None,
        opset_version_number: -1,
    };

    let b_layer = ONNXLayer {
        id: 1,
        name: "B".into(),
        op_type: "Const".into(),
        inputs: vec![],
        outputs: vec![],
        shape: [("B".to_string(), vec![4usize])].into_iter().collect(),
        tensor: Some(b_tensor),
        params: None,
        opset_version_number: -1,
    };

    let wandb = WANDB {
        w_and_b: vec![w_layer, b_layer],
    };

    (params, arch, wandb)
}

fn conv_compile(c: &mut Criterion) {
    let (params, arch, wandb) = make_conv_metadata();
    let mut group = c.benchmark_group("conv/compile");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function("bn254", |b| {
        b.iter_batched(
            || {
                OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
                tempfile::TempDir::new().unwrap()
            },
            |tmp| {
                let path = tmp.path().join("c.bundle").to_str().unwrap().to_string();
                compile_bn254(&path, false, Some(black_box(params.clone()))).unwrap();
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn conv_witness(c: &mut Criterion) {
    let (params, arch, wandb) = make_conv_metadata();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 1 * 8 * 8];

    let mut group = c.benchmark_group("conv/witness");
    group.sample_size(10);
    group.bench_function("bn254", |b| {
        b.iter(|| {
            witness_bn254_from_f64(
                &bundle.circuit,
                &bundle.witness_solver,
                &params,
                &activations,
                &[],
                false,
            )
            .unwrap()
        });
    });
    group.finish();
}

fn conv_prove(c: &mut Criterion) {
    let (params, arch, wandb) = make_conv_metadata();
    OnnxContext::set_all(arch, params.clone(), Some(wandb));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 1 * 8 * 8];
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let mut group = c.benchmark_group("conv/prove");
    group.sample_size(10);
    group.bench_function("bn254", |b| {
        b.iter(|| prove_bn254(&bundle.circuit, &wb.witness, false).unwrap());
    });
    group.finish();
}

criterion_group!(conv_benches, conv_compile, conv_witness, conv_prove);
criterion_main!(conv_benches);
