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

/// Build a 2-D nested `rmpv::Value` with shape [rows, cols], all elements set to `val`.
fn make_weight_2d(rows: usize, cols: usize, val: i64) -> Value {
    Value::Array(
        (0..rows)
            .map(|_| Value::Array((0..cols).map(|_| Value::from(val)).collect()))
            .collect(),
    )
}

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

/// Construct a minimal single-Gemm-layer metadata triple without touching any ONNX file.
///
/// Spec:
///   input  x: [1, 64]  — batch=1, in_features=64
///   weight W: [32, 64] — out_features=32, in_features=64 (transB=1)
///   bias   B: [32]     — zero biases; reshapes to [1,32] matching the core product
///   output y: [1, 32]
fn make_gemm_metadata() -> (CircuitParams, Architecture, WANDB) {
    let params = CircuitParams {
        scale_base: 2,
        scale_exponent: 18,
        rescale_config: [("gemm_0".to_string(), true)].into_iter().collect(),
        inputs: vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![1, 64],
        }],
        outputs: vec![ONNXIO {
            name: "y".into(),
            elem_type: 1,
            shape: vec![1, 32],
        }],
        freivalds_reps: 1,
        n_bits_config: HashMap::new(),
        weights_as_inputs: false,
        proof_system: ProofSystem::Expander,
        curve: None,
        logup_chunk_bits: Some(12),
    };

    let gemm_layer = ONNXLayer {
        id: 0,
        name: "gemm_0".into(),
        op_type: "Gemm".into(),
        inputs: vec!["x".into(), "W".into(), "B".into()],
        outputs: vec!["y".into()],
        shape: [
            ("x", vec![1usize, 64]),
            ("W", vec![32, 64]),
            ("B", vec![32]),
            ("y", vec![1, 32]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect(),
        tensor: None,
        params: Some(Value::Map(vec![
            (Value::String("transA".into()), Value::from(0i64)),
            (Value::String("transB".into()), Value::from(1i64)),
            (Value::String("alpha".into()), Value::from(1.0f64)),
            (Value::String("beta".into()), Value::from(1.0f64)),
        ])),
        opset_version_number: 17,
    };

    let arch = Architecture {
        architecture: vec![gemm_layer],
    };

    // W: all weights = 1 * α (scaled by α¹ as required for Gemm weights)
    let w_tensor = make_weight_2d(32, 64, ALPHA);
    // B: zero bias (0 * α² = 0)
    let b_tensor = Value::Array(vec![Value::from(0i64); 32]);

    let w_layer = ONNXLayer {
        id: 0,
        name: "W".into(),
        op_type: "Const".into(),
        inputs: vec![],
        outputs: vec![],
        shape: [("W".to_string(), vec![32usize, 64])].into_iter().collect(),
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
        shape: [("B".to_string(), vec![32usize])].into_iter().collect(),
        tensor: Some(b_tensor),
        params: None,
        opset_version_number: -1,
    };

    let wandb = WANDB {
        w_and_b: vec![w_layer, b_layer],
    };

    (params, arch, wandb)
}

fn gemm_compile(c: &mut Criterion) {
    let (params, arch, wandb) = make_gemm_metadata();
    let mut group = c.benchmark_group("gemm/compile");
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

fn gemm_witness(c: &mut Criterion) {
    let (params, arch, wandb) = make_gemm_metadata();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 64];

    let mut group = c.benchmark_group("gemm/witness");
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

fn gemm_prove(c: &mut Criterion) {
    let (params, arch, wandb) = make_gemm_metadata();
    OnnxContext::set_all(arch, params.clone(), Some(wandb));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 64];
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let mut group = c.benchmark_group("gemm/prove");
    group.sample_size(10);
    group.bench_function("bn254", |b| {
        b.iter(|| prove_bn254(&bundle.circuit, &wb.witness, false).unwrap());
    });
    group.finish();
}

/// Construct a minimal single-Softmax-layer metadata triple without touching any ONNX file.
///
/// Spec:
///   input  x: [1, 16]  — batch=1, 16 classes along softmax axis
///   output y: [1, 16]  — identical shape (passthrough shape inference)
///   axis=1, opset≥13
fn make_softmax_metadata() -> (CircuitParams, Architecture, WANDB) {
    let params = CircuitParams {
        scale_base: 2,
        scale_exponent: 18,
        rescale_config: HashMap::new(), // no rescale for Softmax
        inputs: vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![1, 16],
        }],
        outputs: vec![ONNXIO {
            name: "y".into(),
            elem_type: 1,
            shape: vec![1, 16],
        }],
        freivalds_reps: 1,
        n_bits_config: HashMap::new(),
        weights_as_inputs: false,
        proof_system: ProofSystem::Expander,
        curve: None,
        logup_chunk_bits: Some(12),
    };

    let softmax_layer = ONNXLayer {
        id: 0,
        name: "softmax_0".into(),
        op_type: "Softmax".into(),
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        shape: [("x", vec![1usize, 16]), ("y", vec![1, 16])]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        tensor: None,
        params: Some(Value::Map(vec![(
            Value::String("axis".into()),
            Value::from(1i64),
        )])),
        opset_version_number: 17,
    };

    let arch = Architecture {
        architecture: vec![softmax_layer],
    };
    let wandb = WANDB { w_and_b: vec![] }; // no weights

    (params, arch, wandb)
}

fn softmax_compile(c: &mut Criterion) {
    let (params, arch, wandb) = make_softmax_metadata();
    let mut group = c.benchmark_group("softmax/compile");
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

fn softmax_witness(c: &mut Criterion) {
    let (params, arch, wandb) = make_softmax_metadata();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 16];

    let mut group = c.benchmark_group("softmax/witness");
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

fn softmax_prove(c: &mut Criterion) {
    let (params, arch, wandb) = make_softmax_metadata();
    OnnxContext::set_all(arch, params.clone(), Some(wandb));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 16];
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let mut group = c.benchmark_group("softmax/prove");
    group.sample_size(10);
    group.bench_function("bn254", |b| {
        b.iter(|| prove_bn254(&bundle.circuit, &wb.witness, false).unwrap());
    });
    group.finish();
}

/// Construct a minimal single-LayerNorm-layer metadata triple without touching any ONNX file.
///
/// Spec:
///   input  x:     [1, 16]  — batch=1, feature_size=16
///   gamma (Scale): [16]    — all = ALPHA (γ = 1.0 at α¹ scale)
///   beta  (B):    [16]    — all = 0    (β = 0.0 at α² scale)
///   output y:     [1, 16]
///   axis = -1, opset 17
fn make_layer_norm_metadata() -> (CircuitParams, Architecture, WANDB) {
    let params = CircuitParams {
        scale_base: 2,
        scale_exponent: 18,
        rescale_config: HashMap::new(), // no rescale for LayerNorm
        inputs: vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![1, 16],
        }],
        outputs: vec![ONNXIO {
            name: "y".into(),
            elem_type: 1,
            shape: vec![1, 16],
        }],
        freivalds_reps: 1,
        n_bits_config: HashMap::new(),
        weights_as_inputs: false,
        proof_system: ProofSystem::Expander,
        curve: None,
        logup_chunk_bits: Some(12),
    };

    let layer_norm_layer = ONNXLayer {
        id: 0,
        name: "layer_norm_0".into(),
        op_type: "LayerNormalization".into(),
        inputs: vec!["x".into(), "Scale".into(), "B".into()],
        outputs: vec!["y".into()],
        shape: [
            ("x", vec![1usize, 16]),
            ("Scale", vec![16]),
            ("B", vec![16]),
            ("y", vec![1, 16]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect(),
        tensor: None,
        params: Some(Value::Map(vec![(
            Value::String("axis".into()),
            Value::from(-1i64),
        )])),
        opset_version_number: 17,
    };

    let arch = Architecture {
        architecture: vec![layer_norm_layer],
    };

    // Scale: γ = 1.0 → ALPHA (α¹ quantised)
    let scale_tensor = Value::Array(vec![Value::from(ALPHA); 16]);
    // B: β = 0.0 → 0 (α² quantised zero)
    let b_tensor = Value::Array(vec![Value::from(0i64); 16]);

    let scale_layer = ONNXLayer {
        id: 0,
        name: "Scale".into(),
        op_type: "Const".into(),
        inputs: vec![],
        outputs: vec![],
        shape: [("Scale".to_string(), vec![16usize])].into_iter().collect(),
        tensor: Some(scale_tensor),
        params: None,
        opset_version_number: -1,
    };

    let b_layer = ONNXLayer {
        id: 1,
        name: "B".into(),
        op_type: "Const".into(),
        inputs: vec![],
        outputs: vec![],
        shape: [("B".to_string(), vec![16usize])].into_iter().collect(),
        tensor: Some(b_tensor),
        params: None,
        opset_version_number: -1,
    };

    let wandb = WANDB {
        w_and_b: vec![scale_layer, b_layer],
    };

    (params, arch, wandb)
}

fn layer_norm_compile(c: &mut Criterion) {
    let (params, arch, wandb) = make_layer_norm_metadata();
    let mut group = c.benchmark_group("layer_norm/compile");
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

fn layer_norm_witness(c: &mut Criterion) {
    let (params, arch, wandb) = make_layer_norm_metadata();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 16];

    let mut group = c.benchmark_group("layer_norm/witness");
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

fn layer_norm_prove(c: &mut Criterion) {
    let (params, arch, wandb) = make_layer_norm_metadata();
    OnnxContext::set_all(arch, params.clone(), Some(wandb));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 16];
    let wb = witness_bn254_from_f64(
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[],
        false,
    )
    .unwrap();

    let mut group = c.benchmark_group("layer_norm/prove");
    group.sample_size(10);
    group.bench_function("bn254", |b| {
        b.iter(|| prove_bn254(&bundle.circuit, &wb.witness, false).unwrap());
    });
    group.finish();
}

/// Construct a minimal single-AveragePool-layer metadata triple without touching any ONNX file.
///
/// Spec:
///   input  x: [1,1,8,8]  — batch=1, channels=1, 8×8 spatial
///   output y: [1,1,4,4]  — after 2×2 kernel, stride=2, no padding
///   kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0]
///   No weights. No rescale.
fn make_averagepool_metadata() -> (CircuitParams, Architecture, WANDB) {
    let params = CircuitParams {
        scale_base: 2,
        scale_exponent: 18,
        rescale_config: HashMap::new(), // no rescale for AveragePool
        inputs: vec![ONNXIO {
            name: "x".into(),
            elem_type: 1,
            shape: vec![1, 1, 8, 8],
        }],
        outputs: vec![ONNXIO {
            name: "y".into(),
            elem_type: 1,
            shape: vec![1, 1, 4, 4],
        }],
        freivalds_reps: 1,
        n_bits_config: HashMap::new(),
        weights_as_inputs: false,
        proof_system: ProofSystem::Expander,
        curve: None,
        logup_chunk_bits: Some(12),
    };

    let averagepool_layer = ONNXLayer {
        id: 0,
        name: "averagepool_0".into(),
        op_type: "AveragePool".into(),
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        shape: [
            ("x", vec![1usize, 1, 8, 8]),
            ("y", vec![1, 1, 4, 4]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect(),
        tensor: None,
        params: Some(Value::Map(vec![
            (
                Value::String("kernel_shape".into()),
                Value::Array(vec![Value::from(2i64), Value::from(2i64)]),
            ),
            (
                Value::String("strides".into()),
                Value::Array(vec![Value::from(2i64), Value::from(2i64)]),
            ),
            (
                Value::String("pads".into()),
                Value::Array(vec![Value::from(0i64); 4]),
            ),
            (
                Value::String("dilations".into()),
                Value::Array(vec![Value::from(1i64), Value::from(1i64)]),
            ),
        ])),
        opset_version_number: 17,
    };

    let arch = Architecture {
        architecture: vec![averagepool_layer],
    };
    let wandb = WANDB { w_and_b: vec![] }; // no weights

    (params, arch, wandb)
}

fn averagepool_compile(c: &mut Criterion) {
    let (params, arch, wandb) = make_averagepool_metadata();
    let mut group = c.benchmark_group("averagepool/compile");
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

fn averagepool_witness(c: &mut Criterion) {
    let (params, arch, wandb) = make_averagepool_metadata();
    OnnxContext::set_all(arch.clone(), params.clone(), Some(wandb.clone()));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("c.bundle");
    compile_bn254(path.to_str().unwrap(), false, Some(params.clone())).unwrap();
    let bundle = read_circuit_msgpack(path.to_str().unwrap()).unwrap();
    let activations = vec![0.0f64; 1 * 1 * 8 * 8];

    let mut group = c.benchmark_group("averagepool/witness");
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

fn averagepool_prove(c: &mut Criterion) {
    let (params, arch, wandb) = make_averagepool_metadata();
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

    let mut group = c.benchmark_group("averagepool/prove");
    group.sample_size(10);
    group.bench_function("bn254", |b| {
        b.iter(|| prove_bn254(&bundle.circuit, &wb.witness, false).unwrap());
    });
    group.finish();
}

criterion_group!(conv_benches, conv_compile, conv_witness, conv_prove);
criterion_group!(gemm_benches, gemm_compile, gemm_witness, gemm_prove);
criterion_group!(softmax_benches, softmax_compile, softmax_witness, softmax_prove);
criterion_group!(
    layer_norm_benches,
    layer_norm_compile,
    layer_norm_witness,
    layer_norm_prove
);
criterion_group!(
    averagepool_benches,
    averagepool_compile,
    averagepool_witness,
    averagepool_prove
);
criterion_main!(
    conv_benches,
    gemm_benches,
    softmax_benches,
    layer_norm_benches,
    averagepool_benches
);
