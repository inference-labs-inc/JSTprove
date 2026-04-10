//! Holographic-GKR setup benchmarks against synthetic circuits
//! sized to match real ML model layer dimensions.
//!
//! These benchmarks exist because the Phase 2a wiring extractor
//! does not yet support the const_ / uni gate kinds and Random
//! coefficients that the expander_compiler emits for production
//! ONNX models like LeNet and MiniResnet. Running setup directly
//! on those compiled circuits would benchmark the wiring
//! extractor's failure path, which is uninformative.
//!
//! Instead, we build synthetic circuits with only `mul` and `add`
//! gates whose per-layer dimensions are sized to match the layer
//! footprints of real LeNet and MiniResnet circuits. This
//! exercises the cryptographic core (wiring extraction +
//! sparse_commit per layer + VK serialization) at production
//! scale and gives a credible wallclock measurement of what
//! `setup_holographic_vk` would cost on the real models once the
//! gate-kind extensions land.
//!
//! The benchmarks are gated behind `--ignored` so they only run
//! when explicitly requested via `cargo test -- --ignored`. They
//! print to stderr; the assertion is just that the setup
//! completes without panicking.

use std::time::Instant;

use circuit::{Circuit, CircuitLayer, CoefType, GateAdd, GateMul, StructureInfo};
use gkr::holographic::setup;
use gkr_engine::GoldilocksExt4x1Config;
use goldilocks::Goldilocks;
use serdes::ExpSerde;
type C = GoldilocksExt4x1Config;

/// Per-layer specification: `(input_var_num, output_var_num,
/// num_mul_gates, num_add_gates)`.
struct LayerSpec {
    input_var_num: usize,
    output_var_num: usize,
    num_mul: usize,
    num_add: usize,
}

fn build_layer(spec: &LayerSpec) -> CircuitLayer<C> {
    let mut layer = CircuitLayer::<C> {
        input_var_num: spec.input_var_num,
        output_var_num: spec.output_var_num,
        input_vals: Vec::new(),
        output_vals: Vec::new(),
        mul: Vec::with_capacity(spec.num_mul),
        add: Vec::with_capacity(spec.num_add),
        const_: Vec::new(),
        uni: Vec::new(),
        structure_info: StructureInfo::default(),
    };
    let m_in = 1usize << spec.input_var_num;
    let m_out = 1usize << spec.output_var_num;
    // Synthetic gate placement: walk a deterministic stride through
    // the address space so the address vectors are non-trivial
    // (exercising the offline-memory-checking gadget at scale).
    for k in 0..spec.num_mul {
        let o = (k * 7919) % m_out;
        let x = (k * 6469) % m_in;
        let y = (k * 5113) % m_in;
        layer.mul.push(GateMul {
            i_ids: [x, y],
            o_id: o,
            coef_type: CoefType::Constant,
            coef: Goldilocks::from((k as u64) + 1),
            gate_type: 0,
        });
    }
    for k in 0..spec.num_add {
        let o = (k * 4051) % m_out;
        let x = (k * 3203) % m_in;
        layer.add.push(GateAdd {
            i_ids: [x],
            o_id: o,
            coef_type: CoefType::Constant,
            coef: Goldilocks::from((k as u64) + 1),
            gate_type: 0,
        });
    }
    layer
}

fn build_synthetic_circuit(specs: &[LayerSpec]) -> Circuit<C> {
    Circuit {
        layers: specs.iter().map(build_layer).collect(),
        public_input: Vec::new(),
        expected_num_output_zeros: 0,
        rnd_coefs_identified: false,
        rnd_coefs: Vec::new(),
    }
}

/// LeNet-shaped synthetic circuit. Layer dimensions approximate
/// the per-layer footprint of a compiled LeNet:
///
///   conv1     : 784 → 4704     (28×28×1   → 28×28×6)
///   maxpool1  : 4704 → 1176    (28×28×6   → 14×14×6)
///   conv2     : 1176 → 1600    (14×14×6   → 10×10×16)
///   maxpool2  : 1600 → 400     (10×10×16  → 5×5×16)
///   gemm1     : 400 → 128      (after flatten)
///   gemm2     : 128 → 84
///   gemm3     : 84 → 16        (rounded up to power of two for
///                               the synthetic shape)
///
/// `num_mul` and `num_add` per layer are sized to nnz = 1024 so
/// the WHIR per-layer commit operates on a 8192-element combined
/// polynomial, comparable to what a real LeNet layer commit would
/// see.
fn lenet_shaped_circuit() -> Circuit<C> {
    build_synthetic_circuit(&[
        // log2-rounded versions of the dimensions above
        LayerSpec {
            input_var_num: 10,  // 1024
            output_var_num: 13, // 8192
            num_mul: 1024,
            num_add: 1024,
        },
        LayerSpec {
            input_var_num: 13,
            output_var_num: 11, // 2048
            num_mul: 1024,
            num_add: 1024,
        },
        LayerSpec {
            input_var_num: 11,
            output_var_num: 11,
            num_mul: 1024,
            num_add: 1024,
        },
        LayerSpec {
            input_var_num: 11,
            output_var_num: 9, // 512
            num_mul: 1024,
            num_add: 1024,
        },
        LayerSpec {
            input_var_num: 9,
            output_var_num: 7, // 128
            num_mul: 512,
            num_add: 512,
        },
        LayerSpec {
            input_var_num: 7,
            output_var_num: 7,
            num_mul: 512,
            num_add: 512,
        },
        LayerSpec {
            input_var_num: 7,
            output_var_num: 4, // 16
            num_mul: 128,
            num_add: 128,
        },
    ])
}

/// MiniResnet-shaped synthetic circuit. 12 layers (5 conv blocks,
/// some residual adds, a final gemm), per-layer footprints sized
/// down a bit relative to LeNet to reflect MiniResnet's smaller
/// per-layer parameter count (~6K total vs LeNet's ~62K).
fn miniresnet_shaped_circuit() -> Circuit<C> {
    let conv_block = || LayerSpec {
        input_var_num: 10,
        output_var_num: 10,
        num_mul: 512,
        num_add: 512,
    };
    let residual_add = || LayerSpec {
        input_var_num: 10,
        output_var_num: 10,
        num_mul: 0,
        num_add: 1024,
    };
    let pool = || LayerSpec {
        input_var_num: 10,
        output_var_num: 9,
        num_mul: 256,
        num_add: 256,
    };
    let gemm = || LayerSpec {
        input_var_num: 9,
        output_var_num: 4,
        num_mul: 256,
        num_add: 256,
    };
    build_synthetic_circuit(&[
        conv_block(),
        conv_block(),
        residual_add(),
        conv_block(),
        conv_block(),
        residual_add(),
        pool(),
        conv_block(),
        conv_block(),
        residual_add(),
        pool(),
        gemm(),
    ])
}

fn run_setup_benchmark(name: &str, circuit: Circuit<C>) {
    let n_layers = circuit.layers.len();
    let total_mul: usize = circuit.layers.iter().map(|l| l.mul.len()).sum();
    let total_add: usize = circuit.layers.iter().map(|l| l.add.len()).sum();

    let t = Instant::now();
    let (pk, vk) = setup::<C>(circuit).expect("setup must succeed on synthetic circuit");
    let setup_elapsed = t.elapsed();

    let mut vk_bytes = Vec::new();
    let t = Instant::now();
    vk.serialize_into(&mut vk_bytes).unwrap();
    let serialize_elapsed = t.elapsed();

    eprintln!();
    eprintln!("=========================================");
    eprintln!("Holographic GKR setup benchmark — {name}");
    eprintln!("=========================================");
    eprintln!("Layers           : {n_layers}");
    eprintln!("Total mul gates  : {total_mul}");
    eprintln!("Total add gates  : {total_add}");
    eprintln!("Setup wallclock  : {:.3} s", setup_elapsed.as_secs_f64());
    eprintln!(
        "Per-layer mean   : {:.3} ms",
        (setup_elapsed.as_secs_f64() * 1000.0) / (n_layers as f64)
    );
    eprintln!(
        "VK serialize     : {:.3} ms",
        serialize_elapsed.as_secs_f64() * 1000.0
    );
    eprintln!("VK size          : {} bytes", vk_bytes.len());
    eprintln!(
        "VK size per layer: {:.1} bytes",
        (vk_bytes.len() as f64) / (n_layers as f64)
    );
    eprintln!();

    // Sanity check the VK shape
    assert_eq!(vk.layers.len(), n_layers);
    // PK is dropped here without consumption — we just want the
    // setup wallclock plus the VK shape.
    drop(pk);
}

#[test]
#[ignore = "benchmark — run with `cargo test --test holographic_benchmarks -- --ignored --nocapture`"]
fn benchmark_setup_lenet_shaped() {
    run_setup_benchmark("LeNet-shaped (synthetic mul/add)", lenet_shaped_circuit());
}

#[test]
#[ignore = "benchmark — run with `cargo test --test holographic_benchmarks -- --ignored --nocapture`"]
fn benchmark_setup_miniresnet_shaped() {
    run_setup_benchmark(
        "MiniResnet-shaped (synthetic mul/add)",
        miniresnet_shaped_circuit(),
    );
}
