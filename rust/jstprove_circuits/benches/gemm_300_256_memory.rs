//! Reproducer for the rfdetr-base slice 310 compile OOM.
//!
//! Slice 310 is a single decoder-layer self-attention output projection:
//!
//!   Gemm(A=[300, 256], B=[256, 256], C=[256]) -> Reshape -> Transpose -> [1, 300, 256]
//!
//! Under goldilocks_ext4_whir compilation on a 512 GB Mac Studio, this
//! slice consumes > 380 GB of compressed memory before macOS Jetsam
//! kills the process. This bench isolates the shape so before/after
//! measurements are reproducible.
//!
//! Run with:
//!
//!   cargo bench --bench gemm_300_256_memory --features peak-mem
//!
//! peak-mem wires peakmem-alloc as the global allocator and exposes
//! `GLOBAL.reset_peak_memory()` + `get_peak_memory()` on the compile
//! window. Without the feature the bench still reports wall time.

use std::path::Path;
use std::time::Instant;

#[cfg(feature = "peak-mem")]
use peakmem_alloc::{INSTRUMENTED_SYSTEM, PeakMemAlloc, PeakMemAllocTrait};
#[cfg(feature = "peak-mem")]
use std::alloc::System;

#[cfg(feature = "peak-mem")]
#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::compile_goldilocks_whir_pq;

const ONNX_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/slice_310_gemm.onnx");

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

#[cfg(feature = "peak-mem")]
fn fmt_mb(bytes: usize) -> String {
    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
}

fn main() {
    let onnx_path = Path::new(ONNX_PATH);
    assert!(
        onnx_path.exists(),
        "missing fixture: {}",
        onnx_path.display()
    );

    println!("fixture: {}", onnx_path.display());

    let metadata =
        expander_metadata::generate_from_onnx(onnx_path).expect("generate_from_onnx failed");
    let params = metadata.circuit_params.clone();

    println!(
        "circuit_params inputs/outputs: {} / {}",
        params.inputs.len(),
        params.outputs.len()
    );
    for inp in &params.inputs {
        println!("  input  {}: shape={:?}", inp.name, inp.shape);
    }
    for out in &params.outputs {
        println!("  output {}: shape={:?}", out.name, out.shape);
    }
    println!(
        "architecture layers: {}",
        metadata.architecture.architecture.len()
    );
    for layer in &metadata.architecture.architecture {
        println!("  layer op_type={:?} name={:?}", layer.op_type, layer.name);
    }

    OnnxContext::set_all(
        metadata.architecture.clone(),
        params.clone(),
        Some(metadata.wandb.clone()),
    );

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let circuit_path = tmp.path().join("circuit.bundle");
    let circuit_path_str = circuit_path.to_str().unwrap();

    #[cfg(feature = "peak-mem")]
    GLOBAL.reset_peak_memory();

    let t0 = Instant::now();
    let result = compile_goldilocks_whir_pq(circuit_path_str, false, Some(params));
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "peak-mem")]
    let peak = GLOBAL.get_peak_memory();

    match result {
        Ok(()) => {
            println!("compile OK in {}", fmt_ms(elapsed_ms));
        }
        Err(e) => {
            println!("compile FAILED in {}: {e:?}", fmt_ms(elapsed_ms));
        }
    }

    #[cfg(feature = "peak-mem")]
    println!("peak compile-time memory: {}", fmt_mb(peak));

    #[cfg(not(feature = "peak-mem"))]
    println!(
        "(build with `--features peak-mem` to get peak memory numbers; \
         this run only reports wall time)"
    );
}
