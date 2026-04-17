use std::io::Write as _;

use crate::tolerance::Tolerance;

/// A single named test case ready to run.
#[derive(Clone)]
pub struct TestCase {
    pub op_name: &'static str,
    /// Seed used to generate this case — included in failure messages for reproducibility.
    pub seed: u64,
    /// Serialized single-op ONNX model bytes.
    pub onnx_bytes: Vec<u8>,
    /// Flattened input tensors. For INT64-typed ops (indices, shapes) these are the
    /// exact integer values. For FLOAT-typed ops these are α-scaled quantized values
    /// (cast to f64 before passing to JSTProve's witness_f64).
    pub inputs: Vec<Vec<i64>>,
    pub tolerance: Tolerance,
}

/// One element-level mismatch found during a conformance run.
#[derive(Debug, Clone)]
pub struct Failure {
    pub op_name: &'static str,
    pub seed: u64,
    /// Flat index into the output tensor.
    pub index: usize,
    pub reference_value: i64,
    pub jstprove_value: i64,
    pub delta: i64,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "op={} seed={} index={} reference={} jstprove={} delta={}",
            self.op_name,
            self.seed,
            self.index,
            self.reference_value,
            self.jstprove_value,
            self.delta,
        )
    }
}

/// The outcome of running one `TestCase`.
pub enum TestResult {
    Pass,
    Fail(Vec<Failure>),
    Error(anyhow::Error),
}

/// Runs a `TestCase` through both backends and compares outputs.
pub struct ConformanceRunner {
    /// When true, only run the tract reference path (skip JSTProve). Useful for
    /// debugging input generators before the full circuit pipeline is wired.
    pub reference_only: bool,
}

impl ConformanceRunner {
    pub fn run(&self, case: &TestCase) -> TestResult {
        // --- Reference path (tract) ---
        let reference_outputs = match run_tract(case) {
            Ok(o) => o,
            Err(e) => {
                log::error!(
                    "CONFORMANCE ERROR (tract)  op={}  seed={}  error={e:#}",
                    case.op_name,
                    case.seed
                );
                return TestResult::Error(e);
            }
        };

        if self.reference_only {
            return TestResult::Pass;
        }

        // --- JSTProve circuit path ---
        let jstprove_outputs = match run_jstprove(case) {
            Ok(o) => o,
            Err(e) => {
                log::error!(
                    "CONFORMANCE ERROR (jstprove)  op={}  seed={}  error={e:#}",
                    case.op_name,
                    case.seed
                );
                return TestResult::Error(e);
            }
        };

        // --- Element-wise comparison ---
        let mut failures = Vec::new();
        let len = reference_outputs.len().min(jstprove_outputs.len());
        for i in 0..len {
            let ref_val = reference_outputs[i];
            let jst_val = jstprove_outputs[i];
            if !case.tolerance.check(ref_val, jst_val) {
                let delta = (ref_val - jst_val).unsigned_abs() as i64;
                log::error!(
                    "CONFORMANCE FAIL  op={}  seed={}\n  index={}  reference={}  jstprove={}  delta={}\n  To reproduce: seed={}",
                    case.op_name, case.seed, i, ref_val, jst_val, delta, case.seed
                );
                failures.push(Failure {
                    op_name: case.op_name,
                    seed: case.seed,
                    index: i,
                    reference_value: ref_val,
                    jstprove_value: jst_val,
                    delta,
                });
            }
        }

        if reference_outputs.len() != jstprove_outputs.len() {
            log::error!(
                "CONFORMANCE FAIL  op={}  seed={}  output length mismatch: reference={} jstprove={}",
                case.op_name, case.seed,
                reference_outputs.len(), jstprove_outputs.len()
            );
            // Push a synthetic failure for the length mismatch
            failures.push(Failure {
                op_name: case.op_name,
                seed: case.seed,
                index: usize::MAX,
                reference_value: reference_outputs.len() as i64,
                jstprove_value: jstprove_outputs.len() as i64,
                delta: (reference_outputs.len() as i64 - jstprove_outputs.len() as i64).abs(),
            });
        }

        if failures.is_empty() {
            TestResult::Pass
        } else {
            TestResult::Fail(failures)
        }
    }
}

// ---------------------------------------------------------------------------
// tract reference backend
// ---------------------------------------------------------------------------

fn run_tract(case: &TestCase) -> anyhow::Result<Vec<i64>> {
    use tract_onnx::prelude::*;

    // Write ONNX bytes to a named tempfile — tract loads from a path
    let mut tmp = tempfile::NamedTempFile::new()?;
    tmp.write_all(&case.onnx_bytes)?;
    tmp.flush()?;

    let model = tract_onnx::onnx()
        .model_for_path(tmp.path())?
        .into_optimized()?
        .into_runnable()?;

    // Build input tensors from i64 values
    let inputs: TVec<TValue> = case
        .inputs
        .iter()
        .map(|vals| -> anyhow::Result<TValue> {
            let arr =
                tract_ndarray::Array1::from_vec(vals.clone()).into_dyn();
            Ok(Tensor::from(arr).into())
        })
        .collect::<anyhow::Result<_>>()?;

    let outputs = model.run(inputs)?;

    // Flatten all output tensors to a single Vec<i64>
    let mut result = Vec::new();
    for out in outputs.iter() {
        let view = out.to_array_view::<i64>()?;
        result.extend(view.iter().copied());
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// JSTProve circuit path (witness generation only — no proving)
// ---------------------------------------------------------------------------

fn run_jstprove(case: &TestCase) -> anyhow::Result<Vec<i64>> {
    use jstprove_circuits::api;
    use jstprove_circuits::proof_config::ProofConfig;

    // Write ONNX bytes to tempfile — generate_metadata needs a &Path
    let mut onnx_tmp = tempfile::NamedTempFile::with_suffix(".onnx")?;
    onnx_tmp.write_all(&case.onnx_bytes)?;
    onnx_tmp.flush()?;

    // Generate ONNX metadata (circuit params + architecture + weights)
    let meta = api::generate_metadata(onnx_tmp.path())
        .map_err(|e| anyhow::anyhow!("generate_metadata: {e}"))?;

    // Compile circuit to a tempfile path
    // Use persist() so the file stays around for read_circuit_bundle
    let circuit_tmp = tempfile::Builder::new()
        .suffix(".msgpack")
        .tempfile()?;
    let circuit_path = circuit_tmp.path().to_str().unwrap().to_string();

    api::compile(
        &circuit_path,
        ProofConfig::GoldilocksRaw, // fastest config; no commitment overhead
        meta.circuit_params.clone(),
        meta.architecture,
        meta.wandb,
        false, // no zstd compression — faster for tests
    )
    .map_err(|e| anyhow::anyhow!("compile: {e}"))?;

    // Load compiled circuit bundle back from disk
    let bundle = api::read_circuit_bundle(&circuit_path)
        .map_err(|e| anyhow::anyhow!("read_circuit_bundle: {e}"))?;

    // The params embedded in the bundle may be more complete than meta.circuit_params
    // (e.g. the proof_config stamp gets written during compile), so prefer bundle.metadata.
    let params = bundle
        .metadata
        .clone()
        .unwrap_or_else(|| meta.circuit_params.clone());

    // Convert i64 inputs to f64.
    // For INT64-typed ops, witness_f64 detects elem_type=7 and bypasses alpha-scaling,
    // so casting i64→f64 is correct (values are exact integers).
    // For FLOAT-typed ops, inputs are pre-scaled by alpha; dividing by alpha here and
    // letting witness_f64 re-multiply recovers the original values exactly.
    let activations: Vec<f64> = case
        .inputs
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f64))
        .collect();

    // Generate witness — this is the correctness check, not GKR proving
    let wb = api::witness_f64(
        ProofConfig::GoldilocksRaw,
        &bundle.circuit,
        &bundle.witness_solver,
        &params,
        &activations,
        &[], // no external initializers — they are already baked into the ONNX model
        false,
    )
    .map_err(|e| anyhow::anyhow!("witness_f64: {e}"))?;

    wb.output_data
        .ok_or_else(|| anyhow::anyhow!("witness_f64 returned no output_data"))
}
