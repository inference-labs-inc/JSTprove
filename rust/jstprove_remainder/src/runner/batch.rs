use std::path::Path;

use anyhow::Result;
use console::style;
use serde::{Deserialize, Serialize};

use crate::cli::{self, OutputMode};

#[derive(Debug, Deserialize)]
pub struct BatchManifest<T> {
    pub jobs: Vec<T>,
}

#[derive(Debug, Serialize)]
pub struct BatchResult {
    pub succeeded: usize,
    pub failed: usize,
    pub errors: Vec<(usize, String)>,
}

#[derive(Debug, Deserialize)]
pub struct WitnessJob {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Deserialize)]
pub struct ProveJob {
    pub witness: String,
    pub proof: String,
}

#[derive(Debug, Deserialize)]
pub struct VerifyJob {
    pub input: String,
    pub proof: String,
}

fn print_batch_summary(result: &BatchResult, mode: OutputMode) {
    match mode {
        OutputMode::Human => {
            eprintln!();
            let ok = style(format!("{} succeeded", result.succeeded)).green();
            let fail = if result.failed > 0 {
                style(format!("{} failed", result.failed)).red().to_string()
            } else {
                style(format!("{} failed", result.failed)).dim().to_string()
            };
            eprintln!("  {ok}, {fail}");

            for (idx, msg) in &result.errors {
                let label = style(format!("  job {}:", idx + 1)).red();
                eprintln!("{label} {msg}");
            }
        }
        OutputMode::Json => {
            let summary = serde_json::json!({
                "succeeded": result.succeeded,
                "failed": result.failed,
                "errors": result.errors.iter().map(|(idx, msg)| {
                    serde_json::json!({"job": idx + 1, "error": msg})
                }).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string(&summary).unwrap_or_default());
        }
        OutputMode::Quiet => {}
    }
}

pub fn run_batch_witness(
    model_path: &Path,
    manifest_path: &Path,
    compress: bool,
    mode: OutputMode,
) -> Result<BatchResult> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<WitnessJob> = jstprove_io::deserialize_from_file(manifest_path)?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };
    let alpha = model.scale_config.alpha;
    let pb = cli::progress_bar(manifest.jobs.len() as u64, "witness", mode);

    for (idx, job) in manifest.jobs.iter().enumerate() {
        pb.set_message(format!("job {}", idx + 1));
        match process_witness_job(&model, job, alpha, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                let msg = format!("{e:#}");
                result.errors.push((idx, msg));
                result.failed += 1;
            }
        }
        pb.inc(1);
    }

    pb.finish_and_clear();
    print_batch_summary(&result, mode);
    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(result)
}

pub fn run_batch_prove(
    model_path: &Path,
    manifest_path: &Path,
    compress: bool,
    mode: OutputMode,
) -> Result<BatchResult> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<ProveJob> = jstprove_io::deserialize_from_file(manifest_path)?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };
    let pb = cli::progress_bar(manifest.jobs.len() as u64, "prove", mode);

    for (idx, job) in manifest.jobs.iter().enumerate() {
        pb.set_message(format!("job {}", idx + 1));
        match process_prove_job(&model, job, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                let msg = format!("{e:#}");
                result.errors.push((idx, msg));
                result.failed += 1;
            }
        }
        pb.inc(1);
    }

    pb.finish_and_clear();
    print_batch_summary(&result, mode);
    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(result)
}

pub fn run_batch_verify(
    model_path: &Path,
    manifest_path: &Path,
    mode: OutputMode,
) -> Result<BatchResult> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<VerifyJob> = jstprove_io::deserialize_from_file(manifest_path)?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };
    let pb = cli::progress_bar(manifest.jobs.len() as u64, "verify", mode);

    for (idx, job) in manifest.jobs.iter().enumerate() {
        pb.set_message(format!("job {}", idx + 1));
        match process_verify_job(&model, job) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                let msg = format!("{e:#}");
                result.errors.push((idx, msg));
                result.failed += 1;
            }
        }
        pb.inc(1);
    }

    pb.finish_and_clear();
    print_batch_summary(&result, mode);
    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(result)
}

fn process_witness_job(
    model: &crate::onnx::quantizer::QuantizedModel,
    job: &WitnessJob,
    alpha: i64,
    compress: bool,
) -> Result<()> {
    let quantized_input = super::witness::load_and_quantize_input(Path::new(&job.input), alpha)?;
    let witness = super::witness::compute_witness(model, &quantized_input)?;
    jstprove_io::serialize_to_file(&witness, Path::new(&job.output), compress)?;
    Ok(())
}

fn process_prove_job(
    model: &crate::onnx::quantizer::QuantizedModel,
    job: &ProveJob,
    compress: bool,
) -> Result<()> {
    let witness_data = super::witness::load_witness(Path::new(&job.witness))?;
    let mut model = model.clone();
    model.apply_observed_n_bits(&witness_data.observed_n_bits);
    let mut proof = super::prove::generate_proof(&model, &witness_data.shreds)?;
    proof.observed_n_bits = witness_data.observed_n_bits;
    jstprove_io::serialize_to_file(&proof, Path::new(&job.proof), compress)?;
    Ok(())
}

fn process_verify_job(
    model: &crate::onnx::quantizer::QuantizedModel,
    job: &VerifyJob,
) -> Result<()> {
    let quantized_input =
        super::witness::load_and_quantize_input(Path::new(&job.input), model.scale_config.alpha)?;
    let proof = super::prove::load_proof(Path::new(&job.proof))?;
    let mut model = model.clone();
    model.apply_observed_n_bits(&proof.observed_n_bits);
    super::verify::verify_with_model(&model, &proof, &quantized_input)
}
