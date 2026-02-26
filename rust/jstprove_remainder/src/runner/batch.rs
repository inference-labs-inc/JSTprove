use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

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

pub fn run_batch_witness(
    model_path: &Path,
    manifest_path: &Path,
    compress: bool,
) -> Result<BatchResult> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<WitnessJob> = jstprove_io::deserialize_from_file(manifest_path)?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };
    let alpha = model.scale_config.alpha;

    for (idx, job) in manifest.jobs.iter().enumerate() {
        tracing::info!("batch witness job {}/{}", idx + 1, manifest.jobs.len());
        match process_witness_job(&model, job, alpha, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                let msg = format!("{:#}", e);
                tracing::error!("job {} failed: {}", idx, msg);
                result.errors.push((idx, msg));
                result.failed += 1;
            }
        }
    }

    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(result)
}

pub fn run_batch_prove(
    model_path: &Path,
    manifest_path: &Path,
    compress: bool,
) -> Result<BatchResult> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<ProveJob> = jstprove_io::deserialize_from_file(manifest_path)?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        tracing::info!("batch prove job {}/{}", idx + 1, manifest.jobs.len());
        match process_prove_job(&model, job, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                let msg = format!("{:#}", e);
                tracing::error!("job {} failed: {}", idx, msg);
                result.errors.push((idx, msg));
                result.failed += 1;
            }
        }
    }

    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(result)
}

pub fn run_batch_verify(model_path: &Path, manifest_path: &Path) -> Result<BatchResult> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<VerifyJob> = jstprove_io::deserialize_from_file(manifest_path)?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        tracing::info!("batch verify job {}/{}", idx + 1, manifest.jobs.len());
        match process_verify_job(&model, job) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                let msg = format!("{:#}", e);
                tracing::error!("job {} failed: {}", idx, msg);
                result.errors.push((idx, msg));
                result.failed += 1;
            }
        }
    }

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
