use std::io::{self, Read};
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::batch::{BatchManifest, BatchResult, WitnessJob, ProveJob, VerifyJob};

fn read_stdin_json<T: for<'de> Deserialize<'de>>() -> Result<T> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    let value: T = serde_json::from_str(&buf)?;
    Ok(value)
}

fn print_result(result: &BatchResult) {
    let json = serde_json::to_string(result).unwrap_or_default();
    println!("{}", json);
}

pub fn run_pipe_witness(model_path: &Path, compress: bool) -> Result<()> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<WitnessJob> = read_stdin_json()?;

    let alpha = model.scale_config.alpha;
    let mut result = BatchResult { succeeded: 0, failed: 0, errors: vec![] };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        match process_pipe_witness_job(&model, job, alpha, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                result.errors.push((idx, format!("{:#}", e)));
                result.failed += 1;
            }
        }
    }

    print_result(&result);
    Ok(())
}

pub fn run_pipe_prove(model_path: &Path, compress: bool) -> Result<()> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<ProveJob> = read_stdin_json()?;

    let mut result = BatchResult { succeeded: 0, failed: 0, errors: vec![] };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        match process_pipe_prove_job(&model, job, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                result.errors.push((idx, format!("{:#}", e)));
                result.failed += 1;
            }
        }
    }

    print_result(&result);
    Ok(())
}

pub fn run_pipe_verify(model_path: &Path) -> Result<()> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<VerifyJob> = read_stdin_json()?;

    let mut result = BatchResult { succeeded: 0, failed: 0, errors: vec![] };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        match process_pipe_verify_job(&model, job) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                result.errors.push((idx, format!("{:#}", e)));
                result.failed += 1;
            }
        }
    }

    print_result(&result);
    Ok(())
}

fn process_pipe_witness_job(
    model: &crate::onnx::quantizer::QuantizedModel,
    job: &WitnessJob,
    alpha: i64,
    compress: bool,
) -> Result<()> {
    let quantized_input = super::witness::load_and_quantize_input(Path::new(&job.input), alpha)?;
    let witness = super::witness::compute_witness(model, &quantized_input)?;
    super::serialization::serialize_to_file(&witness, Path::new(&job.output), compress)?;
    Ok(())
}

fn process_pipe_prove_job(
    model: &crate::onnx::quantizer::QuantizedModel,
    job: &ProveJob,
    compress: bool,
) -> Result<()> {
    let witness_data = super::witness::load_witness(Path::new(&job.witness))?;
    let mut model = model.clone();
    model.apply_observed_n_bits(&witness_data.observed_n_bits);
    let mut proof = super::prove::generate_proof(&model, &witness_data.shreds)?;
    proof.observed_n_bits = witness_data.observed_n_bits;
    super::serialization::serialize_to_file(&proof, Path::new(&job.proof), compress)?;
    Ok(())
}

fn process_pipe_verify_job(
    model: &crate::onnx::quantizer::QuantizedModel,
    job: &VerifyJob,
) -> Result<()> {
    let quantized_input = super::witness::load_and_quantize_input(
        Path::new(&job.input),
        model.scale_config.alpha,
    )?;
    let proof = super::prove::load_proof(Path::new(&job.proof))?;
    let mut model = model.clone();
    model.apply_observed_n_bits(&proof.observed_n_bits);
    super::verify::verify_with_model(&model, &proof, &quantized_input)
}
