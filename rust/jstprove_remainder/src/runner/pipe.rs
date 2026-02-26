use std::path::Path;

use anyhow::Result;

use super::batch::{BatchManifest, BatchResult, ProveJob, VerifyJob, WitnessJob};

pub fn run_pipe_witness(model_path: &Path, compress: bool) -> Result<()> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<WitnessJob> = jstprove_io::read_msgpack_stdin()?;

    let alpha = model.scale_config.alpha;
    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        match process_pipe_witness_job(&model, job, alpha, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                result.errors.push((idx, format!("{:#}", e)));
                result.failed += 1;
            }
        }
    }

    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(())
}

pub fn run_pipe_prove(model_path: &Path, compress: bool) -> Result<()> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<ProveJob> = jstprove_io::read_msgpack_stdin()?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        match process_pipe_prove_job(&model, job, compress) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                result.errors.push((idx, format!("{:#}", e)));
                result.failed += 1;
            }
        }
    }

    jstprove_io::write_msgpack_stdout(&result)?;
    Ok(())
}

pub fn run_pipe_verify(model_path: &Path) -> Result<()> {
    let model = super::compile::load_model(model_path)?;
    let manifest: BatchManifest<VerifyJob> = jstprove_io::read_msgpack_stdin()?;

    let mut result = BatchResult {
        succeeded: 0,
        failed: 0,
        errors: vec![],
    };

    for (idx, job) in manifest.jobs.iter().enumerate() {
        match process_pipe_verify_job(&model, job) {
            Ok(()) => result.succeeded += 1,
            Err(e) => {
                result.errors.push((idx, format!("{:#}", e)));
                result.failed += 1;
            }
        }
    }

    jstprove_io::write_msgpack_stdout(&result)?;
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
    jstprove_io::serialize_to_file(&witness, Path::new(&job.output), compress)?;
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
    jstprove_io::serialize_to_file(&proof, Path::new(&job.proof), compress)?;
    Ok(())
}

fn process_pipe_verify_job(
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
