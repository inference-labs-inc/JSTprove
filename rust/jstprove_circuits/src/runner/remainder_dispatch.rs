use crate::runner::errors::{CliError, RunError};

#[cfg(feature = "remainder")]
use crate::runner::main_runner::get_arg;
#[cfg(feature = "remainder")]
use std::path::Path;

#[cfg(feature = "remainder")]
pub(super) fn get_model_or_circuit(matches: &clap::ArgMatches) -> Result<String, CliError> {
    get_arg(matches, "model").or_else(|_| get_arg(matches, "circuit_path"))
}

#[cfg(feature = "remainder")]
#[allow(clippy::too_many_lines)]
pub(super) fn dispatch(
    matches: &clap::ArgMatches,
    command: &str,
    compress: bool,
) -> Result<(), CliError> {
    match command {
        "run_compile_circuit" | "msgpack_compile" => {
            let model_path = get_arg(matches, "model")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            jstprove_remainder::runner::compile::run(
                Path::new(&model_path),
                Path::new(&circuit_path),
                compress,
            )
            .map_err(|e| RunError::Compile(format!("{e:#}")))?;
        }
        "run_gen_witness" => {
            let model_path = get_model_or_circuit(matches)?;
            let input_path = get_arg(matches, "input")?;
            let witness_path = get_arg(matches, "witness")?;
            jstprove_remainder::runner::witness::run(
                Path::new(&model_path),
                Path::new(&input_path),
                Path::new(&witness_path),
                compress,
            )
            .map_err(|e| RunError::Witness(format!("{e:#}")))?;
        }
        "run_prove_witness" => {
            let model_path = get_model_or_circuit(matches)?;
            let witness_path = get_arg(matches, "witness")?;
            let proof_path = get_arg(matches, "proof")?;
            jstprove_remainder::runner::prove::run(
                Path::new(&model_path),
                Path::new(&witness_path),
                Path::new(&proof_path),
                compress,
            )
            .map_err(|e| RunError::Prove(format!("{e:#}")))?;
        }
        "run_gen_verify" => {
            let model_path = get_model_or_circuit(matches)?;
            let proof_path = get_arg(matches, "proof")?;
            let input_path = get_arg(matches, "input")?;
            jstprove_remainder::runner::verify::run(
                Path::new(&model_path),
                Path::new(&proof_path),
                Path::new(&input_path),
            )
            .map_err(|e| RunError::Verify(format!("{e:#}")))?;
        }
        "run_batch_witness" => {
            let model_path = get_model_or_circuit(matches)?;
            let manifest_path = get_arg(matches, "manifest")?;
            jstprove_remainder::runner::batch::run_batch_witness(
                Path::new(&model_path),
                Path::new(&manifest_path),
                compress,
            )
            .map_err(|e| RunError::Witness(format!("{e:#}")))?;
        }
        "run_batch_prove" => {
            let model_path = get_model_or_circuit(matches)?;
            let manifest_path = get_arg(matches, "manifest")?;
            jstprove_remainder::runner::batch::run_batch_prove(
                Path::new(&model_path),
                Path::new(&manifest_path),
                compress,
            )
            .map_err(|e| RunError::Prove(format!("{e:#}")))?;
        }
        "run_batch_verify" => {
            let model_path = get_model_or_circuit(matches)?;
            let manifest_path = get_arg(matches, "manifest")?;
            jstprove_remainder::runner::batch::run_batch_verify(
                Path::new(&model_path),
                Path::new(&manifest_path),
            )
            .map_err(|e| RunError::Verify(format!("{e:#}")))?;
        }
        "run_pipe_witness" => {
            let model_path = get_model_or_circuit(matches)?;
            jstprove_remainder::runner::pipe::run_pipe_witness(Path::new(&model_path), compress)
                .map_err(|e| RunError::Witness(format!("{e:#}")))?;
        }
        "run_pipe_prove" => {
            let model_path = get_model_or_circuit(matches)?;
            jstprove_remainder::runner::pipe::run_pipe_prove(Path::new(&model_path), compress)
                .map_err(|e| RunError::Prove(format!("{e:#}")))?;
        }
        "run_pipe_verify" => {
            let model_path = get_model_or_circuit(matches)?;
            jstprove_remainder::runner::pipe::run_pipe_verify(Path::new(&model_path))
                .map_err(|e| RunError::Verify(format!("{e:#}")))?;
        }
        "run_debug_witness" => {
            return Err(RunError::Unsupported(
                "debug witness not supported for Remainder backend".into(),
            )
            .into());
        }
        "msgpack_prove" => {
            let model_path = get_model_or_circuit(matches)?;
            let witness_path = get_arg(matches, "witness")?;
            let proof_path = get_arg(matches, "proof")?;
            jstprove_remainder::runner::prove::run(
                Path::new(&model_path),
                Path::new(&witness_path),
                Path::new(&proof_path),
                compress,
            )
            .map_err(|e| RunError::Prove(format!("{e:#}")))?;
        }
        "msgpack_verify" => {
            let model_path = get_model_or_circuit(matches)?;
            let proof_path = get_arg(matches, "proof")?;
            let input_path = get_arg(matches, "input")?;
            jstprove_remainder::runner::verify::run(
                Path::new(&model_path),
                Path::new(&proof_path),
                Path::new(&input_path),
            )
            .map_err(|e| RunError::Verify(format!("{e:#}")))?;
        }
        "msgpack_prove_stdin" => {
            let model_path = get_model_or_circuit(matches)?;
            jstprove_remainder::runner::pipe::run_pipe_prove(Path::new(&model_path), compress)
                .map_err(|e| RunError::Prove(format!("{e:#}")))?;
        }
        "msgpack_verify_stdin" => {
            let model_path = get_model_or_circuit(matches)?;
            jstprove_remainder::runner::pipe::run_pipe_verify(Path::new(&model_path))
                .map_err(|e| RunError::Verify(format!("{e:#}")))?;
        }
        "msgpack_witness_stdin" => {
            let model_path = get_model_or_circuit(matches)?;
            jstprove_remainder::runner::pipe::run_pipe_witness(Path::new(&model_path), compress)
                .map_err(|e| RunError::Witness(format!("{e:#}")))?;
        }
        _ => return Err(CliError::UnknownCommand(command.to_string())),
    }
    Ok(())
}

#[cfg(not(feature = "remainder"))]
pub(super) fn dispatch(
    _matches: &clap::ArgMatches,
    _command: &str,
    _compress: bool,
) -> Result<(), CliError> {
    Err(RunError::Unsupported("remainder backend requires the 'remainder' feature".into()).into())
}
