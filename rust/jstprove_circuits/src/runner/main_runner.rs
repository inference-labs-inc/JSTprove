#[cfg(feature = "peak-mem")]
use std::alloc::System;
use std::borrow::Cow;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use clap::{Arg, Command};
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::{Circuit, NormalInputType};
use expander_compiler::frontend::{
    ChallengeField, CircuitField, CompileOptions, Config, Define, Variable, WitnessSolver, compile,
    extra::debug_eval, internal::DumpLoadTwoVariables,
};
use expander_compiler::gkr_engine::{FieldEngine, GKREngine, MPIConfig};
use expander_compiler::serdes::ExpSerde;
use io_reader::IOReader;
#[cfg(feature = "peak-mem")]
use peakmem_alloc::{INSTRUMENTED_SYSTEM, PeakMemAlloc, PeakMemAllocTrait};
use rmpv::Value;
use serde::{Deserialize, Serialize};

use crate::circuit_functions::hints::build_logup_hint_registry;
use crate::circuit_functions::utils::onnx_model::CircuitParams;
use crate::io::io_reader;
use crate::runner::errors::{CliError, RunError};
use crate::runner::schema::{
    CompiledCircuit, ProofBundle, ProveRequest, VerifyRequest, VerifyResponse, WitnessBundle,
    WitnessRequest,
};
use crate::runner::version::jstprove_artifact_version;
use expander_compiler::expander_binary::executor;

pub(crate) fn auto_decompress_bytes(data: &[u8]) -> Result<Cow<[u8]>, RunError> {
    jstprove_io::auto_decompress_bytes(data)
        .map_err(|e| RunError::Deserialize(format!("zstd decompress: {e}")))
}

fn maybe_compress_bytes(data: Vec<u8>, compress: bool) -> Result<Vec<u8>, RunError> {
    jstprove_io::maybe_compress_bytes(data, compress)
        .map_err(|e| RunError::Serialize(format!("zstd compress: {e}")))
}

fn write_msgpack_stdout<T: Serialize>(value: &T) -> Result<(), RunError> {
    jstprove_io::write_msgpack_stdout(value).map_err(|e| RunError::Serialize(format!("{e:#}")))
}

fn compressed_writer(
    file: std::fs::File,
    compress: bool,
) -> Result<jstprove_io::MaybeCompressed, RunError> {
    jstprove_io::compressed_writer(file, compress)
        .map_err(|e| RunError::Serialize(format!("zstd encoder: {e:#}")))
}

fn auto_reader(file: std::fs::File) -> Result<Box<dyn std::io::Read>, RunError> {
    jstprove_io::auto_reader(file).map_err(|e| RunError::Deserialize(format!("auto reader: {e:#}")))
}

#[cfg(feature = "peak-mem")]
#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

fn get_witness_solver_path(input: &str) -> PathBuf {
    let path = Path::new(input);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(input);
    let parent = path.parent().and_then(|s| s.to_str()).unwrap_or(input);

    let file_name = if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        format!("{parent}/{stem}_witness_solver.{ext}")
    } else {
        format!("{parent}/{stem}_witness_solver")
    };

    PathBuf::from(file_name)
}

fn compile_to_bundle<C: Config, CircuitType>(
    metadata: Option<CircuitParams>,
) -> Result<CompiledCircuit, RunError>
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone + MaybeConfigure,
{
    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit)?;

    let compile_result = compile(&circuit, CompileOptions::default())
        .map_err(|e| RunError::Compile(format!("{e:?}")))?;

    let mut circuit_buf = Vec::new();
    compile_result
        .layered_circuit
        .serialize_into(&mut circuit_buf)
        .map_err(|e| RunError::Serialize(format!("circuit: {e:?}")))?;

    let mut ws_buf = Vec::new();
    compile_result
        .witness_solver
        .serialize_into(&mut ws_buf)
        .map_err(|e| RunError::Serialize(format!("witness_solver: {e:?}")))?;

    Ok(CompiledCircuit {
        circuit: circuit_buf,
        witness_solver: ws_buf,
        metadata,
        version: Some(jstprove_artifact_version()),
    })
}

/// Compiles a circuit and writes a single msgpack bundle.
///
/// The output path is derived from `circuit_path` with the extension
/// replaced by `.msgpack`. When `compress` is true the bundle is
/// zstd-compressed; otherwise it is written as raw msgpack.
///
/// # Errors
///
/// Returns a [`RunError`] on compilation, serialization, or file I/O failure.
pub fn run_compile_and_serialize<C: Config, CircuitType>(
    circuit_path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError>
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone + MaybeConfigure,
{
    #[cfg(feature = "peak-mem")]
    GLOBAL.reset_peak_memory();

    let bundle = compile_to_bundle::<C, CircuitType>(metadata)?;
    let msgpack_path = Path::new(circuit_path).with_extension("msgpack");
    write_circuit_bundle(
        msgpack_path.to_str().ok_or_else(|| {
            RunError::Unsupported("msgpack output path is not valid UTF-8".into())
        })?,
        &bundle,
        compress,
    )
}

/// Generates a witness for a circuit from provided inputs and outputs, then writes it to disk.
///
/// This function loads the witness solver and circuit, reads inputs and expected
/// outputs, and computes the witness assignments. It also runs
/// a sanity check by evaluating the circuit with the generated witness to ensure
/// that the outputs match the expected values. Finally, the witness is serialized
/// and written to the specified output file.
///
/// # Arguments
///
/// - `io_reader` – Reader instance used to load inputs/outputs.
/// - `input_path` – Path to the input file containing circuit inputs.
/// - `output_path` – Path to the output file containing expected circuit outputs.
/// - `witness_path` – Path where the generated witness will be written.
/// - `circuit_path` – Path to the serialized circuit definition.
///
/// # Errors
///
/// Returns a [`RunError`] if:
///
/// - The witness solver or circuit file cannot be opened (`RunError::Io`)
/// - The witness solver or circuit cannot be deserialized (`RunError::Deserialize`)
/// - The witness generation fails (`RunError::Witness`)
/// - The generated witness fails the sanity check (outputs don’t match inputs)
/// - The witness cannot be serialized (`RunError::Serialize`)
///
pub fn run_witness<C: Config, I, CircuitDefaultType>(
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    witness_path: &str,
    circuit_path: &str,
    compress: bool,
) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let (layered_circuit, witness_solver) = load_circuit_and_solver::<C>(circuit_path)?;

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    #[cfg(feature = "peak-mem")]
    GLOBAL.reset_peak_memory();
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    witness_core::<C, CircuitDefaultType>(
        &witness_solver,
        &layered_circuit,
        &hint_registry,
        &assignment,
        witness_path,
        compress,
    )?;

    println!("Witness Generated");
    Ok(())
}

/// Debugs witness generation by running a circuit evaluation with the given inputs and outputs.
///
/// This function is similar to [`run_witness`], but intended for debugging purposes.
/// It loads the witness solver and circuit, reads inputs/outputs, and generates
/// a witness. Additionally, it instantiates the circuit of type `CircuitType`
/// and performs a debug evaluation with [`debug_eval`].
///
/// If the generated witness fails to satisfy the circuit (i.e., outputs do not match),
/// the function returns an error. Unlike [`run_witness`], this does not serialize.
/// Running `debug_witness`, will print any `api.display` calls to console.
///
/// # Arguments
///
/// - `io_reader` – Reader instance used to load inputs/outputs.
/// - `input_path` – Path to the input file containing circuit inputs.
/// - `output_path` – Path to the output file containing expected circuit outputs.
/// - `_witness_path` – (Unused) Placeholder path for the witness file.
/// - `circuit_path` – Path to the serialized circuit definition.
///
/// # Errors
///
/// Returns a [`RunError`] if:
///
/// - The witness solver or circuit file cannot be opened (`RunError::Io`)
/// - The witness solver or circuit cannot be deserialized (`RunError::Deserialize`)
/// - Witness generation fails (`RunError::Witness`)
/// - The generated witness fails to satisfy the circuit
///
pub fn debug_witness<C: Config, I, CircuitDefaultType, CircuitType>(
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    _witness_path: &str,
    circuit_path: &str,
) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
    CircuitType: Default
        + DumpLoadTwoVariables<Variable>
        + expander_compiler::frontend::Define<C>
        + Clone
        + MaybeConfigure,
{
    let (layered_circuit, witness_solver) = load_circuit_and_solver::<C>(circuit_path)?;

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit)?;

    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    debug_eval(&circuit, &assignment, hint_registry.clone());

    let witness = witness_solver
        .solve_witness_with_hints(&assignment, &hint_registry)
        .map_err(|e| RunError::Witness(format!("{e:?}")))?;

    let output = layered_circuit.run(&witness);
    for x in &output {
        if !(*x) {
            return Err(RunError::Witness("Debug witness generation failed".into()));
        }
    }
    Ok(())
}

/// Generates a proof for a circuit using a provided witness.
///
/// This function loads a circuit definition and witness, evaluates
/// the circuit, and produces a cryptographic proof of correctness. The proof is
/// then serialized and written to the specified output file. This function uses
/// Expander's backend prover to compute this proof.
///
/// # Arguments
///
/// - `circuit_path` – Path to the serialized circuit file.
/// - `witness_path` – Path to the witness file.
/// - `proof_path` – Path where the generated proof will be written.
///
/// # Errors
///
/// Returns a [`RunError`] if:
///
/// - The circuit or witness file cannot be opened (`RunError::Io`)
/// - The circuit or witness cannot be deserialized (`RunError::Deserialize`)
/// - The proof cannot be serialized to bytes or written to disk (`RunError::Serialize`)
///
pub fn run_prove_witness<C: Config, CircuitDefaultType>(
    circuit_path: &str,
    witness_path: &str,
    proof_path: &str,
    compress: bool,
) -> Result<(), RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    #[cfg(feature = "peak-mem")]
    GLOBAL.reset_peak_memory();
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;
    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    prove_core::<C>(&mut expander_circuit, witness_path, proof_path, compress)?;
    println!("Proved");
    Ok(())
}

/// Verifies a proof against a circuit, witness, and expected I/O assignments.
///
/// This function loads a circuit definition, a witness, and a proof, and
/// verifies that the proof is valid with respect to the circuit and
/// provided inputs/outputs. It uses the given [`IOReader`] to deserialize the
/// input/output assignments into the default circuit representation.
///
/// The verifier also performs consistency checks:
/// - Ensures that public inputs derived from the witness match those from the
///   expander circuit.
/// - Ensures that the witness matches the claimed output values.
///
/// # Arguments
///
/// - `circuit_path` – Path to the serialized circuit file.
/// - `io_reader` – The I/O reader used to load inputs and outputs.
/// - `input_path` – Path to the msgpack-encoded input file.
/// - `output_path` – Path to the msgpack-encoded output file.
/// - `witness_path` – Path to the witness file.
/// - `proof_path` – Path to the proof file.
///
/// # Errors
///
/// Returns a [`RunError`] if:
///
/// - The circuit, witness, or proof file cannot be opened (`RunError::Io`)
/// - Deserialization of the circuit, witness, or proof fails (`RunError::Deserialize`)
/// - The proof fails to verify (`RunError::Verify`)
///
fn run_verify_io<C: Config, I, CircuitDefaultType>(
    circuit_path: &str,
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    witness_path: &str,
    proof_path: &str,
) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    #[cfg(feature = "peak-mem")]
    GLOBAL.reset_peak_memory();
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;
    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    verify_core::<C, I, CircuitDefaultType>(
        &mut expander_circuit,
        io_reader,
        input_path,
        output_path,
        witness_path,
        proof_path,
    )?;
    println!("Verified");
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct WitnessJob {
    pub input: String,
    pub output: String,
    pub witness: String,
}

#[derive(Debug, Deserialize)]
pub struct ProveJob {
    pub witness: String,
    pub proof: String,
}

#[derive(Debug, Deserialize)]
pub struct VerifyJob {
    pub input: String,
    pub output: String,
    pub witness: String,
    pub proof: String,
}

#[derive(Debug, Deserialize)]
pub struct PipeWitnessJob {
    pub input: Value,
    pub output: Value,
    pub witness: String,
}

#[derive(Debug, Deserialize)]
pub struct PipeVerifyJob {
    pub input: Value,
    pub output: Value,
    pub witness: String,
    pub proof: String,
}

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

fn check_batch_result(operation: &str, result: &BatchResult) -> Result<(), CliError> {
    println!(
        "Batch {operation} complete: {} succeeded, {} failed",
        result.succeeded, result.failed
    );
    if result.failed > 0 {
        let msgs: Vec<_> = result
            .errors
            .iter()
            .map(|(idx, msg)| format!("  job {idx}: {msg}"))
            .collect();
        return Err(CliError::Other(format!(
            "Batch {operation}: {} job(s) failed:\n{}",
            result.failed,
            msgs.join("\n")
        )));
    }
    Ok(())
}

fn load_manifest<T: serde::de::DeserializeOwned>(path: &str) -> Result<BatchManifest<T>, RunError> {
    let file = std::fs::File::open(path).map_err(|e| RunError::Io {
        source: e,
        path: path.into(),
    })?;
    let reader = auto_reader(file)?;
    rmp_serde::from_read(reader).map_err(|e| RunError::Deserialize(format!("{e:?}")))
}

fn load_layered_circuit<C: Config>(path: &str) -> Result<Circuit<C, NormalInputType>, RunError> {
    let p = Path::new(path);
    if p.extension().and_then(|e| e.to_str()) == Some("msgpack") {
        let bundle = read_circuit_msgpack(path)?;
        return load_circuit_from_bytes::<C>(&bundle.circuit);
    }
    if p.exists() {
        let file = std::fs::File::open(path).map_err(|e| RunError::Io {
            source: e,
            path: path.into(),
        })?;
        let reader = auto_reader(file)?;
        return Circuit::<C, NormalInputType>::deserialize_from(reader)
            .map_err(|e| RunError::Deserialize(format!("{e:?}")));
    }
    let msgpack_path = p.with_extension("msgpack");
    let bundle =
        read_circuit_msgpack(msgpack_path.to_str().ok_or_else(|| {
            RunError::Unsupported("msgpack output path is not valid UTF-8".into())
        })?)?;
    load_circuit_from_bytes::<C>(&bundle.circuit)
}

fn load_circuit_and_solver<C: Config>(
    circuit_path: &str,
) -> Result<(Circuit<C, NormalInputType>, WitnessSolver<C>), RunError> {
    let p = Path::new(circuit_path);
    if p.extension().and_then(|e| e.to_str()) == Some("msgpack") {
        let bundle = read_circuit_msgpack(circuit_path)?;
        let circuit = load_circuit_from_bytes::<C>(&bundle.circuit)?;
        let solver = load_witness_solver_from_bytes::<C>(&bundle.witness_solver)?;
        return Ok((circuit, solver));
    }
    let witness_file = get_witness_solver_path(circuit_path);
    if p.exists() && witness_file.exists() {
        let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io {
            source: e,
            path: circuit_path.into(),
        })?;
        let reader = auto_reader(file)?;
        let circuit = Circuit::<C, NormalInputType>::deserialize_from(reader)
            .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;
        let file = std::fs::File::open(&witness_file).map_err(|e| RunError::Io {
            source: e,
            path: witness_file.display().to_string(),
        })?;
        let reader = auto_reader(file)?;
        let solver = WitnessSolver::<C>::deserialize_from(reader)
            .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;
        return Ok((circuit, solver));
    }
    let msgpack_path = p.with_extension("msgpack");
    if msgpack_path.exists() {
        let bundle = read_circuit_msgpack(msgpack_path.to_str().ok_or_else(|| {
            RunError::Unsupported("msgpack output path is not valid UTF-8".into())
        })?)?;
        let circuit = load_circuit_from_bytes::<C>(&bundle.circuit)?;
        let solver = load_witness_solver_from_bytes::<C>(&bundle.witness_solver)?;
        return Ok((circuit, solver));
    }
    Err(RunError::Io {
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no witness solver file or msgpack bundle found",
        ),
        path: circuit_path.into(),
    })
}

fn prove_core<C: Config>(
    expander_circuit: &mut expander_compiler::expander_circuit::Circuit<C::FieldConfig>,
    witness_path: &str,
    proof_path: &str,
    compress: bool,
) -> Result<(), RunError> {
    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let reader = auto_reader(file)?;
    let witness: Witness<C> =
        Witness::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input.clone_from(&simd_public_input);
    expander_circuit.evaluate();

    let mpi_config = MPIConfig::prover_new();
    let (claimed_v, proof) = executor::prove::<C>(expander_circuit, mpi_config);

    let proof_bytes: Vec<u8> = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| RunError::Serialize(format!("Error serializing proof {e:?}")))?;

    let file = std::fs::File::create(proof_path).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;
    let mut writer = compressed_writer(file, compress)?;
    proof_bytes
        .serialize_into(&mut writer)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;
    writer
        .finish()
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;

    Ok(())
}

fn verify_core<C: Config, I, CircuitDefaultType>(
    expander_circuit: &mut expander_compiler::expander_circuit::Circuit<C::FieldConfig>,
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    witness_path: &str,
    proof_path: &str,
) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    let mut vars: Vec<_> = Vec::new();
    let mut public_vars: Vec<_> = Vec::new();
    assignment.dump_into(&mut vars, &mut public_vars);

    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let reader = auto_reader(file)?;
    let witness = Witness::<C>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0]
        .input_vals
        .clone_from(&simd_input);
    expander_circuit.public_input.clone_from(&simd_public_input);

    for (i, _) in public_vars.iter().enumerate() {
        let x = format!("{:?}", public_vars[i]);
        let y = format!("{:?}", expander_circuit.public_input[i]);
        if x != y {
            return Err(RunError::Verify(
                "inputs/outputs don't match the witness".into(),
            ));
        }
    }

    let file = std::fs::File::open(proof_path).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;
    let reader = auto_reader(file)?;
    let proof_and_claimed_v: Vec<u8> =
        Vec::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let (proof, claimed_v) =
        executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v)
            .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let mpi_config = MPIConfig::verifier_new(1);
    if !executor::verify::<C>(expander_circuit, mpi_config, &proof, &claimed_v) {
        return Err(RunError::Verify("Verification failed".into()));
    }

    Ok(())
}

/// # Errors
/// Returns `RunError` on witness solving or validation failure.
pub fn solve_and_validate_witness<C: Config, CircuitDefaultType>(
    witness_solver: &WitnessSolver<C>,
    layered_circuit: &Circuit<C, NormalInputType>,
    hint_registry: &expander_compiler::frontend::HintRegistry<CircuitField<C>>,
    assignment: &CircuitDefaultType,
) -> Result<Witness<C>, RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let witness = witness_solver
        .solve_witness_with_hints(assignment, hint_registry)
        .map_err(|e| RunError::Witness(format!("{e:?}")))?;

    let output = layered_circuit.run(&witness);
    for x in &output {
        if !(*x) {
            return Err(RunError::Witness(
                "Outputs generated do not match outputs supplied".into(),
            ));
        }
    }

    Ok(witness)
}

fn witness_core<C: Config, CircuitDefaultType>(
    witness_solver: &WitnessSolver<C>,
    layered_circuit: &Circuit<C, NormalInputType>,
    hint_registry: &expander_compiler::frontend::HintRegistry<CircuitField<C>>,
    assignment: &CircuitDefaultType,
    witness_path: &str,
    compress: bool,
) -> Result<(), RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let witness =
        solve_and_validate_witness(witness_solver, layered_circuit, hint_registry, assignment)?;

    let file = std::fs::File::create(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let mut writer = compressed_writer(file, compress)?;
    witness
        .serialize_into(&mut writer)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;
    writer
        .finish()
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;

    Ok(())
}

/// Generates witnesses for multiple inputs sequentially.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the manifest, circuit, or witness solver fails.
pub fn run_batch_witness<C: Config, I, CircuitDefaultType>(
    io_reader_factory: impl Fn() -> I,
    manifest_path: &str,
    circuit_path: &str,
    compress: bool,
) -> Result<BatchResult, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let manifest: BatchManifest<WitnessJob> = load_manifest(manifest_path)?;
    let (layered_circuit, witness_solver) = load_circuit_and_solver::<C>(circuit_path)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    let job_count = manifest.jobs.len();
    println!("Loaded circuit and witness solver. Processing {job_count} jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        let mut io_reader = io_reader_factory();
        let assignment = CircuitDefaultType::default();
        let assignment = match io_reader.read_inputs(&job.input, assignment) {
            Ok(a) => a,
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
                continue;
            }
        };
        let assignment = match io_reader.read_outputs(&job.output, assignment) {
            Ok(a) => a,
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
                continue;
            }
        };

        match witness_core::<C, CircuitDefaultType>(
            &witness_solver,
            &layered_circuit,
            &hint_registry,
            &assignment,
            &job.witness,
            compress,
        ) {
            Ok(()) => {
                succeeded += 1;
                println!("[{}/{}] witness: {}", idx + 1, job_count, job.witness);
            }
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
            }
        }
    }

    Ok(BatchResult {
        succeeded,
        failed,
        errors,
    })
}

/// Generates witnesses for multiple inputs via stdin/stdout piping.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the circuit, reading stdin, or witness generation fails.
pub fn run_pipe_witness<C: Config, I, CircuitDefaultType>(
    io_reader_factory: impl Fn() -> I,
    circuit_path: &str,
    compress: bool,
) -> Result<BatchResult, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let (layered_circuit, witness_solver) = load_circuit_and_solver::<C>(circuit_path)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    let stdin = std::io::stdin();
    let manifest: BatchManifest<PipeWitnessJob> =
        rmp_serde::from_read(stdin.lock()).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let job_count = manifest.jobs.len();
    eprintln!("Loaded circuit and witness solver. Processing {job_count} piped jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        let mut io_reader = io_reader_factory();
        let assignment = CircuitDefaultType::default();
        let assignment = match io_reader.apply_values(job.input, job.output, assignment) {
            Ok(a) => a,
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
                continue;
            }
        };

        match solve_and_validate_witness::<C, CircuitDefaultType>(
            &witness_solver,
            &layered_circuit,
            &hint_registry,
            &assignment,
        ) {
            Ok(witness) => {
                let file = match std::fs::File::create(&job.witness) {
                    Ok(f) => f,
                    Err(e) => {
                        failed += 1;
                        let msg = format!("witness I/O: {e:?}");
                        eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, msg);
                        errors.push((idx, msg));
                        continue;
                    }
                };
                let mut writer = match compressed_writer(file, compress) {
                    Ok(w) => w,
                    Err(e) => {
                        failed += 1;
                        let msg = format!("witness compress: {e:?}");
                        eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, msg);
                        errors.push((idx, msg));
                        continue;
                    }
                };
                if let Err(e) = witness
                    .serialize_into(&mut writer)
                    .map_err(|e| RunError::Serialize(format!("{e:?}")))
                    .and_then(|()| {
                        writer
                            .finish()
                            .map_err(|e| RunError::Serialize(format!("{e:?}")))
                    })
                {
                    failed += 1;
                    let msg = format!("witness serialize: {e:?}");
                    eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, msg);
                    errors.push((idx, msg));
                    continue;
                }
                succeeded += 1;
                eprintln!("[{}/{}] witness: {}", idx + 1, job_count, job.witness);
            }
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
            }
        }
    }

    let result = BatchResult {
        succeeded,
        failed,
        errors,
    };
    write_msgpack_stdout(&result)?;

    Ok(result)
}

/// Generates proofs for multiple witnesses via stdin/stdout piping.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the circuit, reading stdin, or proving fails.
pub fn run_pipe_prove<C: Config, CircuitDefaultType>(
    circuit_path: &str,
    compress: bool,
) -> Result<BatchResult, RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;

    let stdin = std::io::stdin();
    let manifest: BatchManifest<ProveJob> =
        rmp_serde::from_read(stdin.lock()).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let job_count = manifest.jobs.len();
    eprintln!("Loaded circuit. Processing {job_count} piped prove jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        let mut circuit = layered_circuit.export_to_expander_flatten();
        match prove_core::<C>(&mut circuit, &job.witness, &job.proof, compress) {
            Ok(()) => {
                succeeded += 1;
                eprintln!("[{}/{}] proof: {}", idx + 1, job_count, job.proof);
            }
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
            }
        }
    }

    let result = BatchResult {
        succeeded,
        failed,
        errors,
    };
    write_msgpack_stdout(&result)?;

    Ok(result)
}

fn verify_single_job<C: Config, I, CircuitDefaultType>(
    io_reader_factory: &impl Fn() -> I,
    layered_circuit: &Circuit<C, NormalInputType>,
    job: PipeVerifyJob,
) -> Result<String, String>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let mut io_reader = io_reader_factory();
    let assignment = CircuitDefaultType::default();
    let assignment = io_reader
        .apply_values(job.input, job.output, assignment)
        .map_err(|e| e.to_string())?;

    let mut vars: Vec<_> = Vec::new();
    let mut public_vars: Vec<_> = Vec::new();
    assignment.dump_into(&mut vars, &mut public_vars);

    let mut circuit = layered_circuit.export_to_expander_flatten();

    let file = std::fs::File::open(&job.witness).map_err(|e| format!("witness I/O: {e:?}"))?;
    let reader = auto_reader(file).map_err(|e| format!("witness reader: {e:?}"))?;
    let witness = Witness::<C>::deserialize_from(reader)
        .map_err(|e| format!("witness deserialize: {e:?}"))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    circuit.layers[0].input_vals.clone_from(&simd_input);
    circuit.public_input.clone_from(&simd_public_input);

    let mismatch = public_vars
        .iter()
        .enumerate()
        .any(|(i, _)| format!("{:?}", public_vars[i]) != format!("{:?}", circuit.public_input[i]));
    if mismatch {
        return Err("inputs/outputs don't match the witness".into());
    }

    let file = std::fs::File::open(&job.proof).map_err(|e| format!("proof I/O: {e:?}"))?;
    let reader = auto_reader(file).map_err(|e| format!("proof reader: {e:?}"))?;
    let proof_and_claimed_v: Vec<u8> =
        Vec::deserialize_from(reader).map_err(|e| format!("proof deserialize: {e:?}"))?;

    let (proof, claimed_v) =
        executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v)
            .map_err(|e| format!("proof load: {e:?}"))?;

    let mpi_config = MPIConfig::verifier_new(1);
    if executor::verify::<C>(&mut circuit, mpi_config, &proof, &claimed_v) {
        Ok(format!("verified: {}", job.proof))
    } else {
        Err("Verification failed".into())
    }
}

/// Verifies multiple proofs via stdin/stdout piping.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the circuit, reading stdin, or verification fails.
pub fn run_pipe_verify<C: Config, I, CircuitDefaultType>(
    io_reader_factory: impl Fn() -> I,
    circuit_path: &str,
) -> Result<BatchResult, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;

    let stdin = std::io::stdin();
    let manifest: BatchManifest<PipeVerifyJob> =
        rmp_serde::from_read(stdin.lock()).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let job_count = manifest.jobs.len();
    eprintln!("Loaded circuit. Processing {job_count} piped verify jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        match verify_single_job::<C, I, CircuitDefaultType>(
            &io_reader_factory,
            &layered_circuit,
            job,
        ) {
            Ok(msg) => {
                succeeded += 1;
                eprintln!("[{}/{}] {msg}", idx + 1, job_count);
            }
            Err(msg) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {msg}", idx + 1, job_count);
                errors.push((idx, msg));
            }
        }
    }

    let result = BatchResult {
        succeeded,
        failed,
        errors,
    };
    write_msgpack_stdout(&result)?;

    Ok(result)
}

/// Generates proofs for multiple witnesses sequentially.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the manifest or circuit fails.
pub fn run_batch_prove<C: Config, CircuitDefaultType>(
    manifest_path: &str,
    circuit_path: &str,
    compress: bool,
) -> Result<BatchResult, RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let manifest: BatchManifest<ProveJob> = load_manifest(manifest_path)?;
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;

    let job_count = manifest.jobs.len();
    println!("Loaded circuit. Processing {job_count} prove jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        let mut circuit = layered_circuit.export_to_expander_flatten();
        match prove_core::<C>(&mut circuit, &job.witness, &job.proof, compress) {
            Ok(()) => {
                succeeded += 1;
                println!("[{}/{}] proof: {}", idx + 1, job_count, job.proof);
            }
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
            }
        }
    }

    Ok(BatchResult {
        succeeded,
        failed,
        errors,
    })
}

/// Verifies multiple proofs sequentially.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the manifest or circuit fails.
pub fn run_batch_verify<C: Config, I, CircuitDefaultType>(
    io_reader_factory: impl Fn() -> I,
    manifest_path: &str,
    circuit_path: &str,
) -> Result<BatchResult, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let manifest: BatchManifest<VerifyJob> = load_manifest(manifest_path)?;
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;

    let job_count = manifest.jobs.len();
    println!("Loaded circuit. Processing {job_count} verify jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        let mut circuit = layered_circuit.export_to_expander_flatten();
        let mut io_reader = io_reader_factory();
        match verify_core::<C, I, CircuitDefaultType>(
            &mut circuit,
            &mut io_reader,
            &job.input,
            &job.output,
            &job.witness,
            &job.proof,
        ) {
            Ok(()) => {
                succeeded += 1;
                println!("[{}/{}] verified: {}", idx + 1, job_count, job.proof);
            }
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] FAILED: {}", idx + 1, job_count, e);
                errors.push((idx, e.to_string()));
            }
        }
    }

    Ok(BatchResult {
        succeeded,
        failed,
        errors,
    })
}

/// # Errors
/// Returns `RunError` on decompression or deserialization failure.
pub fn load_circuit_from_bytes<C: Config>(
    data: &[u8],
) -> Result<Circuit<C, NormalInputType>, RunError> {
    let data = auto_decompress_bytes(data)?;
    Circuit::<C, NormalInputType>::deserialize_from(Cursor::new(&*data))
        .map_err(|e| RunError::Deserialize(format!("circuit: {e:?}")))
}

/// # Errors
/// Returns `RunError` on decompression or deserialization failure.
pub fn load_witness_solver_from_bytes<C: Config>(
    data: &[u8],
) -> Result<WitnessSolver<C>, RunError> {
    let data = auto_decompress_bytes(data)?;
    WitnessSolver::<C>::deserialize_from(Cursor::new(&*data))
        .map_err(|e| RunError::Deserialize(format!("witness_solver: {e:?}")))
}

/// # Errors
/// Returns `RunError` on decompression or deserialization failure.
pub fn load_witness_from_bytes<C: Config>(data: &[u8]) -> Result<Witness<C>, RunError> {
    let data = auto_decompress_bytes(data)?;
    Witness::<C>::deserialize_from(Cursor::new(&*data))
        .map_err(|e| RunError::Deserialize(format!("witness: {e:?}")))
}

/// # Errors
/// Returns `RunError` on serialization or compression failure.
pub fn serialize_witness<C: Config>(
    witness: &Witness<C>,
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    let mut buf = Vec::new();
    witness
        .serialize_into(&mut buf)
        .map_err(|e| RunError::Serialize(format!("witness: {e:?}")))?;
    maybe_compress_bytes(buf, compress)
}

/// # Errors
/// Returns `RunError` on proof generation failure.
pub fn prove_from_bytes<C: Config>(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    compress: bool,
) -> Result<Vec<u8>, RunError> {
    let layered_circuit = load_circuit_from_bytes::<C>(circuit_bytes)?;
    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    let witness = load_witness_from_bytes::<C>(witness_bytes)?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input.clone_from(&simd_public_input);
    expander_circuit.evaluate();

    let mpi_config = MPIConfig::prover_new();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config);
    let proof_bytes = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| RunError::Serialize(format!("proof: {e:?}")))?;

    maybe_compress_bytes(proof_bytes, compress)
}

/// # Errors
/// Returns `RunError` on verification failure.
pub fn verify_from_bytes<C: Config>(
    circuit_bytes: &[u8],
    witness_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, RunError> {
    let layered_circuit = load_circuit_from_bytes::<C>(circuit_bytes)?;
    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    let witness = load_witness_from_bytes::<C>(witness_bytes)?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input.clone_from(&simd_public_input);

    let proof_data = auto_decompress_bytes(proof_bytes)?;
    let (proof, claimed_v) = executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_data)
        .map_err(|e| RunError::Deserialize(format!("proof: {e:?}")))?;

    let mpi_config = MPIConfig::verifier_new(1);
    Ok(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v,
    ))
}

/// Compiles a circuit and writes it to a msgpack file.
///
/// # Errors
/// Returns `RunError` if compilation, serialization, or file I/O fails.
pub fn write_circuit_msgpack<C: Config, CircuitType>(
    path: &str,
    compress: bool,
    metadata: Option<CircuitParams>,
) -> Result<(), RunError>
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone + MaybeConfigure,
{
    let bundle = compile_to_bundle::<C, CircuitType>(metadata)?;
    write_circuit_bundle(path, &bundle, compress)
}

fn write_circuit_bundle(
    path: &str,
    bundle: &CompiledCircuit,
    compress: bool,
) -> Result<(), RunError> {
    jstprove_io::serialize_to_file(bundle, Path::new(path), compress).map_err(|e| {
        let _ = std::fs::remove_file(path);
        jstprove_io_to_run_error(e, path, true)
    })?;
    Ok(())
}

fn jstprove_io_to_run_error(e: jstprove_io::Error, path: &str, is_write: bool) -> RunError {
    match e.downcast::<std::io::Error>() {
        Ok(io_err) => RunError::Io {
            source: io_err,
            path: path.to_string(),
        },
        Err(e) if is_write => RunError::Serialize(format!("{e:#}")),
        Err(e) => RunError::Deserialize(format!("{e:#}")),
    }
}

/// Reads a compiled circuit bundle from a msgpack file.
///
/// # Errors
/// Returns `RunError` if file I/O or deserialization fails.
pub fn read_circuit_msgpack(path: &str) -> Result<CompiledCircuit, RunError> {
    jstprove_io::deserialize_from_file(Path::new(path))
        .map_err(|e| jstprove_io_to_run_error(e, path, false))
}

#[must_use]
pub fn try_load_metadata_from_circuit(circuit_path: &str) -> Option<CircuitParams> {
    let msgpack_path = Path::new(circuit_path).with_extension("msgpack");
    if !msgpack_path.exists() {
        return None;
    }
    read_circuit_msgpack(msgpack_path.to_str()?).ok()?.metadata
}

/// Reads a prove request from stdin and writes the proof to stdout via msgpack.
///
/// # Errors
/// Returns `RunError` if deserialization, proving, or serialization fails.
pub fn msgpack_prove_stdin<C: Config>(compress: bool) -> Result<(), RunError> {
    let stdin = std::io::stdin();
    let req: ProveRequest = rmp_serde::decode::from_read(stdin.lock())
        .map_err(|e| RunError::Deserialize(format!("msgpack stdin: {e:?}")))?;

    let proof = prove_from_bytes::<C>(&req.circuit, &req.witness, compress)?;

    let resp = ProofBundle {
        proof,
        version: Some(jstprove_artifact_version()),
    };
    write_msgpack_stdout(&resp)?;

    Ok(())
}

/// Reads a verify request from stdin and writes the result to stdout via msgpack.
///
/// # Errors
/// Returns `RunError` if deserialization, verification, or serialization fails.
pub fn msgpack_verify_stdin<C: Config>() -> Result<(), RunError> {
    let stdin = std::io::stdin();
    let req: VerifyRequest = rmp_serde::decode::from_read(stdin.lock())
        .map_err(|e| RunError::Deserialize(format!("msgpack stdin: {e:?}")))?;

    let valid = verify_from_bytes::<C>(&req.circuit, &req.witness, &req.proof)?;

    let resp = VerifyResponse { valid, error: None };
    write_msgpack_stdout(&resp)?;

    Ok(())
}

/// # Errors
/// Returns `RunError` on witness generation or serialization failure.
pub fn witness_from_request<C: Config, I, CircuitDefaultType>(
    req: &WitnessRequest,
    io_reader: &mut I,
    compress: bool,
) -> Result<WitnessBundle, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let layered_circuit = load_circuit_from_bytes::<C>(&req.circuit)?;
    let witness_solver = load_witness_solver_from_bytes::<C>(&req.witness_solver)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    let input_value: Value = rmp_serde::from_slice(&req.inputs)
        .map_err(|e| RunError::Deserialize(format!("input msgpack: {e:?}")))?;
    let output_value: Value = rmp_serde::from_slice(&req.outputs)
        .map_err(|e| RunError::Deserialize(format!("output msgpack: {e:?}")))?;

    let output_data = flatten_value_to_i64(&output_value);

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader
        .apply_values(input_value, output_value, assignment)
        .map_err(|e| RunError::Witness(format!("apply_values: {e:?}")))?;

    let witness = solve_and_validate_witness(
        &witness_solver,
        &layered_circuit,
        &hint_registry,
        &assignment,
    )?;

    let witness_bytes = serialize_witness::<C>(&witness, compress)?;

    Ok(WitnessBundle {
        witness: witness_bytes,
        output_data: if output_data.is_empty() {
            None
        } else {
            Some(output_data)
        },
        version: Some(jstprove_artifact_version()),
    })
}

fn split_trailing_number(s: &str) -> (&str, Option<u64>) {
    let pos = s.rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    if pos < s.len() {
        (&s[..pos], s[pos..].parse::<u64>().ok())
    } else {
        (s, None)
    }
}

fn natural_key_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (a_prefix, a_num) = split_trailing_number(a);
    let (b_prefix, b_num) = split_trailing_number(b);
    a_prefix
        .cmp(b_prefix)
        .then_with(|| a_num.cmp(&b_num))
        .then_with(|| a.cmp(b))
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn flatten_value_to_i64(val: &Value) -> Vec<i64> {
    match val {
        Value::Array(arr) => arr.iter().flat_map(flatten_value_to_i64).collect(),
        Value::Integer(n) => {
            if let Some(i) = n.as_i64() {
                vec![i]
            } else {
                vec![]
            }
        }
        Value::F64(f) => {
            if f.is_finite() && *f >= i64::MIN as f64 && *f < 9_223_372_036_854_775_808.0_f64 {
                vec![*f as i64]
            } else {
                vec![]
            }
        }
        Value::F32(f) => {
            let d = f64::from(*f);
            if f.is_finite() && d >= i64::MIN as f64 && d < 9_223_372_036_854_775_808.0_f64 {
                vec![d as i64]
            } else {
                vec![]
            }
        }
        Value::Map(entries) => {
            let mut pairs: Vec<_> = entries
                .iter()
                .filter_map(|(k, v)| k.as_str().map(|s| (s.to_string(), v)))
                .collect();
            pairs.sort_by(|(a, _), (b, _)| natural_key_cmp(a, b));
            pairs
                .into_iter()
                .flat_map(|(_, v)| flatten_value_to_i64(v))
                .collect()
        }
        _ => vec![],
    }
}

/// Reads a witness request from stdin and writes the witness to stdout via msgpack.
///
/// # Errors
/// Returns `RunError` if deserialization, witness generation, or serialization fails.
pub fn msgpack_witness_stdin<C: Config, I, CircuitDefaultType>(
    io_reader: &mut I,
    compress: bool,
) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let stdin = std::io::stdin();
    let req: WitnessRequest = rmp_serde::decode::from_read(stdin.lock())
        .map_err(|e| RunError::Deserialize(format!("msgpack stdin: {e:?}")))?;

    let resp = witness_from_request::<C, I, CircuitDefaultType>(&req, io_reader, compress)?;

    write_msgpack_stdout(&resp)?;

    Ok(())
}

/// Proves a circuit from msgpack files and writes the proof to a file.
///
/// # Errors
/// Returns `RunError` if file I/O, deserialization, proving, or serialization fails.
pub fn msgpack_prove_file<C: Config>(
    circuit_path: &str,
    witness_path: &str,
    proof_path: &str,
    compress: bool,
) -> Result<(), RunError> {
    let circuit_bundle = read_circuit_msgpack(circuit_path)?;
    let witness_bundle: WitnessBundle = {
        let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
            source: e,
            path: witness_path.into(),
        })?;
        rmp_serde::decode::from_read(std::io::BufReader::new(file))
            .map_err(|e| RunError::Deserialize(format!("witness msgpack: {e:?}")))?
    };

    let proof = prove_from_bytes::<C>(&circuit_bundle.circuit, &witness_bundle.witness, compress)?;

    let resp = ProofBundle {
        proof,
        version: Some(jstprove_artifact_version()),
    };
    let file = std::fs::File::create(proof_path).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;
    let mut writer = std::io::BufWriter::new(file);
    resp.serialize(&mut rmp_serde::Serializer::new(&mut writer).with_struct_map())
        .map_err(|e| RunError::Serialize(format!("proof msgpack: {e:?}")))?;
    std::io::Write::flush(&mut writer).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;

    Ok(())
}

/// Verifies a proof from msgpack files.
///
/// # Errors
/// Returns `RunError` if file I/O, deserialization, or verification fails.
pub fn msgpack_verify_file<C: Config>(
    circuit_path: &str,
    witness_path: &str,
    proof_path: &str,
) -> Result<bool, RunError> {
    let circuit_bundle = read_circuit_msgpack(circuit_path)?;
    let witness_bundle: WitnessBundle = {
        let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
            source: e,
            path: witness_path.into(),
        })?;
        rmp_serde::decode::from_read(std::io::BufReader::new(file))
            .map_err(|e| RunError::Deserialize(format!("witness msgpack: {e:?}")))?
    };
    let proof_bundle: ProofBundle = {
        let file = std::fs::File::open(proof_path).map_err(|e| RunError::Io {
            source: e,
            path: proof_path.into(),
        })?;
        rmp_serde::decode::from_read(std::io::BufReader::new(file))
            .map_err(|e| RunError::Deserialize(format!("proof msgpack: {e:?}")))?
    };

    verify_from_bytes::<C>(
        &circuit_bundle.circuit,
        &witness_bundle.witness,
        &proof_bundle.proof,
    )
}

/// A trait for circuits that can configure their internal inputs/outputs
/// before being used.
///
/// This is intended to abstract over circuits that may need to
/// initialize variable shapes or parameters (e.g., based on an
/// [`Architecture`] loaded at runtime).
///
/// # Errors
///
/// Implementations should return a [`RunError`] if configuration fails.
/// Typical failure cases may include:
/// - Mismatched dimensions between input/output definitions.
/// - Missing or invalid runtime parameters.
/// - Errors from underlying utility functions.
pub trait ConfigurableCircuit {
    /// Configure the circuit inputs and outputs.
    ///
    /// Implementations should resize or re-initialize inputs and outputs
    /// as needed (for example, flattening ONNX tensors into a flat vector
    /// of variables).
    ///
    /// # Errors
    ///
    /// Returns a [`RunError`] if configuration fails due to invalid
    /// architecture, parameter mismatch, or other issues.
    fn configure(&mut self) -> Result<(), RunError>;
}

pub trait MaybeConfigure {
    /// # Errors
    /// Returns `RunError` if configuration fails.
    fn maybe_configure(&mut self) -> Result<(), RunError> {
        Ok(())
    }
}

impl<T: ConfigurableCircuit> MaybeConfigure for T {
    fn maybe_configure(&mut self) -> Result<(), RunError> {
        self.configure()
    }
}

fn configure_if_possible<T: MaybeConfigure>(circuit: &mut T) -> Result<(), RunError> {
    circuit.maybe_configure()
}

/// Retrieves the value of a CLI argument as a `String`.
///
/// # Arguments
///
/// * `matches` - Parsed [`clap::ArgMatches`] object, usually from [`clap::Command::get_matches`].
/// * `name` - The name of the argument to look up (must match how it was defined in your CLI).
///
/// # Returns
///
/// * `Ok(String)` - The value of the argument, if present.
///
/// # Errors
///
/// Returns [`CliError::MissingArgument`] if the argument was not provided on the command line.
///
pub fn get_arg(matches: &clap::ArgMatches, name: &'static str) -> Result<String, CliError> {
    matches
        .get_one::<String>(name)
        .map(ToString::to_string)
        .ok_or(CliError::MissingArgument(name))
}

fn is_remainder_backend(matches: &clap::ArgMatches, metadata: Option<&CircuitParams>) -> bool {
    if matches.value_source("backend") == Some(clap::parser::ValueSource::CommandLine) {
        return matches
            .get_one::<String>("backend")
            .is_some_and(|b| b == "remainder");
    }
    metadata.is_some_and(|m| m.backend.is_remainder())
}

#[cfg(feature = "remainder")]
fn get_model_or_circuit(matches: &clap::ArgMatches) -> Result<String, CliError> {
    get_arg(matches, "model").or_else(|_| get_arg(matches, "circuit_path"))
}

#[cfg(feature = "remainder")]
fn dispatch_remainder(
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
fn dispatch_remainder(
    _matches: &clap::ArgMatches,
    _command: &str,
    _compress: bool,
) -> Result<(), CliError> {
    Err(RunError::Unsupported("remainder backend requires the 'remainder' feature".into()).into())
}

/// Handles CLI arguments and dispatches to the appropriate runner command.
///
/// # Errors
/// Returns `CliError` on missing arguments, unknown commands, or runner failures.
#[allow(clippy::too_many_lines)]
pub fn handle_args<
    C: Config,
    CircuitType,
    CircuitDefaultType,
    Filereader: IOReader<CircuitDefaultType, C> + Send + Sync + Clone,
>(
    matches: &clap::ArgMatches,
    file_reader: &mut Filereader,
    metadata: Option<CircuitParams>,
) -> Result<(), CliError>
where
    CircuitDefaultType: std::default::Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + std::clone::Clone
        + Send
        + Sync,
    CircuitType: std::default::Default
        + expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
        + std::clone::Clone
        + Define<C>
        + MaybeConfigure,
{
    let command = get_arg(matches, "type")?;
    let compress = !matches.get_flag("no_compress");
    let use_remainder = is_remainder_backend(matches, metadata.as_ref());

    if use_remainder {
        return dispatch_remainder(matches, &command, compress);
    }

    match command.as_str() {
        "run_compile_circuit" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_compile_and_serialize::<C, CircuitType>(&circuit_path, compress, metadata)?;
        }
        "run_gen_witness" => {
            let input_path = get_arg(matches, "input")?;
            let output_path = get_arg(matches, "output")?;
            let witness_path = get_arg(matches, "witness")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_witness::<C, _, CircuitDefaultType>(
                file_reader,
                &input_path,
                &output_path,
                &witness_path,
                &circuit_path,
                compress,
            )?;
        }
        "run_debug_witness" => {
            let input_path = get_arg(matches, "input")?;
            let output_path = get_arg(matches, "output")?;
            let witness_path = get_arg(matches, "witness")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            debug_witness::<C, _, CircuitDefaultType, CircuitType>(
                file_reader,
                &input_path,
                &output_path,
                &witness_path,
                &circuit_path,
            )?;
        }
        "run_prove_witness" => {
            let witness_path = get_arg(matches, "witness")?;
            let proof_path = get_arg(matches, "proof")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_prove_witness::<C, CircuitDefaultType>(
                &circuit_path,
                &witness_path,
                &proof_path,
                compress,
            )?;
        }
        "run_gen_verify" => {
            let input_path = get_arg(matches, "input")?;
            let output_path = get_arg(matches, "output")?;
            let witness_path = get_arg(matches, "witness")?;
            let proof_path = get_arg(matches, "proof")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_verify_io::<C, Filereader, CircuitDefaultType>(
                &circuit_path,
                file_reader,
                &input_path,
                &output_path,
                &witness_path,
                &proof_path,
            )?;
        }
        "run_batch_witness" => {
            let manifest_path = get_arg(matches, "manifest")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            let reader_template = file_reader.clone();
            let result = run_batch_witness::<C, _, CircuitDefaultType>(
                move || reader_template.clone(),
                &manifest_path,
                &circuit_path,
                compress,
            )?;
            check_batch_result("witness", &result)?;
        }
        "run_batch_prove" => {
            let manifest_path = get_arg(matches, "manifest")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            let result =
                run_batch_prove::<C, CircuitDefaultType>(&manifest_path, &circuit_path, compress)?;
            check_batch_result("prove", &result)?;
        }
        "run_batch_verify" => {
            let manifest_path = get_arg(matches, "manifest")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            let reader_template = file_reader.clone();
            let result = run_batch_verify::<C, _, CircuitDefaultType>(
                move || reader_template.clone(),
                &manifest_path,
                &circuit_path,
            )?;
            check_batch_result("verify", &result)?;
        }
        "run_pipe_witness" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            let reader_template = file_reader.clone();
            run_pipe_witness::<C, _, CircuitDefaultType>(
                move || reader_template.clone(),
                &circuit_path,
                compress,
            )?;
        }
        "run_pipe_prove" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_pipe_prove::<C, CircuitDefaultType>(&circuit_path, compress)?;
        }
        "run_pipe_verify" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            let reader_template = file_reader.clone();
            run_pipe_verify::<C, _, CircuitDefaultType>(
                move || reader_template.clone(),
                &circuit_path,
            )?;
        }
        "msgpack_compile" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            write_circuit_msgpack::<C, CircuitType>(&circuit_path, compress, metadata)?;
            eprintln!("Compiled to {circuit_path}");
        }
        "msgpack_prove" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            let witness_path = get_arg(matches, "witness")?;
            let proof_path = get_arg(matches, "proof")?;
            msgpack_prove_file::<C>(&circuit_path, &witness_path, &proof_path, compress)?;
            eprintln!("Proved to {proof_path}");
        }
        "msgpack_verify" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            let witness_path = get_arg(matches, "witness")?;
            let proof_path = get_arg(matches, "proof")?;
            let valid = msgpack_verify_file::<C>(&circuit_path, &witness_path, &proof_path)?;
            if valid {
                eprintln!("Verified");
            } else {
                return Err(RunError::Verify("verification failed".into()).into());
            }
        }
        "msgpack_prove_stdin" => {
            msgpack_prove_stdin::<C>(compress)?;
        }
        "msgpack_verify_stdin" => {
            msgpack_verify_stdin::<C>()?;
        }
        "msgpack_witness_stdin" => {
            msgpack_witness_stdin::<C, _, CircuitDefaultType>(file_reader, compress)?;
        }
        _ => return Err(CliError::UnknownCommand(command.to_string())),
    }
    Ok(())
}

#[must_use]
#[allow(clippy::too_many_lines)]
pub fn get_args() -> clap::ArgMatches {
    let matches: clap::ArgMatches = Command::new("File Copier")
        .version("1.0")
        .about("Copies content from input file to output file")
        .arg(
            Arg::new("type")
                .help("The type of main runner we want to run")
                .required(true) // This argument is required
                .index(1), // Positional argument (first argument)
        )
        .arg(
            Arg::new("input")
                .help("The file to read circuit inputs from")
                .required(false) // This argument is required
                .long("input") // Use a long flag (e.g., --name)
                .short('i'), // Use a short flag (e.g., -n)
                             // .index(2), // Positional argument (first argument)
        )
        .arg(
            Arg::new("output")
                .help("The file to read outputs to the circuit")
                .required(false) // This argument is also required
                .long("output") // Use a long flag (e.g., --name)
                .short('o'), // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("witness")
                .help("The witness file path")
                .required(false) // This argument is also required
                .long("witness") // Use a long flag (e.g., --name)
                .short('w'), // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("proof")
                .help("The proof file path")
                .required(false) // This argument is also required
                .long("proof") // Use a long flag (e.g., --name)
                .short('p'), // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("circuit_path")
                .help("The circuit file path")
                .required(false) // This argument is also required
                .long("circuit") // Use a long flag (e.g., --name)
                .short('c'), // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("name")
                .help("The name of the circuit for the file names to serialize/deserialize")
                .required(false) // This argument is also required
                .long("name") // Use a long flag (e.g., --name)
                .short('n'), // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("meta")
                .help("Path to the ONNX circuit params and metadata (msgpack or JSON)")
                .required(false)
                .long("meta")
                .short('m'),
        )
        .arg(
            Arg::new("arch")
                .help("Path to the ONNX generic circuit architecture (msgpack)")
                .required(false)
                .long("arch")
                .short('a'),
        )
        .arg(
            Arg::new("wandb")
                .help("Path to the ONNX generic circuit W&B (msgpack)")
                .required(false)
                .long("wandb")
                .short('b'),
        )
        .arg(
            Arg::new("manifest")
                .help("Path to batch manifest (msgpack)")
                .required(false)
                .long("manifest")
                .short('f'),
        )
        .arg(
            Arg::new("no_compress")
                .help("Disable zstd compression for output files")
                .required(false)
                .long("no-compress")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("backend")
                .help("Proving backend: 'expander' (default) or 'remainder'")
                .required(false)
                .long("backend")
                .value_parser(["expander", "remainder"])
                .default_value("expander"),
        )
        .arg(
            Arg::new("model")
                .help("Path to quantized ONNX model (Remainder backend)")
                .required(false)
                .long("model"),
        )
        .get_matches();
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use jstprove_io::{ENVELOPE_MAGIC, ZSTD_COMPRESSION_LEVEL, ZSTD_MAGIC};

    struct TempFile(PathBuf);

    impl TempFile {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(name))
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn maybe_compress_roundtrip() {
        let original = b"hello world, this is a test payload for zstd compression".to_vec();
        let compressed = maybe_compress_bytes(original.clone(), true).unwrap();
        assert_eq!(&compressed[..4], &ZSTD_MAGIC);
        assert_ne!(compressed, original);
        let decompressed = auto_decompress_bytes(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), &original[..]);
    }

    #[test]
    fn auto_decompress_passthrough() {
        let data = b"not zstd data".to_vec();
        let result = auto_decompress_bytes(&data).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.as_ref(), &data[..]);
    }

    #[test]
    fn write_read_bundle_uncompressed() {
        let tmp = TempFile::new("jstprove_test_uncompressed.msgpack");
        let bundle = CompiledCircuit {
            circuit: vec![1, 2, 3, 4],
            witness_solver: vec![5, 6, 7, 8],
            metadata: None,
            version: None,
        };
        write_circuit_bundle(tmp.path(), &bundle, false).unwrap();
        let loaded = read_circuit_msgpack(tmp.path()).unwrap();
        assert_eq!(loaded.circuit, bundle.circuit);
        assert_eq!(loaded.witness_solver, bundle.witness_solver);
        assert!(loaded.metadata.is_none());
    }

    #[test]
    fn write_read_bundle_compressed() {
        let tmp = TempFile::new("jstprove_test_compressed.msgpack");
        let bundle = CompiledCircuit {
            circuit: vec![10; 1024],
            witness_solver: vec![20; 512],
            metadata: None,
            version: None,
        };
        write_circuit_bundle(tmp.path(), &bundle, true).unwrap();
        let raw = std::fs::read(tmp.path()).unwrap();
        assert_eq!(&raw[..4], &ENVELOPE_MAGIC);
        let loaded = read_circuit_msgpack(tmp.path()).unwrap();
        assert_eq!(loaded.circuit, bundle.circuit);
        assert_eq!(loaded.witness_solver, bundle.witness_solver);
    }

    #[test]
    fn write_read_bundle_with_metadata() {
        let tmp = TempFile::new("jstprove_test_metadata.msgpack");
        let meta: CircuitParams = serde_json::from_str(
            r#"{
                "scale_base": 2,
                "scale_exponent": 18,
                "rescale_config": {"conv1": true},
                "inputs": [{"name": "x", "elem_type": 1, "shape": [1, 3]}],
                "outputs": [{"name": "y", "elem_type": 1, "shape": [1, 10]}]
            }"#,
        )
        .unwrap();
        let bundle = CompiledCircuit {
            circuit: vec![0xAA; 64],
            witness_solver: vec![0xBB; 64],
            metadata: Some(meta),
            version: None,
        };
        write_circuit_bundle(tmp.path(), &bundle, true).unwrap();
        let loaded = read_circuit_msgpack(tmp.path()).unwrap();
        assert_eq!(loaded.circuit, bundle.circuit);
        assert_eq!(loaded.witness_solver, bundle.witness_solver);
        let m = loaded.metadata.unwrap();
        assert_eq!(m.scale_base, 2);
        assert_eq!(m.scale_exponent, 18);
        assert!(m.rescale_config.contains_key("conv1"));
    }

    #[test]
    fn witness_solver_path_with_extension() {
        let p = get_witness_solver_path("/tmp/circuit.bin");
        assert_eq!(p, PathBuf::from("/tmp/circuit_witness_solver.bin"));
    }

    #[test]
    fn witness_solver_path_without_extension() {
        let p = get_witness_solver_path("/tmp/circuit");
        assert_eq!(p, PathBuf::from("/tmp/circuit_witness_solver"));
    }

    #[test]
    fn auto_reader_plain_data() {
        let tmp = TempFile::new("jstprove_test_plain_reader.bin");
        let data = b"raw binary content here";
        std::fs::write(tmp.path(), data).unwrap();
        let file = std::fs::File::open(tmp.path()).unwrap();
        let mut reader = auto_reader(file).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf).unwrap();
        assert_eq!(buf, data);
    }

    #[test]
    fn auto_reader_zstd_data() {
        let tmp = TempFile::new("jstprove_test_zstd_reader.bin");
        let original = b"some data to compress for auto_reader test";
        let compressed =
            zstd::encode_all(Cursor::new(original.as_slice()), ZSTD_COMPRESSION_LEVEL).unwrap();
        std::fs::write(tmp.path(), &compressed).unwrap();
        let file = std::fs::File::open(tmp.path()).unwrap();
        let mut reader = auto_reader(file).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf).unwrap();
        assert_eq!(buf, original);
    }

    #[test]
    fn natural_key_cmp_sorts_numeric_suffixes() {
        let mut keys = vec![
            "output_10",
            "output_2",
            "output_0",
            "output_1",
            "other",
            "output_20",
        ];
        keys.sort_by(|a, b| natural_key_cmp(a, b));
        assert_eq!(
            keys,
            vec![
                "other",
                "output_0",
                "output_1",
                "output_2",
                "output_10",
                "output_20"
            ]
        );
    }

    #[test]
    fn natural_key_cmp_tie_breaker_for_distinct_numerically_equal_keys() {
        let mut keys = vec!["output_1", "output_01"];
        keys.sort_by(|a, b| natural_key_cmp(a, b));
        assert_eq!(keys, vec!["output_01", "output_1"]);
    }
}
