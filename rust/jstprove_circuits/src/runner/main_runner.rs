use clap::{Arg, Command};
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::{Circuit, NormalInputType};
use expander_compiler::frontend::{
    ChallengeField, CircuitField, CompileOptions, Config, Define, Variable, WitnessSolver, compile,
    extra::debug_eval, internal::DumpLoadTwoVariables,
};
use gkr_engine::{FieldEngine, GKREngine, MPIConfig};
use io_reader::IOReader;
use peakmem_alloc::{INSTRUMENTED_SYSTEM, PeakMemAlloc, PeakMemAllocTrait};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serdes::ExpSerde;
use std::alloc::System;
use std::path::{Path, PathBuf};

use crate::io::io_reader;
use crate::runner::errors::{CliError, RunError};
use expander_binary::executor;

use crate::circuit_functions::hints::build_logup_hint_registry;

// use crate::io::io_reader;

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

/// Compiles a circuit and serializes both the compiled circuit and witness solver.
///
/// This function instantiates a `CircuitType`, configures it, and
/// compiles it into a layered circuit representation along with its associated
/// witness solver using Expander's circuit compiler.
/// Both artifacts are then serialized:
///
/// - The compiled circuit is written to `circuit_path`.
/// - The witness solver is written to a companion file determined by
///   [`get_witness_solver_path`].
///
/// # Arguments
///
/// - `circuit_path` – Path where the compiled circuit will be written.
///   A companion file at the same location (with a modified extension)
///   will hold the serialized witness solver.
///
/// # Errors
///
/// Returns a [`RunError`] if:
///
/// - The circuit fails to compile (`RunError::Compile`)
/// - The circuit or witness solver cannot be serialized (`RunError::Serialize`)
/// - The circuit or witness solver files cannot be created (`RunError::Io`)
///
pub fn run_compile_and_serialize<C: Config, CircuitType>(circuit_path: &str) -> Result<(), RunError>
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone + MaybeConfigure,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    // let start = Instant::now();

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);

    let compile_result = compile(&circuit, CompileOptions::default())
        .map_err(|e| RunError::Compile(format!("{e:?}")))?;
    // let mem_mb = GLOBAL.get_peak_memory().saturating_div(1024 * 1024);
    // println!("Peak Memory used Overall : {mem_mb:.2}");
    // let duration = start.elapsed();

    let file = std::fs::File::create(circuit_path).map_err(|e| RunError::Io {
        source: e,
        path: circuit_path.into(),
    })?;

    let writer = std::io::BufWriter::new(file);
    compile_result
        .layered_circuit
        .serialize_into(writer)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;

    let witness_path = get_witness_solver_path(circuit_path);
    let file = std::fs::File::create(&witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.display().to_string(),
    })?;
    let writer = std::io::BufWriter::new(file);
    compile_result
        .witness_solver
        .serialize_into(writer)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;

    // println!(
    //     "Time elapsed: {}.{} seconds",
    //     duration.as_secs(),
    //     duration.subsec_millis()
    // );
    Ok(())
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
) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let witness_solver = load_witness_solver::<C>(circuit_path)?;
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    GLOBAL.reset_peak_memory();
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    witness_core::<C, CircuitDefaultType>(
        &witness_solver,
        &layered_circuit,
        &hint_registry,
        &assignment,
        witness_path,
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
    let witness_path = get_witness_solver_path(circuit_path);
    let file = std::fs::File::open(&witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.display().to_string(),
    })?;
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io {
        source: e,
        path: circuit_path.into(),
    })?;
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);

    // Build LogUp registry once
    let logup_hints = build_logup_hint_registry::<CircuitField<C>>();

    // Use it for the frontend debug evaluation
    debug_eval(&circuit, &assignment, logup_hints.clone());

    // And for the witness solver
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

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
) -> Result<(), RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory();
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;
    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    prove_core::<C>(&mut expander_circuit, witness_path, proof_path)?;
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
/// - `input_path` – Path to the JSON input file.
/// - `output_path` – Path to the JSON output file.
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
    let content = std::fs::read_to_string(path).map_err(|e| RunError::Io {
        source: e,
        path: path.into(),
    })?;
    serde_json::from_str(&content).map_err(|e| RunError::Json(format!("{e:?}")))
}

fn load_layered_circuit<C: Config>(path: &str) -> Result<Circuit<C, NormalInputType>, RunError> {
    let file = std::fs::File::open(path).map_err(|e| RunError::Io {
        source: e,
        path: path.into(),
    })?;
    let reader = std::io::BufReader::new(file);
    Circuit::<C, NormalInputType>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))
}

fn load_witness_solver<C: Config>(circuit_path: &str) -> Result<WitnessSolver<C>, RunError> {
    let witness_file = get_witness_solver_path(circuit_path);
    let file = std::fs::File::open(&witness_file).map_err(|e| RunError::Io {
        source: e,
        path: witness_file.display().to_string(),
    })?;
    let reader = std::io::BufReader::new(file);
    WitnessSolver::<C>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))
}

fn prove_core<C: Config>(
    expander_circuit: &mut expander_circuit::Circuit<C::FieldConfig>,
    witness_path: &str,
    proof_path: &str,
) -> Result<(), RunError> {
    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let reader = std::io::BufReader::new(file);
    let witness: Witness<C> =
        Witness::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input.clone_from(&simd_public_input);

    expander_circuit.evaluate();
    let mpi_config = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<C>(expander_circuit, mpi_config);

    let proof_bytes: Vec<u8> = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| RunError::Serialize(format!("Error serializing proof {e:?}")))?;

    let file = std::fs::File::create(proof_path).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;
    let writer = std::io::BufWriter::new(file);
    proof_bytes
        .serialize_into(writer)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;

    Ok(())
}

fn verify_core<C: Config, I, CircuitDefaultType>(
    expander_circuit: &mut expander_circuit::Circuit<C::FieldConfig>,
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
    let reader = std::io::BufReader::new(file);
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
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> =
        Vec::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let (proof, claimed_v) =
        executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v)
            .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let mpi_config = gkr_engine::MPIConfig::verifier_new(1);
    if !executor::verify::<C>(expander_circuit, mpi_config, &proof, &claimed_v) {
        return Err(RunError::Verify("Verification failed".into()));
    }

    Ok(())
}

fn solve_and_validate_witness<C: Config, CircuitDefaultType>(
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
    let writer = std::io::BufWriter::new(file);
    witness
        .serialize_into(writer)
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
) -> Result<BatchResult, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let manifest: BatchManifest<WitnessJob> = load_manifest(manifest_path)?;
    let witness_solver = load_witness_solver::<C>(circuit_path)?;
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;
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
) -> Result<BatchResult, RunError>
where
    I: IOReader<CircuitDefaultType, C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let witness_solver = load_witness_solver::<C>(circuit_path)?;
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();

    let stdin = std::io::stdin();
    let manifest: BatchManifest<PipeWitnessJob> =
        serde_json::from_reader(stdin.lock()).map_err(|e| RunError::Json(format!("{e:?}")))?;

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
                let writer = std::io::BufWriter::new(file);
                if let Err(e) = witness.serialize_into(writer) {
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
    let stdout = std::io::stdout();
    serde_json::to_writer(stdout.lock(), &result)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;
    println!();

    Ok(result)
}

/// Generates proofs for multiple witnesses via stdin/stdout piping.
///
/// # Errors
///
/// Returns a [`RunError`] if loading the circuit, reading stdin, or proving fails.
pub fn run_pipe_prove<C: Config, CircuitDefaultType>(
    circuit_path: &str,
) -> Result<BatchResult, RunError>
where
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let layered_circuit = load_layered_circuit::<C>(circuit_path)?;

    let stdin = std::io::stdin();
    let manifest: BatchManifest<ProveJob> =
        serde_json::from_reader(stdin.lock()).map_err(|e| RunError::Json(format!("{e:?}")))?;

    let job_count = manifest.jobs.len();
    eprintln!("Loaded circuit. Processing {job_count} piped prove jobs.");

    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut errors: Vec<(usize, String)> = Vec::new();

    for (idx, job) in manifest.jobs.into_iter().enumerate() {
        let mut circuit = layered_circuit.export_to_expander_flatten();
        match prove_core::<C>(&mut circuit, &job.witness, &job.proof) {
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
    let stdout = std::io::stdout();
    serde_json::to_writer(stdout.lock(), &result)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;
    println!();

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
    let witness = Witness::<C>::deserialize_from(std::io::BufReader::new(file))
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
    let proof_and_claimed_v: Vec<u8> = Vec::deserialize_from(std::io::BufReader::new(file))
        .map_err(|e| format!("proof deserialize: {e:?}"))?;

    let (proof, claimed_v) =
        executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v)
            .map_err(|e| format!("proof load: {e:?}"))?;

    let mpi_config = gkr_engine::MPIConfig::verifier_new(1);
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
        serde_json::from_reader(stdin.lock()).map_err(|e| RunError::Json(format!("{e:?}")))?;

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
    let stdout = std::io::stdout();
    serde_json::to_writer(stdout.lock(), &result)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;
    println!();

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
        match prove_core::<C>(&mut circuit, &job.witness, &job.proof) {
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
    fn maybe_configure(&mut self) {}
}

impl<T: ConfigurableCircuit> MaybeConfigure for T {
    fn maybe_configure(&mut self) {
        let _ = self.configure();
    }
}

// Usage
fn configure_if_possible<T: MaybeConfigure>(circuit: &mut T) {
    circuit.maybe_configure();
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

/// Handles CLI arguments and dispatches to the appropriate runner command.
///
/// This function parses command-line arguments (using [`clap`]) and executes
/// one of the supported commands:
///
/// - `run_compile_circuit` – creates and compiles a circuit and serializes it
/// - `run_gen_witness` – generates a witness from inputs and outputs
/// - `run_debug_witness` – runs witness generation in debug mode
/// - `run_prove_witness` – produces a proof from a witness
/// - `run_gen_verify` – verifies the proof
///
/// # Arguments
///
/// - `file_reader` - A mutable reference to the file reader used for circuit input/output.
///
/// # Errors
///
/// Returns a [`CliError`] if:
///
/// - A required argument is missing
/// - An unknown command is provided
/// - Any of the underlying runner functions (`run_compile_and_serialize`,
///   `run_witness`, `debug_witness`, `run_prove_witness`, `run_verify_io`)
///   fails during execution
///
/// # Example
///
/// ```ignore
/// use mycrate::{handle_args, MyConfig, MyCircuit, MyDefaultCircuit, MyFileReader};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut file_reader = FileReader {
///        path: "my_path",
///     };
///     handle_args::<MyConfig, MyCircuit, MyDefaultCircuit, _>(&mut reader)?;
///     Ok(())
/// }
/// ```
#[allow(clippy::too_many_lines)]
pub fn handle_args<
    C: Config,
    CircuitType,
    CircuitDefaultType,
    Filereader: IOReader<CircuitDefaultType, C> + Send + Sync + Clone,
>(
    matches: &clap::ArgMatches,
    file_reader: &mut Filereader,
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

    match command.as_str() {
        "run_compile_circuit" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_compile_and_serialize::<C, CircuitType>(&circuit_path)?;
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
            run_prove_witness::<C, CircuitDefaultType>(&circuit_path, &witness_path, &proof_path)?;
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
            )?;
            check_batch_result("witness", &result)?;
        }
        "run_batch_prove" => {
            let manifest_path = get_arg(matches, "manifest")?;
            let circuit_path = get_arg(matches, "circuit_path")?;
            let result = run_batch_prove::<C, CircuitDefaultType>(&manifest_path, &circuit_path)?;
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
            )?;
        }
        "run_pipe_prove" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            run_pipe_prove::<C, CircuitDefaultType>(&circuit_path)?;
        }
        "run_pipe_verify" => {
            let circuit_path = get_arg(matches, "circuit_path")?;
            let reader_template = file_reader.clone();
            run_pipe_verify::<C, _, CircuitDefaultType>(
                move || reader_template.clone(),
                &circuit_path,
            )?;
        }
        _ => return Err(CliError::UnknownCommand(command.to_string())),
    }
    Ok(())
}

#[must_use]
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
                .help("Path to the ONNX generic circuit params and metadata JSON")
                .required(false)
                .long("meta")
                .short('m'),
        )
        .arg(
            Arg::new("arch")
                .help("Path to the ONNX generic circuit architecture JSON")
                .required(false)
                .long("arch")
                .short('a'),
        )
        .arg(
            Arg::new("wandb")
                .help("Path to the ONNX generic circuit W&B JSON")
                .required(false)
                .long("wandb")
                .short('b'),
        )
        .arg(
            Arg::new("manifest")
                .help("Path to batch manifest JSON file")
                .required(false)
                .long("manifest")
                .short('f'),
        )
        .get_matches();
    matches
}
