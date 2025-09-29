use clap::{Arg, Command};
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::{Circuit, NormalInputType};
use io_reader::IOReader;
// use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
// use expander_compiler::utils::serde::Serde;
use expander_compiler::frontend::{
    ChallengeField, CompileOptions, Config, Define, EmptyHintCaller, Variable, WitnessSolver,
    compile, extra::debug_eval, internal::DumpLoadTwoVariables,
};
use gkr_engine::{FieldEngine, GKREngine, MPIConfig};
use peakmem_alloc::{INSTRUMENTED_SYSTEM, PeakMemAlloc, PeakMemAllocTrait};
use serdes::ExpSerde;
// use serde_json::{from_reader, to_writer};
use std::alloc::System;
use std::path::{Path, PathBuf};
// use std::io::{Read, Write};
// use std::time::Instant;

use crate::io::io_reader;
use crate::runner::errors::{CliError, RunError};
use expander_binary::executor;

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
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    let witness_file = get_witness_solver_path(circuit_path);
    let file = std::fs::File::open(&witness_file).map_err(|e| RunError::Io {
        source: e,
        path: witness_file.display().to_string(),
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
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    // let start = Instant::now();

    let assignments = vec![assignment; 1];
    let witness = witness_solver
        .solve_witnesses(&assignments)
        .map_err(|e| RunError::Witness(format!("{e:?}")))?;
    // #### Sanity check, can be removed in prod ####
    let output = layered_circuit.run(&witness);
    // unwrap
    for x in &output {
        if !(*x) {
            return Err(RunError::Witness("Witness generation failed sanity check. Outputs generated do not match outputs supplied.".into()));
        }
    }
    // #### Until here #######

    println!("Witness Generated");

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    // let mem_mb = GLOBAL.get_peak_memory().saturating_div(1024 * 1024);
    // println!("Peak Memory used Overall : {mem_mb:.2}");
    // let duration = start.elapsed();
    // println!(
    //     "Time elapsed: {}.{} seconds",
    //     duration.as_secs(),
    //     duration.subsec_millis(),
    // );

    let file = std::fs::File::create(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let writer = std::io::BufWriter::new(file);
    witness
        .serialize_into(writer)
        .map_err(|e| RunError::Serialize(format!("{e:?}")))?;
    // layered_circuit.evaluate();
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
    let assignments = vec![assignment.clone(); 1];

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);
    debug_eval(&circuit, &assignment, EmptyHintCaller);

    let witness = witness_solver
        .solve_witnesses(&assignments)
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
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    // let start = Instant::now();

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io {
        source: e,
        path: circuit_path.into(),
    })?;
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    let mpi_config = MPIConfig::prover_new(None, None);

    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let reader = std::io::BufReader::new(file);
    let witness: Witness<C> =
        Witness::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input.clone_from(&simd_public_input);

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    let proof: Vec<u8> = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| RunError::Serialize(format!("Error serializing proof {e:?}")))?;

    let file = std::fs::File::create(proof_path).map_err(|e| RunError::Io {
        source: e,
        path: proof_path.into(),
    })?;
    let writer = std::io::BufWriter::new(file);
    proof
        .serialize_into(writer)
        .map_err(|e| RunError::Serialize(format!("Error serializing proof to file {e:?}")))?;

    println!("Proved");
    // let mem_mb = GLOBAL.get_peak_memory().saturating_div(1024 * 1024);
    // println!("Peak Memory used Overall : {mem_mb:.2}");
    // let duration = start.elapsed();
    // println!(
    //     "Time elapsed: {}.{} seconds",
    //     duration.as_secs(),
    //     duration.subsec_millis()
    // );
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
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    // let start = Instant::now();

    // let mpi_config = MPIConfig::prover_new(None, None);
    let mpi_config = gkr_engine::MPIConfig::verifier_new(1);

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io {
        source: e,
        path: circuit_path.into(),
    })?;

    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io {
        source: e,
        path: witness_path.into(),
    })?;
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader)
        .map_err(|e| RunError::Deserialize(format!("{e:?}")))?;
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd();

    expander_circuit.layers[0]
        .input_vals
        .clone_from(&simd_input);
    expander_circuit.public_input.clone_from(&simd_public_input);
    let assignment = CircuitDefaultType::default();

    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    // let mut vars: Vec<<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField> = Vec::new();
    let mut vars: Vec<_> = Vec::new();

    let mut public_vars: Vec<_> = Vec::new();
    assignment.dump_into(&mut vars, &mut public_vars);
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
    if !executor::verify::<C>(&mut expander_circuit, mpi_config, &proof, &claimed_v) {
        return Err(RunError::Verify("Verification failed".into()));
    }

    println!("Verified");
    // let mem_mb = GLOBAL.get_peak_memory().saturating_div(1024 * 1024);
    // println!("Peak Memory used Overall : {mem_mb:.2}");
    // let duration = start.elapsed();
    // println!(
    //     "Time elapsed: {}.{} seconds",
    //     duration.as_secs(),
    //     duration.subsec_millis()
    // );
    Ok(())
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
pub fn handle_args<
    C: Config,
    CircuitType,
    CircuitDefaultType,
    Filereader: IOReader<CircuitDefaultType, C>,
>(
    matches: &clap::ArgMatches,
    file_reader: &mut Filereader,
) -> Result<(), CliError>
where
    CircuitDefaultType: std::default::Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + std::clone::Clone,
    CircuitType: std::default::Default
        + expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
        + std::clone::Clone
        + Define<C>
        + MaybeConfigure, //+ RunBehavior<C>,
{
    // The first argument is the command we need to identify
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
            // debug_witness::<C, _, CircuitDefaultType, CircuitType>(file_reader, input_path, output_path, &witness_path, circuit_path);
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

            // run_verify::<BN254Config, Filereader, CircuitDefaultType>(&circuit_name);
            run_verify_io::<C, Filereader, CircuitDefaultType>(
                &circuit_path,
                file_reader,
                &input_path,
                &output_path,
                &witness_path,
                &proof_path,
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
        .get_matches();
    matches
}
