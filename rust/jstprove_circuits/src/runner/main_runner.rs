use io_reader::IOReader;
use clap::{Arg, Command};
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::{Circuit, NormalInputType};
// use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
// use expander_compiler::utils::serde::Serde;
use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
use gkr_engine::{MPIConfig, GKREngine, FieldEngine};
use peakmem_alloc::*;
use serdes::ExpSerde;
// use serde_json::{from_reader, to_writer};
use std::alloc::System;
use std::path::{Path, PathBuf};
// use std::io::{Read, Write};
use std::time::Instant;

use crate::io::io_reader;
use crate::runner::errors::{CliError, RunError};
use expander_binary::executor;

// use crate::io::io_reader;


#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

fn get_witness_solver_path(input: &str) -> PathBuf {
    let path = Path::new(input);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(input);

    let file_name = if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        format!("{}_witness_solver.{}", stem, ext)
    } else {
        format!("{}_witness_solver", stem)
    };

    PathBuf::from(file_name)
}

pub fn run_compile_and_serialize<C: Config, CircuitType>(circuit_path: &str) -> Result<(), RunError>
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();
    

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);
    

    let compile_result =
        compile(&circuit, CompileOptions::default()).map_err(|e| RunError::Compile(format!("{:?}", e)))?;
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();

    let file = std::fs::File::create(circuit_path).map_err(|e| RunError::Io { source: e, path: circuit_path.into() })?;

    let writer = std::io::BufWriter::new(file);
    compile_result.layered_circuit.serialize_into(writer).map_err(|e| RunError::Serialize(format!("{:?}", e)))?;

    let witness_path = get_witness_solver_path(circuit_path);
    let file = std::fs::File::create(&witness_path).map_err(|e| RunError::Io { source: e, path: witness_path.display().to_string() })?;
    let writer = std::io::BufWriter::new(file);
    compile_result.witness_solver.serialize_into(writer).map_err(|e| RunError::Serialize(format!("{:?}", e)))?;


    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
    Ok(())
}

pub fn run_witness<C: Config, I, CircuitDefaultType>(io_reader: &mut I, input_path: &str, output_path:&str, witness_path: &str, circuit_path: &str) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType,C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
    
{
    let witness_file = get_witness_solver_path(circuit_path);
    let file = std::fs::File::open(&witness_file).map_err(|e| RunError::Io { source: e, path: witness_file.display().to_string() })?;
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io { source: e, path: circuit_path.into() })?;
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;

    let assignment = CircuitDefaultType::default();

    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).map_err(|e| RunError::Witness(format!("{:?}", e)))?;
    // #### Sanity check, can be removed in prod ####
    let output = layered_circuit.run(&witness);
    // unwrap
    for x in output.iter() {
        assert_eq!(*x, true);
    }
    // #### Until here #######

    println!("Witness Generated");

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis(),
    );

    let file = std::fs::File::create(witness_path).map_err(|e| RunError::Io { source: e, path: witness_path.into() })?;
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).map_err(|e| RunError::Serialize(format!("{:?}", e)))?;
    // layered_circuit.evaluate();
    Ok(())
}


pub fn debug_witness<C: Config, I, CircuitDefaultType, CircuitType>(io_reader: &mut I, input_path: &str, output_path:&str, _witness_path: &str, circuit_path: &str) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType,C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
        CircuitType: Default
        + DumpLoadTwoVariables<Variable>
        + expander_compiler::frontend::Define<C>
        + Clone
{
    let witness_path = get_witness_solver_path(circuit_path);
    let file = std::fs::File::open(&witness_path).map_err(|e| RunError::Io { source: e, path: witness_path.display().to_string() })?;
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;
    

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io { source: e, path: circuit_path.into() })?;
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;


    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;
    let assignments = vec![assignment.clone(); 1];


    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);
    debug_eval(&circuit, &assignment, EmptyHintCaller);


    let witness = witness_solver.solve_witnesses(&assignments).map_err(|e| RunError::Witness(format!("{:?}", e)))?;
    let output = layered_circuit.run(&witness);
    for x in output.iter() {
        assert_eq!(*x, true);
    }
    Ok(())

    
}

pub fn run_prove_witness<C: Config, CircuitDefaultType>(circuit_path: &str, witness_path: &str, proof_path: &str) -> Result<(), RunError>
where
    // I: IOReader<CircuitDefaultType,C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io { source: e, path: circuit_path.into() })?;
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    let mpi_config = MPIConfig::prover_new(None, None);

    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io { source: e, path: witness_path.into() })?;
    let reader = std::io::BufReader::new(file);
    let witness: Witness<C> = Witness::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    let proof: Vec<u8> = executor::dump_proof_and_claimed_v(&proof, &claimed_v).map_err(|e| RunError::Serialize(format!("{:?}", e)))?;

    let file = std::fs::File::create(proof_path).map_err(|e| RunError::Io { source: e, path: proof_path.into() })?;
    let writer = std::io::BufWriter::new(file);
    proof.serialize_into(writer).map_err(|e| RunError::Serialize(format!("{:?}", e)))?;

    println!("Proved");
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
    Ok(())
}

pub fn run_verify_io<C: Config, I, CircuitDefaultType>(circuit_path: &str, io_reader: &mut I, input_path: &str, output_path:&str, witness_path: &str, proof_path: &str) -> Result<(), RunError>
where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone + 
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    // let mpi_config = MPIConfig::prover_new(None, None);
    let mpi_config = gkr_engine::MPIConfig::verifier_new(1);

    let file = std::fs::File::open(circuit_path).map_err(|e| RunError::Io { source: e, path: circuit_path.into() })?;

    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();


    let file = std::fs::File::open(witness_path).map_err(|e| RunError::Io { source: e, path: witness_path.into() })?;
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd();

    expander_circuit.layers[0].input_vals = simd_input.clone();
    expander_circuit.public_input = simd_public_input.clone();
    let assignment = CircuitDefaultType::default();
    
    let assignment = io_reader.read_inputs(input_path, assignment)?;
    let assignment = io_reader.read_outputs(output_path, assignment)?;

    // let mut vars: Vec<<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField> = Vec::new();
    let mut vars: Vec<_> = Vec::new();


    let mut public_vars: Vec<_> = Vec::new();
    assignment.dump_into(&mut vars, &mut public_vars);
    for (i, _) in public_vars.iter().enumerate(){
        let x = format!("{:?}", public_vars[i]);
        let y = format!("{:?}",expander_circuit.public_input[i]);

        assert_eq!(x,y);
    }

    let file = std::fs::File::open(proof_path).map_err(|e| RunError::Io { source: e, path: proof_path.into() })?;
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> = Vec::deserialize_from(reader).map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;


    let (proof, claimed_v) =
        executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v)
            .map_err(|e| RunError::Deserialize(format!("{:?}", e)))?;
    // verify
    // assert!(executor::verify::<C>(
    //     &mut expander_circuit,
    //     mpi_config,
    //     &proof,
    //     &claimed_v
    // ));
    if !executor::verify::<C>(&mut expander_circuit, mpi_config, &proof, &claimed_v) {
        return Err(RunError::Verify("Verification failed".into()));
    }

    println!("Verified");
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
    Ok(())
}

pub trait ConfigurableCircuit {
    fn configure(&mut self);
}


//This solution requires specialization. If specialization is broken on any given release, we can replace configure_if_possible, to take in a bool. 
// The bool can be passed from bin main into handle args and into the relevant functions that call configure_if_possible for a manual implementation. 
// For ease of use, I prefer the current solution, however if it proves to cause problems, we can replace it
trait MaybeConfigure {
    fn maybe_configure(&mut self);
}

// Default impl: do nothing
impl<T> MaybeConfigure for T {
    default fn maybe_configure(&mut self) {
        // Not configurable
    }
}

// Special impl: if also ConfigurableCircuit, call configure()
impl<T> MaybeConfigure for T
where
    T: ConfigurableCircuit,
{
    fn maybe_configure(&mut self) {
        self.configure();
    }
}

fn configure_if_possible<T: MaybeConfigure>(circuit: &mut T) {
    circuit.maybe_configure();
}

fn get_arg(matches: &clap::ArgMatches, name: &'static str) -> Result<String, CliError> {
    matches
        .get_one::<String>(name)
        .map(|s| s.to_string())
        .ok_or(CliError::MissingArgument(name))
}

pub fn handle_args<C: Config,CircuitType, CircuitDefaultType, Filereader: IOReader<CircuitDefaultType, C>>(file_reader: &mut  Filereader) -> Result<(), CliError>
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
    + std::clone::Clone + Define<C> //+ RunBehavior<C>,
    {

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
                .short('i')  // Use a short flag (e.g., -n)
                // .index(2), // Positional argument (first argument)
        )
        .arg(
            Arg::new("output")
                .help("The file to read outputs to the circuit")
                .required(false) // This argument is also required
                .long("output") // Use a long flag (e.g., --name)
                .short('o')  // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("witness")
                .help("The witness file path")
                .required(false) // This argument is also required
                .long("witness") // Use a long flag (e.g., --name)
                .short('w')  // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("proof")
                .help("The proof file path")
                .required(false) // This argument is also required
                .long("proof") // Use a long flag (e.g., --name)
                .short('p')  // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("circuit_path")
                .help("The circuit file path")
                .required(false) // This argument is also required
                .long("circuit") // Use a long flag (e.g., --name)
                .short('c')  // Use a short flag (e.g., -n)
        )
        .arg(
            Arg::new("name")
                .help("The name of the circuit for the file names to serialize/deserialize")
                .required(false) // This argument is also required
                .long("name") // Use a long flag (e.g., --name)
                .short('n')  // Use a short flag (e.g., -n)
        )
        .get_matches();

    // The first argument is the command we need to identify
    // let command = &args[1];
    let command = get_arg(&matches, "type")?;
    

    match command.as_str() {                                    
        "run_compile_circuit" => {
            let circuit_path = get_arg(&matches, "circuit_path")?;
            run_compile_and_serialize::<C,CircuitType>(&circuit_path)?;
        }
        "run_gen_witness" => {

            let input_path = get_arg(&matches, "input")?;
            let output_path = get_arg(&matches, "output")?;
            let witness_path = get_arg(&matches, "witness")?;
            let circuit_path = get_arg(&matches, "circuit_path")?;
            run_witness::<C, _, CircuitDefaultType>(file_reader, &input_path, &output_path, &witness_path, &circuit_path)?;
            // debug_witness::<C, _, CircuitDefaultType, CircuitType>(file_reader, input_path, output_path, &witness_path, circuit_path);
        }
        "run_debug_witness" => {

            let input_path = get_arg(&matches, "input")?;
            let output_path = get_arg(&matches, "output")?;
            let witness_path = get_arg(&matches, "witness")?;
            let circuit_path = get_arg(&matches, "circuit_path")?;
            debug_witness::<C, _, CircuitDefaultType, CircuitType>(file_reader, &input_path, &output_path, &witness_path, &circuit_path)?;
        }
        "run_prove_witness" => {
            let witness_path = get_arg(&matches, "witness")?;
        let proof_path = get_arg(&matches, "proof")?;
        let circuit_path = get_arg(&matches, "circuit_path")?;

            run_prove_witness::<C, CircuitDefaultType>( &circuit_path, &witness_path, &proof_path)?;
        }
        "run_gen_verify"=> {
            let input_path = get_arg(&matches, "input")?;
            let output_path = get_arg(&matches, "output")?;
            let witness_path = get_arg(&matches, "witness")?;
            let proof_path = get_arg(&matches, "proof")?;
            let circuit_path = get_arg(&matches, "circuit_path")?;


            // run_verify::<BN254Config, Filereader, CircuitDefaultType>(&circuit_name);
            run_verify_io::<C, Filereader, CircuitDefaultType>(&circuit_path, file_reader, &input_path, &output_path, &witness_path, &proof_path)?;
        }
        _ => return Err(CliError::UnknownCommand(command.to_string())),
    };
    Ok(())
}
