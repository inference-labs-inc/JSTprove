use clap::{Arg, Command};
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::{Circuit, NormalInputType};
use io_reader::IOReader;
// use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
// use expander_compiler::utils::serde::Serde;
use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
use gkr_engine::{FieldEngine, GKREngine, MPIConfig};
use peakmem_alloc::*;
use serdes::ExpSerde;
// use serde_json::{from_reader, to_writer};
use std::alloc::System;
use std::path::Path;
// use std::io::{Read, Write};
use std::time::Instant;

use crate::io::io_reader;
use expander_binary::executor;

// use crate::io::io_reader;

#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

pub fn run_main<C: Config, I, CircuitType, CircuitDefaultType>(
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
) where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();
    println!("Compiling Circuit...");

    // let compile_result: CompileResult<C> = compile(&CircuitType::default()).unwrap();
    let compile_result = compile(&CircuitType::default(), CompileOptions::default()).unwrap();
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
    println!("Generating witness...");

    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;

    let assignment = CircuitDefaultType::default();

    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    let output = layered_circuit.run(&witness);

    for x in output.iter() {
        assert_eq!(*x, true);
    }
    println!("Witness Generated");
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
    println!("Generating proof...");
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    // TODO explore other configs
    let mpi_config = MPIConfig::prover_new(None, None);

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    println!("Proven");
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
    println!("Verifying proof...");
    // println!("");
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    // // verify
    assert!(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v
    ));

    println!("Verified");

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    )
}

fn get_witness_solver_path(input: &str) -> String {
    let path = Path::new(input);
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            return format!("{}_witness_solver.{}", stem, extension);
        } else {
            return format!("{}_witness_solver", stem);
        }
    }
    input.to_string()
}

pub fn run_compile_and_serialize<C: Config, CircuitType>(circuit_path: &str)
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    // let compile_result: CompileResult<C> = compile(&CircuitType::default()).unwrap();

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);

    // circuit.output = vec![vec![Variable::default(); 64 * 8]; N_HASHES];
    // circuit.out = vec![vec![Variable::default(); 32 * 8]; N_HASHES];

    let compile_result = compile(&circuit, CompileOptions::default()).unwrap();
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();

    // let file = std::fs::File::create(format!("{}_circuit.txt", name)).unwrap();
    let file = std::fs::File::create(circuit_path).unwrap();

    let writer = std::io::BufWriter::new(file);
    compile_result
        .layered_circuit
        .serialize_into(writer)
        .unwrap();

    let file = std::fs::File::create(get_witness_solver_path(circuit_path)).unwrap();
    // let file = std::fs::File::create(format!("{}_witness_solver.txt", name)).unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result
        .witness_solver
        .serialize_into(writer)
        .unwrap();

    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
}

pub fn run_compile_and_serialize_configurable<C: Config, CircuitType>(circuit_path: &str)
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone + ConfigurableCircuit,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    // let compile_result: CompileResult<C> = compile(&CircuitType::default()).unwrap();
    let mut circuit = CircuitType::default();
    // circuit.output = vec![vec![Variable::default(); 64 * 8]; N_HASHES];
    //

    circuit.configure();

    let compile_result = compile(&circuit, CompileOptions::default()).unwrap();
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();

    // let file = std::fs::File::create(format!("{}_circuit.txt", name)).unwrap();
    let file = std::fs::File::create(circuit_path).unwrap();

    let writer = std::io::BufWriter::new(file);
    compile_result
        .layered_circuit
        .serialize_into(writer)
        .unwrap();

    let file = std::fs::File::create(get_witness_solver_path(circuit_path)).unwrap();
    // let file = std::fs::File::create(format!("{}_witness_solver.txt", name)).unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result
        .witness_solver
        .serialize_into(writer)
        .unwrap();

    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
}

pub fn run_rest<C: Config, I, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();
    let matches = Command::new("File Copier")
        .version("1.0")
        .about("Copies content from input file to output file")
        .arg(
            Arg::new("input")
                .help("The input file to read from")
                .required(true) // This argument is required
                .index(1), // Positional argument (first argument)
        )
        .arg(
            Arg::new("output")
                .help("The output file to write to")
                .required(true) // This argument is also required
                .index(2), // Positional argument (second argument)
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
    let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"

    // let CompileResult {
    //     witness_solver,
    //     layered_circuit,
    // } = compile_result;

    let file = std::fs::File::open(format!("{}_witness_solver.txt", io_reader.get_path())).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).unwrap();

    let file = std::fs::File::open(format!("{}_circuit.txt", io_reader.get_path())).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    // layered_circuit.evaluate();
    let output = layered_circuit.run(&witness);

    for x in output.iter() {
        assert_eq!(*x, true);
    }

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let mpi_config = MPIConfig::prover_new(None, None);

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    let proof = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| e.to_string())
        .unwrap();

    // let proof = executor::dump_proof_and_claimed_v(&proof, &claimed_v).map_err(|e| e.to_string()).unwrap();

    let (proof, claimed_v) = match executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof) {
        Ok((proof, claimed_v)) => (proof, claimed_v),
        Err(_) => {
            return;
        }
    };

    // verify
    assert!(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v
    ));

    println!("Verified");

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    )
}

pub fn run_witness<C: Config, I, CircuitDefaultType>(
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    witness_path: &str,
    circuit_path: &str,
) where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    // GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    // let start = Instant::now();
    // println!("{:?}", format!("{}_witness_solver.txt", io_reader.get_path()));
    // let file = std::fs::File::open(format!("{}_witness_solver.txt", io_reader.get_path())).unwrap();
    println!("{}", get_witness_solver_path(circuit_path));
    let file = std::fs::File::open(get_witness_solver_path(circuit_path)).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).unwrap();

    let file = std::fs::File::open(circuit_path).unwrap();
    // let file = std::fs::File::open(format!("{}_circuit.txt", io_reader.get_path())).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let assignment = CircuitDefaultType::default();

    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    // This can be removed
    let output = layered_circuit.run(&witness);

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
        "Time elapsed: {}.{} seconds {}",
        duration.as_secs(),
        duration.subsec_millis(),
        duration.as_nanos()
    );

    let file = std::fs::File::create(witness_path).unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();
    // layered_circuit.evaluate();
}

pub fn debug_witness<C: Config, I, CircuitDefaultType, CircuitType>(
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    _witness_path: &str,
    circuit_path: &str,
) where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
    CircuitType:
        Default + DumpLoadTwoVariables<Variable> + expander_compiler::frontend::Define<C> + Clone,
{
    let file = std::fs::File::open(get_witness_solver_path(circuit_path)).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).unwrap();

    let file = std::fs::File::open(circuit_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let assignment = CircuitDefaultType::default();
    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);
    let assignments = vec![assignment.clone(); 1];

    let mut circuit = CircuitType::default();
    configure_if_possible::<CircuitType>(&mut circuit);
    debug_eval(&circuit, &assignment, EmptyHintCaller);

    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    let output = layered_circuit.run(&witness);
    for x in output.iter() {
        assert_eq!(*x, true);
    }
}

pub fn run_prove_witness<C: Config, CircuitDefaultType>(
    circuit_path: &str,
    witness_path: &str,
    proof_path: &str,
) where
    // I: IOReader<CircuitDefaultType,C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let file = std::fs::File::open(circuit_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let mpi_config = MPIConfig::prover_new(None, None);
    // let universe = mpi::initialize().unwrap();
    // let world = universe.world();
    // let mpi_config = MPIConfig::prover_new(Some(&universe), Some(&world));

    let file = std::fs::File::open(witness_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness: Witness<C> = Witness::deserialize_from(reader).unwrap();

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    let proof: Vec<u8> = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| e.to_string())
        .unwrap();

    let file = std::fs::File::create(proof_path).unwrap();
    let writer = std::io::BufWriter::new(file);
    proof.serialize_into(writer).unwrap();
    // to_writer(writer, &proof).unwrap();

    println!("Proved");
    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    )
}

pub fn run_witness_and_proof<C: Config, I, CircuitDefaultType>(
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    _circuit_name: &str,
) where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();
    let file = std::fs::File::open(format!("{}_witness_solver.txt", io_reader.get_path())).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).unwrap();

    let file = std::fs::File::open(format!("{}_circuit.txt", io_reader.get_path())).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let assignment = CircuitDefaultType::default();

    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();

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
        duration.subsec_millis()
    );
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let file = std::fs::File::create(format!("{}_witness.txt", io_reader.get_path())).unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();
    // layered_circuit.evaluate();
    let output = layered_circuit.run(&witness);

    for x in output.iter() {
        assert_eq!(*x, true);
    }

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let mpi_config = MPIConfig::prover_new(None, None);

    let (simd_input, simd_public_input) = witness.to_simd();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut expander_circuit, mpi_config.clone());

    let proof: Vec<u8> = executor::dump_proof_and_claimed_v(&proof, &claimed_v)
        .map_err(|e| e.to_string())
        .unwrap();

    let file = std::fs::File::create(format!("{}_proof.bin", io_reader.get_path())).unwrap();
    let writer = std::io::BufWriter::new(file);
    // println!("{:?}", proof);
    // writer.write_all(&proof).unwrap();
    // proof.serialize_into(writer).unwrap();
    proof.serialize_into(writer).unwrap();
    // to_writer(writer, &proof).unwrap();

    println!("Proved");
    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    )
}

pub fn run_verify<C: Config, I, CircuitDefaultType>(name: &str)
where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    let mpi_config = MPIConfig::prover_new(None, None);

    let file = std::fs::File::open(format!("{}_circuit.txt", name)).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let file = std::fs::File::open(format!("{}_witness.txt", name)).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader).unwrap();
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd();

    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    let file = std::fs::File::open(format!("{}_proof.bin", name)).unwrap();
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> = Vec::deserialize_from(reader).unwrap();

    let (proof, claimed_v) =
        match executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v) {
            Ok((proof, claimed_v)) => (proof, claimed_v),
            Err(_) => {
                return;
            }
        };
    // verify
    assert!(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v
    ));

    println!("Verified");

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    )
}

pub fn run_verify_io<C: Config, I, CircuitDefaultType>(
    circuit_path: &str,
    io_reader: &mut I,
    input_path: &str,
    output_path: &str,
    witness_path: &str,
    proof_path: &str,
) where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    // let mpi_config = MPIConfig::prover_new(None, None);
    let mpi_config = gkr_engine::MPIConfig::verifier_new(1);

    // let file = std::fs::File::open(format!("{}_circuit.txt", name)).unwrap();
    let file = std::fs::File::open(circuit_path).unwrap();

    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let file = std::fs::File::open(witness_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader).unwrap();
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd();

    expander_circuit.layers[0].input_vals = simd_input.clone();
    expander_circuit.public_input = simd_public_input.clone();
    let assignment = CircuitDefaultType::default();

    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    // let mut vars: Vec<<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField> = Vec::new();
    let mut vars: Vec<_> = Vec::new();

    let mut public_vars: Vec<_> = Vec::new();
    assignment.dump_into(&mut vars, &mut public_vars);
    for (i, _) in public_vars.iter().enumerate() {
        let x = format!("{:?}", public_vars[i]);
        let y = format!("{:?}", expander_circuit.public_input[i]);

        // println!("{}", x);
        // println!("{}", y);
        //TODO This can potentially be improved

        assert_eq!(x, y);
    }
    println!("{}", proof_path);
    println!("{}", witness_path);

    let file = std::fs::File::open(proof_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> = Vec::deserialize_from(reader).unwrap();

    let (proof, claimed_v) =
        match executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v) {
            Ok((proof, claimed_v)) => (proof, claimed_v),
            Err(_) => {
                return;
            }
        };
    // verify
    assert!(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v
    ));

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
    )
}

pub fn run_verify_no_circuit<C: Config, I, CircuitDefaultType>(name: &str)
where
    I: IOReader<CircuitDefaultType, C>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();

    // TODO explore other configs
    // let mpi_config = MPIConfig::prover_new(None, None);
    let mpi_config = gkr_engine::MPIConfig::verifier_new(1);

    let file = std::fs::File::open(format!("{}_circuit.txt", name)).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let mut expander_circuit = layered_circuit.export_to_expander_flatten();

    let file = std::fs::File::open(format!("{}_witness.txt", name)).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader).unwrap();
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    let file = std::fs::File::open(format!("{}_proof.bin", name)).unwrap();
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> = Vec::deserialize_from(reader).unwrap();
    // let proof_and_claimed_v: Vec<u8> = from_reader(reader).unwrap();

    let (proof, claimed_v) =
        match executor::load_proof_and_claimed_v::<ChallengeField<C>>(&proof_and_claimed_v) {
            Ok((proof, claimed_v)) => (proof, claimed_v),
            Err(_) => {
                return;
            }
        };

    // verify
    assert!(executor::verify::<C>(
        &mut expander_circuit,
        mpi_config,
        &proof,
        &claimed_v
    ));

    println!("Verified");

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    )
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

pub fn handle_args<
    C: Config,
    CircuitType,
    CircuitDefaultType,
    Filereader: IOReader<CircuitDefaultType, C>,
>(
    file_reader: &mut Filereader,
) where
    CircuitDefaultType: std::default::Default
        + DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        + std::clone::Clone,

    CircuitType: std::default::Default
        + expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
        // + std::clone::Clone + GenericDefine<expander_compiler::frontend::BN254Config>,
        // + expander_compiler::frontend::Define<expander_compiler::frontend::BN254Config>
        + std::clone::Clone
        + Define<C>, //+ RunBehavior<C>,
                     // WrappedCircuitType: RunBehavior<C>
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
        .get_matches();

    // let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
    // let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"

    // The first argument is the command we need to identify
    // let command = &args[1];
    let command = matches.get_one::<String>("type").unwrap();

    match command.as_str() {
        //ignore
        "run_proof" => {
            let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
            run_main::<C, _, CircuitType, CircuitDefaultType>(
                file_reader,
                &input_path,
                &output_path,
            );
        }

        "run_compile_circuit" => {
            // let circuit_name = &args[2];
            // let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
            let circuit_path = matches.get_one::<String>("circuit_path").unwrap(); //"outputs/reward_output.json"

            run_compile_and_serialize::<C, CircuitType>(&circuit_path);
        }
        "run_gen_witness" => {
            let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
                                                                            // let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
            let witness_path = matches.get_one::<String>("witness").unwrap(); //"outputs/reward_output.json"
            let circuit_path = matches.get_one::<String>("circuit_path").unwrap(); //"outputs/reward_output.json"
            run_witness::<C, _, CircuitDefaultType>(
                file_reader,
                input_path,
                output_path,
                &witness_path,
                circuit_path,
            );
            // debug_witness::<C, _, CircuitDefaultType, CircuitType>(file_reader, input_path, output_path, &witness_path, circuit_path);

            // debug_bn254::<BN254Config, _, CircuitType>(file_reader);
        }
        "run_debug_witness" => {
            let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
                                                                            // let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
            let witness_path = matches.get_one::<String>("witness").unwrap(); //"outputs/reward_output.json"
            let circuit_path = matches.get_one::<String>("circuit_path").unwrap(); //"outputs/reward_output.json"
            debug_witness::<C, _, CircuitDefaultType, CircuitType>(
                file_reader,
                input_path,
                output_path,
                &witness_path,
                circuit_path,
            );
        }
        "run_prove_witness" => {
            // let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            // let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
            // let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
            let witness_path = matches.get_one::<String>("witness").unwrap(); //"outputs/reward_output.json"
            let proof_path = matches.get_one::<String>("proof").unwrap(); //"outputs/reward_output.json"
            let circuit_path = matches.get_one::<String>("circuit_path").unwrap(); //"outputs/reward_output.json"

            run_prove_witness::<C, CircuitDefaultType>(circuit_path, witness_path, proof_path);
        }
        "run_gen_verify" => {
            let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
                                                                            // let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
            let witness_path = matches.get_one::<String>("witness").unwrap(); //"outputs/reward_output.json"
            let proof_path = matches.get_one::<String>("proof").unwrap(); //"outputs/reward_output.json"
            let circuit_path = matches.get_one::<String>("circuit_path").unwrap(); //"outputs/reward_output.json"

            // run_verify::<BN254Config, Filereader, CircuitDefaultType>(&circuit_name);
            run_verify_io::<C, Filereader, CircuitDefaultType>(
                &circuit_path,
                file_reader,
                &input_path,
                &output_path,
                witness_path,
                proof_path,
            );
        }
        _ => {
            panic!("Unknown command or missing arguments.");
            // exit(1);
        }
    }
}
