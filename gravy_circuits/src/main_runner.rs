use crate::io_reader::{FileReader, IOReader};
use clap::{Arg, Command};
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::{Circuit, CrossLayerInputType, NormalInputType};
use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
use expander_compiler::utils::serde::Serde;
use gkr_field_config::GKRFieldConfig;
use peakmem_alloc::*;
use serde_json::{from_reader, to_writer};
use std::alloc::System;
use std::default;
use std::time::Instant;

#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

fn run_main<C: Config, I, CircuitType, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitType: Default + DumpLoadTwoVariables<Variable> + GenericDefine<C> + Clone,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
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

    // let compile_result: CompileResult<C> = compile(&CircuitType::default()).unwrap();
    let compile_result =
        compile_generic(&CircuitType::default(), CompileOptions::default()).unwrap();
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
    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;

    let assignment = CircuitDefaultType::default();
    let _input_reader = FileReader {
        path: input_path.clone(),
    };
    let _output_reader = FileReader {
        path: output_path.clone(),
    };
    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    let output = layered_circuit.run(&witness);

    for x in output.iter() {
        assert_eq!(*x, true);
    }

    

    let mut expander_circuit = layered_circuit
        .export_to_expander::<<C>::DefaultGKRFieldConfig>()
        .flatten();
    let config = expander_config::Config::<<C>::DefaultGKRConfig>::new(
        expander_config::GKRScheme::Vanilla,
        mpi_config::MPIConfig::new(),
    );

    let (simd_input, simd_public_input) = witness.to_simd::<<C>::DefaultSimdField>();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = gkr::executor::prove(&mut expander_circuit, &config);

    // verify
    assert!(gkr::executor::verify(
        &mut expander_circuit,
        &config,
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

fn run_compile_and_serialize<C: Config,CircuitType>()
where
    CircuitType: Default + DumpLoadTwoVariables<Variable> + GenericDefine<C> + Clone,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now();
    
    // let compile_result: CompileResult<C> = compile(&CircuitType::default()).unwrap();
    let compile_result =
        compile_generic(&CircuitType::default(), CompileOptions::default().with_mul_fanout_limit(1024)).unwrap();
    println!(
        "Peak Memory used Overall : {:.2}",
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();

    let file = std::fs::File::create(format!("trivial_circuit.txt")).unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result.layered_circuit.serialize_into(writer).unwrap();

    let file = std::fs::File::create(format!("trivial_witness_solver.txt")).unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result.witness_solver.serialize_into(writer).unwrap();


    println!(
        "Time elapsed: {}.{} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
}

fn run_rest<C: Config, I, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
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

    let file = std::fs::File::open(format!("trivial_witness_solver.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).unwrap();


    let file = std::fs::File::open(format!("trivial_circuit.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let assignment = CircuitDefaultType::default();
    let _input_reader = FileReader {
        path: input_path.clone(),
    };
    let _output_reader = FileReader {
        path: output_path.clone(),
    };
    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    // layered_circuit.evaluate();
    let output = layered_circuit.run(&witness);

    for x in output.iter() {
        assert_eq!(*x, true);
    }

    let mut expander_circuit = layered_circuit
        .export_to_expander::<<C>::DefaultGKRFieldConfig>()
        .flatten();
    let config = expander_config::Config::<<C>::DefaultGKRConfig>::new(
        expander_config::GKRScheme::Vanilla,
        mpi_config::MPIConfig::new(),
    );

    let (simd_input, simd_public_input) = witness.to_simd::<<C>::DefaultSimdField>();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = gkr::executor::prove(&mut expander_circuit, &config);


    let proof = gkr::executor::dump_proof_and_claimed_v(&proof, &claimed_v).map_err(|e| e.to_string()).unwrap();


    // let proof = gkr::executor::dump_proof_and_claimed_v(&proof, &claimed_v).map_err(|e| e.to_string()).unwrap();

    let (proof, claimed_v) = match gkr::executor::load_proof_and_claimed_v::<<<C as Config>::DefaultGKRFieldConfig as GKRFieldConfig>::ChallengeField>(&proof) {
        Ok((proof, claimed_v)) => (proof, claimed_v),
        Err(_) => {
            return;
        }
    };

    // verify
    assert!(gkr::executor::verify(
        &mut expander_circuit,
        &config,
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

fn run_witness_and_proof<C: Config, I, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
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

    let file = std::fs::File::open(format!("trivial_witness_solver.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness_solver = WitnessSolver::<C>::deserialize_from(reader).unwrap();


    let file = std::fs::File::open(format!("trivial_circuit.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let assignment = CircuitDefaultType::default();
    let _input_reader = FileReader {
        path: input_path.clone(),
    };
    let _output_reader = FileReader {
        path: output_path.clone(),
    };
    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    let assignments = vec![assignment; 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();

    let file = std::fs::File::create(format!("trivial_witness.txt")).unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();
    // layered_circuit.evaluate();
    let output = layered_circuit.run(&witness);

    for x in output.iter() {
        assert_eq!(*x, true);
    }

    let mut expander_circuit = layered_circuit
        .export_to_expander::<<C>::DefaultGKRFieldConfig>()
        .flatten();
    let config = expander_config::Config::<<C>::DefaultGKRConfig>::new(
        expander_config::GKRScheme::Vanilla,
        mpi_config::MPIConfig::new(),
    );

    let (simd_input, simd_public_input) = witness.to_simd::<<C>::DefaultSimdField>();
    println!("{} {}", simd_input.len(), simd_public_input.len());
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let (claimed_v, proof) = gkr::executor::prove(&mut expander_circuit, &config);

    let proof: Vec<u8> = gkr::executor::dump_proof_and_claimed_v(&proof, &claimed_v).map_err(|e| e.to_string()).unwrap();

    let file = std::fs::File::create(format!("trivial_proof.txt")).unwrap();
    let writer = std::io::BufWriter::new(file);
    // proof.serialize_into(writer).unwrap();
    to_writer(writer, &proof).unwrap();



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

fn run_verify<C: Config, I, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
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

    let config = expander_config::Config::<<C>::DefaultGKRConfig>::new(
        expander_config::GKRScheme::Vanilla,
        mpi_config::MPIConfig::new(),
    );

    let file = std::fs::File::open(format!("trivial_circuit.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let mut expander_circuit = layered_circuit
        .export_to_expander::<<C>::DefaultGKRFieldConfig>()
        .flatten();

    let file = std::fs::File::open(format!("trivial_witness.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader).unwrap();
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd::<C::DefaultSimdField>();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    let file = std::fs::File::open(format!("trivial_proof.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> = from_reader(reader).unwrap();



    let (proof, claimed_v) = match gkr::executor::load_proof_and_claimed_v::<<<C as Config>::DefaultGKRFieldConfig as GKRFieldConfig>::ChallengeField>(&proof_and_claimed_v) {
        Ok((proof, claimed_v)) => (proof, claimed_v),
        Err(_) => {
            return;
        }
    };

    // verify
    assert!(gkr::executor::verify(
        &mut expander_circuit,
        &config,
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
fn run_verify_no_circuit<C: Config, I, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
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

    let config = expander_config::Config::<<C>::DefaultGKRConfig>::new(
        expander_config::GKRScheme::Vanilla,
        mpi_config::MPIConfig::new(),
    );

    let file = std::fs::File::open(format!("trivial_circuit.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let layered_circuit = Circuit::<C, NormalInputType>::deserialize_from(reader).unwrap();

    let mut expander_circuit = layered_circuit
        .export_to_expander::<<C>::DefaultGKRFieldConfig>()
        .flatten();

    let file = std::fs::File::open(format!("trivial_witness.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let witness = Witness::<C>::deserialize_from(reader).unwrap();
    // let witness =
    //     layered::witness::Witness::<C>::deserialize_from(witness).map_err(|e| e.to_string())?;
    let (simd_input, simd_public_input) = witness.to_simd::<C::DefaultSimdField>();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    let file = std::fs::File::open(format!("trivial_proof.txt")).unwrap();
    let reader = std::io::BufReader::new(file);
    let proof_and_claimed_v: Vec<u8> = from_reader(reader).unwrap();



    let (proof, claimed_v) = match gkr::executor::load_proof_and_claimed_v::<<<C as Config>::DefaultGKRFieldConfig as GKRFieldConfig>::ChallengeField>(&proof_and_claimed_v) {
        Ok((proof, claimed_v)) => (proof, claimed_v),
        Err(_) => {
            return;
        }
    };

    // verify
    assert!(gkr::executor::verify(
        &mut expander_circuit,
        &config,
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



fn run_debug<C: Config, I, CircuitType, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitType: Default
        + DumpLoadTwoVariables<Variable>
        // + expander_compiler::frontend::Define<C>
        + Clone
        + GenericDefine<C>,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
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

    // let compile_result: CompileResult<C> = compile(&CircuitType::default()).unwrap();
    let compile_result =
        compile_generic(&CircuitType::default(), CompileOptions::default()).unwrap();
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
    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;

    let assignment = CircuitDefaultType::default();
    let _input_reader = FileReader {
        path: input_path.clone(),
    };
    let _output_reader = FileReader {
        path: output_path.clone(),
    };
    let assignment = io_reader.read_inputs(input_path, assignment);
    let assignment = io_reader.read_outputs(output_path, assignment);

    let assignments = vec![assignment.clone(); 1];
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    let _output = layered_circuit.run(&witness);

    debug_eval(&CircuitType::default(), &assignment, EmptyHintCaller);
}




// pub fn run_gf2<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::GF2Config, CircuitDefaultType>>(file_reader: &mut Filereader)
// where
//     CircuitDefaultType: std::default::Default
//     + DumpLoadTwoVariables<<expander_compiler::frontend::GF2Config as expander_compiler::frontend::Config>::CircuitField>
//     + std::clone::Clone,

//     CircuitType: std::default::Default +
//     expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
//     + expander_compiler::frontend::GenericDefine<expander_compiler::frontend::GF2Config>
//     + std::clone::Clone,
// {
//     run_main::<GF2Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
// }

// pub fn run_m31<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::M31Config, CircuitDefaultType>>(file_reader: &mut Filereader)
// where
//     CircuitDefaultType: std::default::Default
//     + DumpLoadTwoVariables<<expander_compiler::frontend::M31Config as expander_compiler::frontend::Config>::CircuitField>
//     + std::clone::Clone,

//     CircuitType: std::default::Default +
//     expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
//     + expander_compiler::frontend::GenericDefine<expander_compiler::frontend::M31Config>
//     + std::clone::Clone,
// {
//     run_main::<M31Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
// }

pub fn run_bn254<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::BN254Config, CircuitDefaultType>>(file_reader: &mut Filereader)
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
    + expander_compiler::frontend::GenericDefine<expander_compiler::frontend::BN254Config>
    
    + std::clone::Clone,
{
    // run_main::<BN254Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);

    run_compile_and_serialize::<BN254Config, CircuitType>();


    // run_rest::<BN254Config, Filereader, CircuitDefaultType>(file_reader);

    run_witness_and_proof::<BN254Config, Filereader, CircuitDefaultType>(file_reader);
    run_verify::<BN254Config, Filereader, CircuitDefaultType>(file_reader);
    


}


pub fn debug_bn254<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::BN254Config, CircuitDefaultType>>(file_reader: &mut Filereader)
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
    + std::clone::Clone + GenericDefine<expander_compiler::frontend::BN254Config>,
{
    run_debug::<BN254Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
}
