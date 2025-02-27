use crate::io_reader::{FileReader, IOReader};
use clap::{Arg, Command};
use expander_compiler::frontend::{extra::debug_eval, internal::DumpLoadTwoVariables, *};
use peakmem_alloc::*;
use std::alloc::System;
use std::time::Instant;

#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

fn run_main<C: Config, I, CircuitType, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitType: Default + DumpLoadTwoVariables<Variable> + Define<C> + Clone,
    CircuitDefaultType: Default
        + DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
        + Clone
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
        compile(&CircuitType::default(), CompileOptions::default().with_mul_fanout_limit(1024)).unwrap();
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

    // let mut expander_circuit = layered_circuit
    //     .export_to_expander::<<C>::DefaultGKRFieldConfig>()
    //     .flatten();
    let mut expander_circuit = layered_circuit
        .export_to_expander::<C::DefaultGKRFieldConfig>()
        .flatten::<C::DefaultGKRConfig>();
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

fn run_debug<C: Config, I, CircuitType, CircuitDefaultType>(io_reader: &mut I)
where
    I: IOReader<C, CircuitDefaultType>, // `CircuitType` should be the same type used in the `IOReader` impl
    CircuitType: Default
        + DumpLoadTwoVariables<Variable>
        // + expander_compiler::frontend::Define<C>
        + Clone
        + Define<C>,
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
        compile(&CircuitType::default(), CompileOptions::default()).unwrap();
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
    // let assignment = io_reader::input_data_from_json::<C>(input_path, assignment);
    // let assignment = io_reader::output_data_from_json::<C>(output_path, assignment);
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




pub fn run_gf2<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::GF2Config, CircuitDefaultType>>(file_reader: &mut Filereader)
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<expander_compiler::frontend::GF2Config as expander_compiler::frontend::Config>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
    + expander_compiler::frontend::Define<expander_compiler::frontend::GF2Config>
    + std::clone::Clone,
{
    run_main::<GF2Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
    // run_main::<GF2Config, GF2ExtConfigSha2Raw, Filereader, CircuitType, CircuitDefaultType>(file_reader);

}

pub fn run_m31<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::M31Config, CircuitDefaultType>>(file_reader: &mut Filereader)
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<expander_compiler::frontend::M31Config as expander_compiler::frontend::Config>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
    + expander_compiler::frontend::Define<expander_compiler::frontend::M31Config>
    + std::clone::Clone,
{
    run_main::<M31Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
    // run_main::<M31Config, M31ExtConfigSha2Raw, Filereader, CircuitType, CircuitDefaultType>(file_reader);

}

pub fn run_bn254<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::BN254Config, CircuitDefaultType>>(file_reader: &mut Filereader)
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
    + expander_compiler::frontend::Define<expander_compiler::frontend::BN254Config>
    + std::clone::Clone,
{
    run_main::<BN254Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
    // run_main::<BN254Config, BN254ConfigSha2Hyrax, Filereader, CircuitType, CircuitDefaultType>(file_reader);

}


pub fn debug_bn254<CircuitType, CircuitDefaultType, Filereader: IOReader<expander_compiler::frontend::BN254Config, CircuitDefaultType>>(file_reader: &mut Filereader)
where
    CircuitDefaultType: std::default::Default
    + DumpLoadTwoVariables<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>
    + std::clone::Clone,

    CircuitType: std::default::Default +
    expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable>
// + expander_compiler::frontend::Define<expander_compiler::frontend::BN254Config>
    + std::clone::Clone + Define<expander_compiler::frontend::BN254Config>,
{
    run_debug::<BN254Config, Filereader, CircuitType, CircuitDefaultType>(file_reader);
}
