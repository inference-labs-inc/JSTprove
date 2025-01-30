
use expander_compiler::frontend::*;
use clap::{Command, Arg};
use peakmem_alloc::*;
use std::alloc::System;
use std::mem;
use std::time::Instant;


#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;
const LENGTH: usize = 100;



declare_circuit!(Circuit {
    input: [[Variable; LENGTH]; 2],
    output: [Variable; LENGTH],
});

impl<C: Config> Define<C> for Circuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define(&self, api: &mut API<C>) {
        // Iterate over each input/output pair (one per batch)
        for i in 0..LENGTH {
            let out = api.add(self.input[0][i],self.input[1][i]);
            api.assert_is_equal(out, self.output[i]);
        }
    }
}

mod io_reader {
    use ethnum::U256;
    use std::io::Read;
    use arith::FieldForECC;
    use serde::Deserialize;

    use super::Circuit;

    use expander_compiler::frontend::*;

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct InputData {
        pub(crate) inputs_1: Vec<u64>,
        pub(crate) inputs_2: Vec<u64>,
    }

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) outputs: Vec<u64>,
    }

    pub(crate) fn input_data_from_json<C: Config>(file_path: &str, mut assignment: Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: InputData = serde_json::from_str(&contents).unwrap();


        for (k, &var) in data.inputs_1.iter().enumerate() {
        // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
            assignment.input[0][k] = C::CircuitField::from_u256(U256::from(var)) ; // Make sure var in this line, lines up with the innermost loop variable name
        }

        for (k, &var) in data.inputs_2.iter().enumerate() {
            // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
                assignment.input[1][k] = C::CircuitField::from_u256(U256::from(var)) ; // Make sure var in this line, lines up with the innermost loop variable name
            }

        // Return the assignment
        assignment
    }

    pub(crate) fn output_data_from_json<C: Config>(file_path: &str, mut assignment: Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: OutputData = serde_json::from_str(&contents).unwrap();

        // Assign inputs to assignment
        for (k, &var) in data.outputs.iter().enumerate() {
            // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
            assignment.output[k] = C::CircuitField::from_u256(U256::from(var)) ; // Make sure var in this line, lines up with the innermost loop variable name
        }
        assignment
    }
}

fn run_main<C: Config>()
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now(); 
    let matches = Command::new("File Copier")
        .version("1.0")
        .about("Copies content from input file to output file")
        .arg(
            Arg::new("input")
                .help("The input file to read from")
                .required(true)  // This argument is required
                .index(1),       // Positional argument (first argument)
        )
        .arg(
            Arg::new("output")
                .help("The output file to write to")
                .required(true)  // This argument is also required
                .index(2),       // Positional argument (second argument)
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();// "inputs/reward_input.json"
    let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"



    let compile_result: CompileResult<C> = compile(&Circuit::default()).unwrap();
    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;

    let mut assignment = Circuit::<C::CircuitField>::default();

    let assignment = io_reader::input_data_from_json::<C>(input_path, assignment);

    let assignment = io_reader::output_data_from_json::<C>(output_path, assignment);

    let assignments = vec![assignment; 1];
    let witness = witness_solver
        .solve_witnesses(&assignments)
        .unwrap();
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

    let (simd_input, simd_public_input) =
        witness.to_simd::<<C>::DefaultSimdField>();
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

    println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}", 
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!("Time elapsed: {}.{} seconds", duration.as_secs(), duration.subsec_millis())

}

//#[test]
#[allow(dead_code)]
fn run_gf2() {
    run_main::<GF2Config>();
}

//#[test]
#[allow(dead_code)]
fn run_m31() {
    run_main::<M31Config>();
}

//#[test]
#[allow(dead_code)]
fn run_bn254() {
    run_main::<BN254Config>();
}

fn main(){
    run_gf2();
    run_m31();
    run_bn254();
}