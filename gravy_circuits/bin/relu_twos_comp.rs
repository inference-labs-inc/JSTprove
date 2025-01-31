use expander_compiler::frontend::*;
use clap::{Command, Arg};
use io_reader::{FileReader, IOReader};
use peakmem_alloc::*;
use std::alloc::System;
use std::time::Instant;
use serde::Deserialize;
use ethnum::U256;
use std::{io::Read, ops::Neg};
use arith::FieldForECC;

use expander_compiler::frontend::*;
#[path = "../src/relu.rs"]
pub mod relu;

#[path = "../src/io_reader.rs"]
pub mod io_reader;


#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;


/*
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
 */

// Specify input and output structure
// This will indicate the input layer and output layer of the circuit, so be careful with how it is defined
// Later, we define how the inputs get read into the input layer
const SIZE1: usize = 28;
const SIZE2: usize = 28;
const SIZE3: usize = 16;

declare_circuit!(Circuit {
    input: [[[Variable; SIZE1]; SIZE2]; SIZE3],
    output: [[[Variable; SIZE1]; SIZE2]; SIZE3],
});
impl<C: Config> Define<C> for Circuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define(&self, api: &mut API<C>) {
        let n_bits = 32;



        // let out = relu::relu_3d_naive(api, self.input,n_bits);

        let out = relu::relu_3d_v2(api, self.input, n_bits);
        // let out = relu::relu_3d_v3(api, self.input);


    
        for i in 0..SIZE3{
            for j in 0..SIZE2{
                for k in 0..SIZE1{
                    api.assert_is_equal(self.output[i][j][k], out[i][j][k]);
                }
            }
        }
    }
}
/*
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
 */


#[derive(Deserialize)]
#[derive(Clone)]
struct OutputData {
    outputs: Vec<Vec<Vec<u64>>>,
}
#[derive(Deserialize)]
#[derive(Clone)]struct InputData {
    inputs_1: Vec<Vec<Vec<i64>>>,
}
impl<C: Config>IOReader<C> for FileReader{
    type CircuitType = Circuit<<C as Config>::CircuitField>;
    // fn read_inputs(&mut self, file_path: &str, mut assignment: &mut Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    fn read_inputs(&mut self, file_path: &str, mut assignment: Self::CircuitType) -> Self::CircuitType
    {
        println!("{}", file_path);
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: InputData = serde_json::from_str(&contents).unwrap();
        // Assign inputs to assignment
        

        for (i,var_vec_vec) in data.inputs_1.iter().enumerate(){
            for (j, var_vec) in var_vec_vec.iter().enumerate(){
                for (k, &var) in var_vec.iter().enumerate(){
                    if var < 0{
                        assignment.input[i][j][k] = C::CircuitField::from(var.abs() as u32).neg();
                    }
                    else{
                        assignment.input[i][j][k] = C::CircuitField::from(var.abs() as u32);
                    }
                    // assignment.input[i][j][k] = C::CircuitField::from_u256(U256::from(var)); 
                }
            }
        }
        assignment
    }
    fn read_outputs(&mut self, file_path: &str, mut assignment: Self::CircuitType) -> Self::CircuitType
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: OutputData = serde_json::from_str(&contents).unwrap();

        // Assign inputs to assignment

        for (i,var_vec_vec) in data.outputs.iter().enumerate(){
            for (j, var_vec) in var_vec_vec.iter().enumerate(){
                for (k, &var) in var_vec.iter().enumerate(){
                    assignment.output[i][j][k] = C::CircuitField::from_u256(U256::from(var));

                }
            }
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
    println!(
        "Peak Memory used Overall : {:.2}", 
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!("Time elapsed: {}.{} seconds", duration.as_secs(), duration.subsec_millis());

    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now(); 
    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;

    let assignment = Circuit::<C::CircuitField>::default();
    // let assignment = io_reader::input_data_from_json::<C>(input_path, assignment);
    // let assignment = io_reader::output_data_from_json::<C>(output_path, assignment);
    let mut input_reader = FileReader{path: input_path.clone()};
    let mut output_reader = FileReader{path: output_path.clone()};
    let assignment = <FileReader as IOReader<C>>::read_inputs(&mut input_reader,input_path, assignment);
    let assignment = <FileReader as IOReader<C>>::read_outputs(&mut output_reader,output_path, assignment);

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

    // println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
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
    // run_gf2();
    // run_m31();
    run_bn254();
}