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

// use expander_compiler::frontend::*;
use expander_compiler::frontend::{internal::DumpLoadTwoVariables, *};

#[path = "../src/relu.rs"]
pub mod relu;

#[path = "../src/io_reader.rs"]
pub mod io_reader;
#[path = "../src/main_runner.rs"]
pub mod main_runner;


// #[global_allocator]
// static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;


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

declare_circuit!(ReLUTwosCircuit {
    input: [[[Variable; SIZE1]; SIZE2]; SIZE3],
    output: [[[Variable; SIZE1]; SIZE2]; SIZE3],
});
impl<C: Config> Define<C> for ReLUTwosCircuit<Variable> {
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
impl<C: Config>IOReader<C, ReLUTwosCircuit<C::CircuitField>> for FileReader
// where ReLUTwosCircuit<<C as expander_compiler::frontend::Config>::CircuitField>: expander_compiler::frontend::Define<C> +
// DumpLoadTwoVariables<expander_compiler::frontend::Variable>
// where ReLUTwosCircuit<expander_compiler::frontend::Variable>: DumpLoadTwoVariables<<C as expander_compiler::frontend::Config>::CircuitField>
{
    // fn read_inputs(&mut self, file_path: &str, mut assignment: &mut Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    fn read_inputs(&mut self, file_path: &str, mut assignment: ReLUTwosCircuit<C::CircuitField>) -> ReLUTwosCircuit<C::CircuitField>
    {
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
    fn read_outputs(&mut self, file_path: &str, mut assignment: ReLUTwosCircuit<C::CircuitField>) -> ReLUTwosCircuit<C::CircuitField>
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

fn main(){
    // let assignment = Circuit::<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>::default();
    // run_gf2();
    // run_m31();
    let mut file_reader = FileReader{path: String::new()};
    main_runner::run_bn254::<ReLUTwosCircuit<Variable>,
                            ReLUTwosCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);

}