
use expander_compiler::frontend::*;
use helper_fn::load_circuit_constant;
use io_reader::{FileReader, IOReader};
#[allow(unused_imports)]
use matrix_computation::{matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
     matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
      matrix_multplication_naive3_array, two_d_array_to_vec};
use serde::Deserialize;
use ethnum::U256;
use std::ops::Neg;
use arith::FieldForECC;
use lazy_static::lazy_static;

#[path = "../src/matrix_computation.rs"]
pub mod matrix_computation;

#[path = "../src/io_reader.rs"]
pub mod io_reader;
#[path = "../src/main_runner.rs"]
pub mod main_runner;
#[path = "../src/helper_fn.rs"]
pub mod helper_fn;

/* 
Part 2 (memorization), Step 1: vanilla matrix multiplication of two matrices of compatible dimensions.
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix product ab has shape (m, k)
*/

const DIM1: usize = 1; // m
const DIM2: usize = 4; // n
const DIM3: usize = 28; // n
const DIM4: usize = 28; // k

//Define structure of inputs, weights and output
#[derive(Deserialize)]
#[derive(Clone)]
struct WeightsData {
    weights: Vec<Vec<Vec<Vec<i64>>>>,
    bias: Vec<i64>,
    strides: Vec<u8>,
    kernel_shape: Vec<u8>,
    group: Vec<u8>,
    dilation: Vec<u8>,
    pads: Vec<u8>,
    input_shape: Vec<u8>,
} 

#[derive(Deserialize)]
#[derive(Clone)]
struct InputData {
    input_arr: Vec<Vec<Vec<Vec<i64>>>>,
} 

#[derive(Deserialize)]
#[derive(Clone)]
struct OutputData {
    conv_out: Vec<Vec<Vec<Vec<i64>>>>,
} 

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/convolution_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData = serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
        
        // let mut y: Vec<Vec<u64>> = Vec::new();
        // for (_, row) in x.matrix_b.iter().enumerate() {
        //     let mut z: Vec<u64> = Vec::new();
        //     for (_, &element) in row.iter().enumerate() {
        //         z.push(element);
        //     }
        //     y.push(z);
        // }
        // y
        };

}

fn read_4d_weights<C: Config>(api: &mut API<C>, weights_data: &Vec<Vec<Vec<Vec<i64>>>>) -> Vec<Vec<Vec<Vec<Variable>>>>{
    let weights: Vec<Vec<Vec<Vec<Variable>>>> = weights_data.clone()
    .into_iter()
    .map(|dim1| dim1.into_iter()
        .map(|dim2| dim2.into_iter()
            .map(|dim3| dim3.into_iter()
                .map(|x| load_circuit_constant(api, x))
                .collect()
            )
            .collect()
        )
        .collect()
    )
    .collect();
weights
}



declare_circuit!(ConvCircuit {
    input_arr: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, n)
    conv_out: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, k)
});
// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable,> {
    fn define(&self, api: &mut API<C>) {
        // Bring the weights into the circuit as constants

        let weights = read_4d_weights(api, &WEIGHTS_INPUT.weights);
        let bias: Vec<Variable> = WEIGHTS_INPUT.bias.clone().into_iter().map(|x|  load_circuit_constant(api, x)).collect();

        //Assert output of matrix multiplication
        for (j,dim1) in self.input_arr.iter().enumerate() {
            for (k,dim2) in dim1.iter().enumerate() {
                for (l,dim3) in dim2.iter().enumerate() {
                    for (m,dim4) in dim3.iter().enumerate() {
                        api.assert_is_equal(self.conv_out[j][k][l][m], dim4);
                    }
                }
            }
        }

    }
}

impl<C: Config>IOReader<C, ConvCircuit<C::CircuitField>> for FileReader
{
    fn read_inputs(&mut self, file_path: &str, mut assignment: ConvCircuit<C::CircuitField>) -> ConvCircuit<C::CircuitField>
    {
        let data: InputData = <FileReader as IOReader<C, ConvCircuit<_>>>::read_data_from_json::<InputData>(file_path); 


        // Assign inputs to assignment
        for (i,dim1) in data.input_arr.iter().enumerate() {
            for (j,dim2) in dim1.iter().enumerate() {
                for (k,dim3) in dim2.iter().enumerate() {
                    for (l,&element) in dim3.iter().enumerate() {
                        if element < 0{
                            assignment.input_arr[i][j][k][l] = C::CircuitField::from(element.abs() as u32).neg();
                        }
                        else{
                            assignment.input_arr[i][j][k][l] = C::CircuitField::from(element.abs() as u32);
                        }
                    }
                }
            }
        }
        // Return the assignment
        assignment
    }
    fn read_outputs(&mut self, file_path: &str, mut assignment: ConvCircuit<C::CircuitField>) -> ConvCircuit<C::CircuitField>
    {

        let data: OutputData = <FileReader as IOReader<C, ConvCircuit<_>>>::read_data_from_json::<OutputData>(file_path); 

        for (i,dim1) in data.conv_out.iter().enumerate() {
            for (j,dim2) in dim1.iter().enumerate() {
                for (k,dim3) in dim2.iter().enumerate() {
                    for (l,&element) in dim3.iter().enumerate() {
                        if element < 0{
                            assignment.conv_out[i][j][k][l] = C::CircuitField::from(element.abs() as u32).neg();
                        }
                        else{
                            assignment.conv_out[i][j][k][l] = C::CircuitField::from(element.abs() as u32);
                        }
                    }
                }
            }
        }
        // Return the assignment
        assignment
    }
}


fn main(){
    let mut file_reader = FileReader{path: String::new()};
    // run_gf2();
    // run_m31();
    main_runner::run_bn254::<ConvCircuit<Variable>,
    ConvCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);

}