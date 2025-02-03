
use expander_compiler::frontend::*;
use io_reader::{FileReader, IOReader};
#[allow(unused_imports)]
use matrix_computation::{matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
     matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
      matrix_multplication_naive3_array, two_d_array_to_vec};
use serde::Deserialize;
use ethnum::U256;
// use std::ops::Neg;
use arith::FieldForECC;
use lazy_static::lazy_static;

#[path = "../src/matrix_computation.rs"]
pub mod matrix_computation;

#[path = "../src/io_reader.rs"]
pub mod io_reader;
#[path = "../src/main_runner.rs"]
pub mod main_runner;

/* 
Part 2 (memorization), Step 1: vanilla matrix multiplication of two matrices of compatible dimensions.
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix product ab has shape (m, k)
*/

const N_ROWS_A: usize = 1; // m
const N_COLS_A: usize = 1568; // n
// const N_ROWS_B: usize = N_COLS_A; // n
const N_COLS_B: usize = 256; // k

//Define structure of inputs, weights and output
#[derive(Deserialize)]
#[derive(Clone)]
struct WeightsData {
    matrix_b: Vec<Vec<u64>>,
} 

#[derive(Deserialize)]
#[derive(Clone)]
struct InputData {
    matrix_a: Vec<Vec<u64>>,
} 

#[derive(Deserialize)]
#[derive(Clone)]
struct OutputData {
    matrix_product_ab: Vec<Vec<u64>>,
} 

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/matrix_multiplication_memorized_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref weights: Vec<Vec<u64>> = {
        let x: WeightsData = serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        
        let mut y: Vec<Vec<u64>> = Vec::new();
        for (_, row) in x.matrix_b.iter().enumerate() {
            let mut z: Vec<u64> = Vec::new();
            for (_, &element) in row.iter().enumerate() {
                z.push(element);
            }
            y.push(z);
        }
        y
        };

}



declare_circuit!(MatMultCircuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_product_ab: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});
// Memorization, in a better place
impl<C: Config> Define<C> for MatMultCircuit<Variable,> {
    fn define(&self, api: &mut API<C>) {
        // Bring the weights into the circuit as constants

        let weights_matrix_multiplication: Vec<Vec<Variable>> = weights.clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| api.constant(x as u32)).collect())
            .collect();
        
        // Compute matrix multiplication
        // let out:[[Variable; 256]; 1] = matrix_multplication_array(api, self.matrix_a,  weights_matrix_multiplication);
        // let out:[[Variable; N_COLS_B]; N_ROWS_A]  = matrix_multplication_naive2_array(api, self.matrix_a,  weights_matrix_multiplication);
        // let out:Vec<Vec<Variable>>  = matrix_multplication_naive3_array::<C,N_ROWS_A, N_COLS_A, N_COLS_B>(api, self.matrix_a,  weights_matrix_multiplication);


        // let out = matrix_multplication(api, two_d_array_to_vec(self.matrix_a),  weights_matrix_multiplication);
        // let out = matrix_multplication_naive(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication);
        let out = matrix_multplication_naive2(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication);
        // let out = matrix_multplication_naive3(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication);

        //Assert output of matrix multiplication
        for (j,row) in out.iter().enumerate() {
            for (k,&element) in row.iter().enumerate() {
                api.assert_is_equal(self.matrix_product_ab[j][k], element);
            }
        }

    }
}

impl<C: Config>IOReader<C, MatMultCircuit<C::CircuitField>> for FileReader
{
    fn read_inputs(&mut self, file_path: &str, mut assignment: MatMultCircuit<C::CircuitField>) -> MatMultCircuit<C::CircuitField>
    {
        let data: InputData = <FileReader as IOReader<C, MatMultCircuit<_>>>::read_data_from_json::<InputData>(file_path); 


        // Assign inputs to assignment
        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from(element as u32) ;
            }
        }
        // Return the assignment
        assignment
    }
    fn read_outputs(&mut self, file_path: &str, mut assignment: MatMultCircuit<C::CircuitField>) -> MatMultCircuit<C::CircuitField>
    {

        let data: OutputData = <FileReader as IOReader<C, MatMultCircuit<_>>>::read_data_from_json::<OutputData>(file_path); 

        for (i, row) in data.matrix_product_ab.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_product_ab[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
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
    main_runner::run_bn254::<MatMultCircuit<Variable>,
    MatMultCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);

}