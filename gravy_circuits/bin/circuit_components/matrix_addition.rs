use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use serde::Deserialize;
use ethnum::U256;
// use std::ops::Neg;
// use arith::FieldForECC;
use gravy_circuits::circuit_functions::matrix_computation::{matrix_addition, matrix_hadamard_product};
use gravy_circuits::runner::main_runner;


/* 
Step 1: vanilla matrix addition of two matrices of compatible dimensions.
matrix a has shape (m, n)
matrix b has shape (m, n)
matrix sum a + b has shape (m, n)
*/

const N_ROWS_A: usize = 17571; // m
const N_COLS_A: usize = 1; // n
const N_ROWS_B: usize = 17571; // m
const N_COLS_B: usize = 1; // n

declare_circuit!(MatAddCircuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B], // shape (n, n)
    matrix_sum_ab: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
});

impl<C: Config> Define<C> for MatAddCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {  
        let matrix_sum = matrix_addition(api, self.matrix_a, self.matrix_b);
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_A {
                api.assert_is_equal(self.matrix_sum_ab[i][j], matrix_sum[i][j]); 
            }                          
        }
    }
}
declare_circuit!(TestCircuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B], // shape (n, n)
    matrix_sum_ab: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
});

impl<C: Config> Define<C> for TestCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {  
        let matrix_sum = matrix_hadamard_product(api, self.matrix_a, self.matrix_b);
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_A {
                api.assert_is_equal(self.matrix_sum_ab[i][j], matrix_sum[i][j]); 
            }                          
        }
        // let matrix_sum = matrix_computation::matrix_addition(api, self.matrix_a, self.matrix_b);
        // for i in 0..N_ROWS_A {
        //     for j in 0..N_COLS_A {
        //         api.assert_is_equal(self.matrix_sum_ab[i][j], matrix_sum[i][j]); 
        //     }                          
        // }
    }
}


#[derive(Deserialize)]
#[derive(Clone)]
struct InputData {
    matrix_a: Vec<Vec<u64>>, // Shape (m, n) 
    matrix_b: Vec<Vec<u64>>, // Shape (m, n) 
}

#[derive(Deserialize)]
#[derive(Clone)]
struct OutputData {
    matrix_sum_ab: Vec<Vec<u64>>, //  Shape (m, n) 
}
impl<C: Config>IOReader<TestCircuit<C::CircuitField>, C> for FileReader
{
    fn read_inputs(&mut self, file_path: &str, mut assignment: TestCircuit<C::CircuitField>) -> TestCircuit<C::CircuitField>
    {
        let data: InputData = <FileReader as IOReader<TestCircuit<_>, C>>::read_data_from_json::<InputData>(file_path); 


        // Assign inputs to assignment

        let rows_a = data.matrix_a.len();  
        let cols_a = if rows_a > 0 { data.matrix_a[0].len() } else { 0 };  
        println!("matrix a shape: ({}, {})", rows_a, cols_a);  
        
        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }

        let rows_b = data.matrix_b.len();  
        let cols_b = if rows_b > 0 { data.matrix_b[0].len() } else { 0 };  
        println!("matrix b shape: ({}, {})", rows_b, cols_b); 

        for (i, row) in data.matrix_b.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_b[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }

        // Return the assignment
        assignment
    }
    fn read_outputs(&mut self, file_path: &str, mut assignment: TestCircuit<C::CircuitField>) -> TestCircuit<C::CircuitField>
    {

        let data: OutputData = <FileReader as IOReader<TestCircuit<_>, C>>::read_data_from_json::<OutputData>(file_path); 

        // Assign inputs to assignment
        let rows_ab = data.matrix_sum_ab.len();  
        let cols_ab = if rows_ab > 0 { data.matrix_sum_ab[0].len() } else { 0 };  
        println!("matrix sum a + b shape: ({}, {})", rows_ab, cols_ab); 

        for (i, row) in data.matrix_sum_ab.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_sum_ab[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }
        assignment
    }
}


fn main(){
    let mut file_reader = FileReader{path: String::new()};
    // run_gf2();
    // run_m31();
    main_runner::run_bn254::<MatAddCircuit<Variable>,
                            TestCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
    //                         build::<M31Config>
    // main_runner::run_m31::<MatAddCircuit<Variable>,
    //                         MatAddCircuit<build::<M31Config>::CircuitField>,
    //                         _>(&mut file_reader);

}