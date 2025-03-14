use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::circuit_functions::matrix_computation::scaled_matrix_product;
use serde::Deserialize;
// use std::ops::Neg;
use arith::FieldForECC;

use gravy_circuits::runner::main_runner;

/*
Step 2: scalar times matrix product of two matrices of compatible dimensions.
scaling factor alpha is an integer
matrix a has shape (m, n)
matrix b has shape (n, k)
scaled matrix product alpha ab has shape (m, k)
*/

const N_ROWS_A: usize = 3; // m
const N_COLS_A: usize = 4; // n
const N_ROWS_B: usize = 4; // n
const N_COLS_B: usize = 2; // k

declare_circuit!(Circuit {
    alpha: Variable,                                                  // scaling factor
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A],                       // shape (m, n)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B],                       // shape (n, k)
    scaled_matrix_product_alpha_ab: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});

//Still to factor this out
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let scaled_matrix_product =
            scaled_matrix_product(api, self.matrix_a, self.matrix_b, self.alpha);
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_B {
                api.assert_is_equal(
                    self.scaled_matrix_product_alpha_ab[i][j],
                    scaled_matrix_product[i][j],
                );
            }
        }
    }
}

#[derive(Deserialize, Clone)]
struct InputData {
    alpha: u64,
    matrix_a: Vec<Vec<u64>>, // Shape (m, n)
    matrix_b: Vec<Vec<u64>>, // Shape (n, k)
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    scaled_matrix_product_alpha_ab: Vec<Vec<u64>>,
}

impl<C: Config> IOReader<Circuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<C::CircuitField>,
    ) -> Circuit<C::CircuitField> {
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path);

        // Assign inputs to assignment
        assignment.alpha = C::CircuitField::from_u256(U256::from(data.alpha));

        let rows_a = data.matrix_a.len();
        let cols_a = if rows_a > 0 {
            data.matrix_a[0].len()
        } else {
            0
        };
        println!("matrix a shape: ({}, {})", rows_a, cols_a);

        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from_u256(U256::from(element));
            }
        }

        let rows_b = data.matrix_b.len();
        let cols_b = if rows_b > 0 {
            data.matrix_b[0].len()
        } else {
            0
        };
        println!("matrix b shape: ({}, {})", rows_b, cols_b);

        for (i, row) in data.matrix_b.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_b[i][j] = C::CircuitField::from_u256(U256::from(element));
            }
        }

        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<C::CircuitField>,
    ) -> Circuit<C::CircuitField> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path);

        // Assign inputs to assignment
        let rows_ab = data.scaled_matrix_product_alpha_ab.len();
        let cols_ab = if rows_ab > 0 {
            data.scaled_matrix_product_alpha_ab[0].len()
        } else {
            0
        };
        println!(
            "scaled matrix product alpha ab shape: ({}, {})",
            rows_ab, cols_ab
        );

        for (i, row) in data.scaled_matrix_product_alpha_ab.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.scaled_matrix_product_alpha_ab[i][j] =
                    C::CircuitField::from_u256(U256::from(element));
            }
        }
        assignment
    }
    fn get_path(&self) -> &str {
        &self.path
    }
}

/*
        #######################################################################################################
        #####################################  Shouldn't need to change  ######################################
        #######################################################################################################
*/

fn main() {
    let mut file_reader = FileReader {
        path: "scaled_matrix_product".to_owned(),
    };
    // run_gf2();
    // run_m31();
    main_runner::run_bn254::<Circuit<Variable>,
    Circuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
}
