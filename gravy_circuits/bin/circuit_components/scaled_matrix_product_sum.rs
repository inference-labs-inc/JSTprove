use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::circuit_functions::matrix_computation::scaled_matrix_product_sum;
use serde::Deserialize;
// use std::ops::Neg;
use gravy_circuits::runner::main_runner::handle_args;

/*
Step 3: scalar times matrix product of two matrices of compatible dimensions, plus a third matrix of campatible dimensions.
scaling factor alpha is an integer
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix c has shape (m, k)
scaled matrix product plus matrix alpha ab + c has shape (m, k)
*/

const N_ROWS_A: usize = 3; // m
const N_COLS_A: usize = 4; // n
const N_ROWS_B: usize = 4; // n
const N_COLS_B: usize = 2; // k
const N_ROWS_C: usize = 3; // m
const N_COLS_C: usize = 2; // k

declare_circuit!(Circuit {
    alpha: Variable,                            // scaling factor
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B], // shape (n, k)
    matrix_c: [[Variable; N_COLS_C]; N_ROWS_C], // shape (m, k)
    scaled_matrix_product_sum_alpha_ab_plus_c: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});

// Still to factor this
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let scaled_row_col_product_sum_array =
            scaled_matrix_product_sum(api, self.matrix_a, self.matrix_b, self.matrix_c, self.alpha);
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_B {
                api.assert_is_equal(
                    self.scaled_matrix_product_sum_alpha_ab_plus_c[i][j],
                    scaled_row_col_product_sum_array[i][j],
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
    matrix_c: Vec<Vec<u64>>, // Shape (n, k)
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    scaled_matrix_product_sum_alpha_ab_plus_c: Vec<Vec<u64>>,
}

impl<C: Config> IOReader<Circuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField::<C>>,
    ) -> Circuit<CircuitField::<C>> {
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path);

        // Assign inputs to assignment
        assignment.alpha = CircuitField::<C>::from_u256(U256::from(data.alpha));

        let rows_a = data.matrix_a.len();
        let cols_a = if rows_a > 0 {
            data.matrix_a[0].len()
        } else {
            0
        };
        println!("matrix a shape: ({}, {})", rows_a, cols_a);

        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = CircuitField::<C>::from_u256(U256::from(element));
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
                assignment.matrix_b[i][j] = CircuitField::<C>::from_u256(U256::from(element));
            }
        }

        let rows_c = data.matrix_c.len();
        let cols_c = if rows_c > 0 {
            data.matrix_c[0].len()
        } else {
            0
        };
        println!("matrix c shape: ({}, {})", rows_c, cols_c);

        for (i, row) in data.matrix_c.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_c[i][j] = CircuitField::<C>::from_u256(U256::from(element));
            }
        }

        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField::<C>>,
    ) -> Circuit<CircuitField::<C>> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path);
        // Assign inputs to assignment
        let rows_abc = data.scaled_matrix_product_sum_alpha_ab_plus_c.len();
        let cols_abc = if rows_abc > 0 {
            data.scaled_matrix_product_sum_alpha_ab_plus_c[0].len()
        } else {
            0
        };
        println!(
            "scaled matrix product alpha ab plus matrix c shape: ({}, {})",
            rows_abc, cols_abc
        );

        for (i, row) in data
            .scaled_matrix_product_sum_alpha_ab_plus_c
            .iter()
            .enumerate()
        {
            for (j, &element) in row.iter().enumerate() {
                assignment.scaled_matrix_product_sum_alpha_ab_plus_c[i][j] =
                    CircuitField::<C>::from_u256(U256::from(element));
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
        path: "scaled_matrix_product_sum".to_owned(),
    };
    // run_gf2();
    // run_m31();
    handle_args::<BN254Config, Circuit<Variable>,Circuit<_>,_>(&mut file_reader);
    // handle_args::<M31Config, Circuit<Variable>,Circuit<_>,_>(&mut file_reader);
    // handle_args::<GF2Config, Circuit<Variable>,Circuit<_>,_>(&mut file_reader);
}
