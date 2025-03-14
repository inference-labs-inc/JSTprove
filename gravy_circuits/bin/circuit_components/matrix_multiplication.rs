use std::ops::Neg;
#[allow(unused_imports)]
use circuit_std_rs::logup::LogUpRangeProofTable;
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, matrix_multiplication_std
};
use gravy_circuits::circuit_functions::helper_fn::two_d_array_to_vec;

#[warn(unused_imports)]
use gravy_circuits::circuit_functions::quantization::quantize_matrix;
use gravy_circuits::runner::main_runner;
use serde::Deserialize;
// use std::ops::Neg;
use arith::FieldForECC;
use lazy_static::lazy_static;


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
#[derive(Deserialize, Clone)]
struct WeightsData {
    matrix_b: Vec<Vec<i64>>,
    quantized: bool,
    scaling: u32,
    circuit_type: String,
    is_relu: bool
}

#[derive(Deserialize, Clone)]
struct InputData {
    matrix_a: Vec<Vec<i64>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    matrix_product_ab: Vec<Vec<i64>>,
}

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/matrix_multiplication_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref weights: WeightsData = serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
}

declare_circuit!(MatMultCircuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_product_ab: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});
// Memorization, in a better place
impl<C: Config> Define<C> for MatMultCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let v_plus_one: usize = 32;
        let two_v: u32 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << weights.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        // let mut table_rem: LogUpRangeProofTable =LogUpRangeProofTable::new(21);
        // table_rem.initial(api);

        // Bring the weights into the circuit as constants
        let weights_matrix_multiplication: Vec<Vec<Variable>> = weights
            .matrix_b
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| {
                if x < 0 {
                    return api.constant(C::CircuitField::from(x.abs() as u32).neg())
                } else {
                    return api.constant(C::CircuitField::from(x.abs() as u32))
                }
            })
            .collect())
            .collect();


        // Compute matrix multiplication depending on specified inputs
        let mut out = match weights.circuit_type.as_str() {
            "naive_array" => matrix_multplication_naive3_array::<C, Builder, N_ROWS_A, N_COLS_A, N_COLS_B>(api, self.matrix_a,  weights_matrix_multiplication),
            //Traditional is bad
            "traditional" => matrix_multplication(api, two_d_array_to_vec(self.matrix_a),  weights_matrix_multiplication),
            "naive1" => matrix_multplication_naive(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication),
            "naive2" => matrix_multplication_naive2(api,two_d_array_to_vec(self.matrix_a),weights_matrix_multiplication),
            "naive3" => matrix_multplication_naive3(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication),
            // "naive2" => matrix_multiplication_std(api,two_d_array_to_vec(self.matrix_a),weights_matrix_multiplication, two_d_array_to_vec(self.matrix_product_ab)),
            _ => {
                panic!("No matching circuit_type");
            }
        };

        //If test is quantized tests, then add quantization
        if weights.quantized {
            // let scaling = api.constant(weights.scaling as u32);
            // let scaling_factor = scaling_factor_to_constant(api, )
            println!("{}", scaling_factor);
            out = quantize_matrix(api, out, scaling_factor, weights.scaling as usize, v_plus_one, two_v, alpha_2_v, weights.is_relu);
        }

        //Assert output of matrix multiplication
        for (j, row) in out.iter().enumerate() {
            for (k, &element) in row.iter().enumerate() {
                // api.display("out       ", element);
                // api.display("Out actual", self.matrix_product_ab[j][k]);
                api.assert_is_equal(self.matrix_product_ab[j][k], element);
            }
        }
    }
}

impl<C: Config> IOReader<MatMultCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: MatMultCircuit<C::CircuitField>,
    ) -> MatMultCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<MatMultCircuit<_>, C>>::read_data_from_json::<
            InputData,
        >(file_path);

        // Assign inputs to assignment
        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                if element < 0 {
                    assignment.matrix_a[i][j] = C::CircuitField::from(element.abs() as u32).neg();
                } else {
                    assignment.matrix_a[i][j] = C::CircuitField::from(element.abs() as u32);
                }
            }
        }
        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: MatMultCircuit<C::CircuitField>,
    ) -> MatMultCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<MatMultCircuit<_>, C>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for (i, row) in data.matrix_product_ab.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                if element < 0 {
                    assignment.matrix_product_ab[i][j] = C::CircuitField::from_u256(U256::from(element.abs() as u64)).neg();
                } else {
                    assignment.matrix_product_ab[i][j] = C::CircuitField::from_u256(U256::from(element.abs() as u64));
                }
            }
        }
        // Return the assignment
        assignment
    }
    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "matrix_multiplication".to_owned(),
    };
    // run_gf2();
    // run_m31();
    main_runner::run_bn254::<MatMultCircuit<Variable>,
    MatMultCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
    // main_runner::debug_bn254::<MatMultCircuit<Variable>,
    //                         MatMultCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
    //                                                 _>(&mut file_reader);
}
