use ethnum::U256;
use expander_compiler::frontend::*;
use helper_fn::read_2d_weights;
use io_reader::{FileReader, IOReader};
use matrix_computation::{gemm_vec, two_d_array_to_vec};
use quantization::quantize_matrix;
use serde::Deserialize;
// use std::ops::Neg;
use arith::FieldForECC;
use lazy_static::lazy_static;


#[path = "../../src/matrix_computation.rs"]
pub mod matrix_computation;
#[path = "../../src/quantization.rs"]
pub mod quantization;

#[path = "../../src/helper_fn.rs"]
pub mod helper_fn;

#[path = "../../src/io_reader.rs"]
pub mod io_reader;
#[path = "../../src/main_runner.rs"]
pub mod main_runner;

/*
Step 4: general matrix multiplication---scalar times matrix product of two matrices of compatible dimensions, plus scalar times third matrix of campatible dimensions.
scaling factor alpha is an integer
scaling factor beta is an integer
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix c has shape (m, k)
general matrix palpha ab + beta c has shape (m, k)
*/
#[derive(Deserialize, Clone)]
struct WeightsData {
    alpha: u32,
    beta: u32,
    weights: Vec<Vec<i64>>,
    bias: Vec<Vec<i64>>,
    quantized: bool,
    scaling: u32,
}

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<u64>>, // Shape (m, n)
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<u64>>,
}

const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/gemm_weights.json");

lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}

const N_ROWS_A: usize = 3; // m
const N_COLS_A: usize = 4; // n
// const N_ROWS_B: usize = 4; // n
const N_COLS_B: usize = 2; // k
// const N_ROWS_C: usize = 3; // m
// const N_COLS_C: usize = 2; // k

declare_circuit!(Circuit {                             // scaling factor
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    gemm: [[Variable; N_COLS_B]; N_ROWS_A],     // shape (m, k)
});

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let v_plus_one: usize = 32;
        let two_v: u32 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        let weights = read_2d_weights(api, &WEIGHTS_INPUT.weights);
        let bias = read_2d_weights(api, &WEIGHTS_INPUT.bias);

        let alpha = api.constant(WEIGHTS_INPUT.alpha);
        let beta = api.constant(WEIGHTS_INPUT.beta);

        let mut gemm_array = gemm_vec(
            api,
            two_d_array_to_vec(self.matrix_a),
            weights,
            bias,
            alpha,
            beta,
        );
        if WEIGHTS_INPUT.quantized {
            // let scaling = api.constant(weights.scaling as u32);
            // let scaling_factor = scaling_factor_to_constant(api, )
            gemm_array = quantize_matrix(api, gemm_array, scaling_factor, WEIGHTS_INPUT.scaling as usize, v_plus_one, two_v, alpha_2_v, false);
        }

        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_B {
                api.assert_is_equal(self.gemm[i][j], gemm_array[i][j]);
            }
        }
    }
}



impl<C: Config> IOReader<C, Circuit<C::CircuitField>> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<C::CircuitField>,
    ) -> Circuit<C::CircuitField> {
        let data: InputData =
            <FileReader as IOReader<C, Circuit<_>>>::read_data_from_json::<InputData>(file_path);

        let rows_a = data.input.len();
        let cols_a = if rows_a > 0 {
            data.input[0].len()
        } else {
            0
        };
        println!("matrix a shape: ({}, {})", rows_a, cols_a);

        for (i, row) in data.input.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from_u256(U256::from(element));
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
            <FileReader as IOReader<C, Circuit<_>>>::read_data_from_json::<OutputData>(file_path);
        // Assign inputs to assignment
        let rows_abc = data.output.len();
        let cols_abc = if rows_abc > 0 { data.output[0].len() } else { 0 };
        println!("gemm alpha ab + beta c shape: ({}, {})", rows_abc, cols_abc);

        for (i, row) in data.output.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.gemm[i][j] = C::CircuitField::from_u256(U256::from(element));
            }
        }
        assignment
    }
}

/*
        #######################################################################################################
        #####################################  Shouldn't need to change  ######################################
        #######################################################################################################
*/

fn main() {
    let mut file_reader = FileReader {
        path: String::new(),
    };
    main_runner::run_bn254::<Circuit<Variable>,
    Circuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
}
