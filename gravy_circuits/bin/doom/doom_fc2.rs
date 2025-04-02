use arith::FieldForECC;
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{four_d_array_to_vec, read_2d_weights, two_d_array_to_vec};
use gravy_circuits::circuit_functions::quantization::run_if_quantized_2d;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, matrix_addition_vec
};
// use gravy_circuits::circuit_functions::quantization::run_if_quantized_2d;
// use relu::{relu_2d_vec_v2, relu_4d_vec_v2};
use serde::Deserialize;
use std::ops::Neg;

use gravy_circuits::runner::main_runner::handle_args;


/*
Part 2 (memorization), Step 1: vanilla matrix multiplication of two matrices of compatible dimensions.
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix product ab has shape (m, k)
*/


//Define structure of inputs, weights and output
#[derive(Deserialize, Clone)]
struct WeightsData {
    // conv1_weights: Vec<Vec<Vec<Vec<i64>>>>,
    // conv1_bias: Vec<i64>,
    // conv1_strides: Vec<u32>,
    // conv1_kernel_shape: Vec<u32>,
    // conv1_group: Vec<u32>,
    // conv1_dilation: Vec<u32>,
    // conv1_pads: Vec<u32>,
    // conv1_input_shape: Vec<u32>,
    quantized: bool,
    scaling: u64,
    // conv2_weights: Vec<Vec<Vec<Vec<i64>>>>,
    // conv2_bias: Vec<i64>,
    // conv2_strides: Vec<u32>,
    // conv2_kernel_shape: Vec<u32>,
    // conv2_group: Vec<u32>,
    // conv2_dilation: Vec<u32>,
    // conv2_pads: Vec<u32>,
    // conv2_input_shape: Vec<u32>,
    // conv3_weights: Vec<Vec<Vec<Vec<i64>>>>,
    // conv3_bias: Vec<i64>,
    // conv3_strides: Vec<u32>,
    // conv3_kernel_shape: Vec<u32>,
    // conv3_group: Vec<u32>,
    // conv3_dilation: Vec<u32>,
    // conv3_pads: Vec<u32>,
    // conv3_input_shape: Vec<u32>,
    // fc1_alpha: u32,
    // fc1_beta: u32,
    // fc1_weights: Vec<Vec<i64>>,
    // fc1_bias: Vec<Vec<i64>>,
}
#[derive(Deserialize, Clone)]
struct WeightsData2 {
    // fc2_alpha: u32,
    // fc2_beta: u32,
    fc2_weights: Vec<Vec<i64>>,
    fc2_bias: Vec<Vec<i64>>,
}

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<i64>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<i64>>,
}

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/doom_weights.json");
const MATRIX_WEIGHTS_FILE2: &str = include_str!("../../../weights/doom_weights2.json");


//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}

lazy_static! {
    static ref WEIGHTS_INPUT2: WeightsData2 = {
        let x: WeightsData2 =
            serde_json::from_str(MATRIX_WEIGHTS_FILE2).expect("JSON was not well-formatted");
        x
    };
}

declare_circuit!(DoomCircuit {
    input_arr: [[Variable; 256]; 1], // shape (m, n)
    outputs: [[Variable; 7]; 1], // shape (m, k) 1, 16, 28, 28
});


// Memorization, in a better place
impl<C: Config> Define<C> for DoomCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n_bits = 32;

        let v_plus_one: usize = n_bits;
        let two_v: u32 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        // Bring the weights into the circuit as constants

        // if WEIGHTS_INPUT.fc1_alpha != 1 ||WEIGHTS_INPUT.fc1_beta != 1 || WEIGHTS_INPUT2.fc2_alpha != 1 || WEIGHTS_INPUT2.fc2_beta != 1{
        //     panic!("Not yet implemented for fc alpha or beta not equal to 1");
        // }

        // let input_arr = four_d_array_to_vec(self.input_arr);
        let input_arr = two_d_array_to_vec(self.input_arr);

        let weights = read_2d_weights(api, &WEIGHTS_INPUT2.fc2_weights);
        let bias = read_2d_weights(api, &WEIGHTS_INPUT2.fc2_bias);

        let out_2d = matrix_multplication_naive2(api, input_arr, weights);
        let out_2d = matrix_addition_vec(api, out_2d, bias);

        let out_2d = run_if_quantized_2d(api, WEIGHTS_INPUT.scaling, WEIGHTS_INPUT.quantized, out_2d, v_plus_one, two_v, alpha_2_v, true);

        for (j, dim1) in self.outputs.iter().enumerate() {
                for (k, _out) in dim1.iter().enumerate() {
                    api.assert_is_equal(self.outputs[j][k], out_2d[j][k]);
                }
            }
    }
}


impl<C: Config> IOReader<DoomCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: DoomCircuit<C::CircuitField>,
    ) -> DoomCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<DoomCircuit<_>, C>>::read_data_from_json::<
            InputData,
        >(file_path);

        // Assign inputs to assignment
        for (i, dim1) in data.input.iter().enumerate() {
            for (j, &element) in dim1.iter().enumerate() {
                if element < 0 {
                    assignment.input_arr[i][j] =
                        C::CircuitField::from(element.abs() as u32).neg();
                } else {
                    assignment.input_arr[i][j] =
                        C::CircuitField::from(element.abs() as u32);
                }
            }
        }
        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: DoomCircuit<C::CircuitField>,
    ) -> DoomCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<DoomCircuit<_>, C>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for (i, dim1) in data.output.iter().enumerate() {
            for (j, &element) in dim1.iter().enumerate() {
                if element < 0 {
                    assignment.outputs[i][j] =
                        C::CircuitField::from_u256(U256::from(element.abs() as u64)).neg();
                } else {
                    assignment.outputs[i][j] =
                        C::CircuitField::from_u256(U256::from(element.abs() as u64));
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
        path: "doom".to_owned(),
    };
    handle_args::<DoomCircuit<Variable>,DoomCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,_>(&mut file_reader);
}
