use arith::FieldForECC;
use gravy_circuits::circuit_functions::convolution_fn::conv_4d_run;
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{four_d_array_to_vec, load_circuit_constant, read_4d_weights};
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
    conv_weights: Vec<Vec<Vec<Vec<Vec<i64>>>>>,
    conv_bias: Vec<Vec<i64>>,
    conv_strides: Vec<Vec<u32>>,
    conv_kernel_shape: Vec<Vec<u32>>,
    conv_group: Vec<Vec<u32>>,
    conv_dilation: Vec<Vec<u32>>,
    conv_pads: Vec<Vec<u32>>,
    conv_input_shape: Vec<Vec<u32>>,
    scaling: u64,
    // fc_weights: Vec<Vec<Vec<i64>>>,
    // fc_bias: Vec<Vec<Vec<i64>>>,
}

#[derive(Deserialize, Clone)]
struct InputData {
    output: Vec<Vec<Vec<Vec<i64>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<Vec<Vec<i64>>>>,
}

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/doom_weights.json");


//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}

declare_circuit!(DoomCircuit {
    input_arr: [[[[Variable; 28]; 28]; 16]; 1], // shape (m, n)
    outputs: [[[[Variable; 14]; 14]; 32]; 1], // shape (m, k) 1, 16, 28, 28
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

        let weights = read_4d_weights(api, &WEIGHTS_INPUT.conv_weights[1]);
        let bias: Vec<Variable> = WEIGHTS_INPUT
            .conv_bias[1]
            .clone()
            .into_iter()
            .map(|x| load_circuit_constant(api, x))
            .collect();

        let input_arr = four_d_array_to_vec(self.input_arr);

        let out = conv_4d_run(api, input_arr, weights, bias,&WEIGHTS_INPUT.conv_dilation[1], &WEIGHTS_INPUT.conv_kernel_shape[1], &WEIGHTS_INPUT.conv_pads[1], &WEIGHTS_INPUT.conv_strides[1],&WEIGHTS_INPUT.conv_input_shape[1], WEIGHTS_INPUT.scaling, &WEIGHTS_INPUT.conv_group[1], true, v_plus_one, two_v, alpha_2_v, true);
        
        for (j, dim1) in self.outputs.iter().enumerate() {
                for (k, dim2) in dim1.iter().enumerate() {
                    for (l, dim3) in dim2.iter().enumerate() {
                        for (m, _dim4) in dim3.iter().enumerate() {
                            api.assert_is_equal(self.outputs[j][k][l][m], out[j][k][l][m]);
                        }
                    }
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
        for (i, dim1) in data.output.iter().enumerate() {
            for (j, dim2) in dim1.iter().enumerate() {
                for (k, dim3) in dim2.iter().enumerate() {
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            assignment.input_arr[i][j][k][l] =
                                C::CircuitField::from(element.abs() as u32).neg();
                        } else {
                            assignment.input_arr[i][j][k][l] =
                                C::CircuitField::from(element.abs() as u32);
                        }
                    }
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
            for (j, dim2) in dim1.iter().enumerate() {
                for (k, dim3) in dim2.iter().enumerate() {
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            assignment.outputs[i][j][k][l] =
                                C::CircuitField::from_u256(U256::from(element.abs() as u64)).neg();
                        } else {
                            assignment.outputs[i][j][k][l] =
                                C::CircuitField::from_u256(U256::from(element.abs() as u64));
                        }
                    }
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
    handle_args::<BN254Config, DoomCircuit<Variable>,DoomCircuit<_>,_>(&mut file_reader);
    // handle_args::<M31Config, DoomCircuit<Variable>,DoomCircuit<_>,_>(&mut file_reader);
    // handle_args::<GF2Config, DoomCircuit<Variable>,DoomCircuit<_>,_>(&mut file_reader);
}
