use gravy_circuits::circuit_functions::convolution_fn::conv_4d_run;
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{four_d_array_to_vec, load_circuit_constant, read_2d_weights, read_4d_weights};
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, matrix_addition_vec
};
use gravy_circuits::circuit_functions::quantization::run_if_quantized_2d;
// use relu::{relu_2d_vec_v2, relu_4d_vec_v2};
use serde::Deserialize;
use std::ops::Neg;

use gravy_circuits::runner::main_runner::handle_args;


//Define structure of inputs, weights and output
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
    fc_weights: Vec<Vec<Vec<i64>>>,
    fc_bias: Vec<Vec<Vec<i64>>>,
}

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<Vec<Vec<i64>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<i64>>,
}


// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/doom_weights.json");


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
    input_arr: [[[[Variable; 28]; 28]; 4]; 1], // shape (m, n)
    outputs: [[Variable; 7]; 1], // shape (m, k)
});

// declare_circuit!(DoomFakeCircuit {
//     input_arr: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, n)
//     outputs: [[Variable; 7]; 1], // shape (m, k)
// });

// impl<C: Config> Define<C> for DoomFakeCircuit<Variable> {
//     fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
//         api.assert_is_equal(0, self.outputs[0][0]);
//     }
// }

// Memorization, in a better place
impl<C: Config> Define<C> for DoomCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n_bits = 32;

        let v_plus_one: usize = n_bits;
        let two_v: u64 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, CircuitField::<C>::from_u256(U256::from(two_v)));

        // Bring the weights into the circuit as constants

        // if WEIGHTS_INPUT.fc1_alpha != 1 ||WEIGHTS_INPUT.fc1_beta != 1 || WEIGHTS_INPUT2.fc2_alpha != 1 || WEIGHTS_INPUT2.fc2_beta != 1{
        //     panic!("Not yet implemented for fc alpha or beta not equal to 1");
        // }
        let mut out = four_d_array_to_vec(self.input_arr);
        // api.display("1", input_arr[0][0][0][0]);
        
        // Conv 1
        for (i, _) in WEIGHTS_INPUT.conv_weights.iter().enumerate(){
            let weights = read_4d_weights(api, &WEIGHTS_INPUT.conv_weights[i]);
            let bias: Vec<Variable> = WEIGHTS_INPUT
                .conv_bias[i]
                .clone()
                .into_iter()
                .map(|x| load_circuit_constant(api, x))
                .collect();

            
            out = conv_4d_run(api, out, weights, bias,&WEIGHTS_INPUT.conv_dilation[i], &WEIGHTS_INPUT.conv_kernel_shape[i], &WEIGHTS_INPUT.conv_pads[i], &WEIGHTS_INPUT.conv_strides[i],&WEIGHTS_INPUT.conv_input_shape[i], WEIGHTS_INPUT.scaling, &WEIGHTS_INPUT.conv_group[i], true, v_plus_one, two_v, alpha_2_v, true);
            api.display("2", out[0][0][0][0]);
        }

        //Reshape
        let out_1d: Vec<Variable> = out.iter()
                .flat_map(|x| x.iter())
                .flat_map(|x| x.iter())
                .flat_map(|x| x.iter())
                .copied()
                .collect();

        let mut out_2d = vec![out_1d];
        for (i, _) in WEIGHTS_INPUT.fc_weights.iter().enumerate(){
            // if WEIGHTS_INPUT2.fc_alpha[i] != 1 ||WEIGHTS_INPUT2.fc_beta[i] != 1 {
            //     panic!("Not yet implemented for fc alpha or beta not equal to 1");
            // }
            let weights = read_2d_weights(api, &WEIGHTS_INPUT.fc_weights[i]);
            let bias = read_2d_weights(api, &WEIGHTS_INPUT.fc_bias[i]);

            out_2d = matrix_multplication_naive2(api, out_2d, weights);
            out_2d = matrix_addition_vec(api, out_2d, bias);
            api.display("3", out_2d[0][0]);

            if i != WEIGHTS_INPUT.fc_weights.len() - 1{
                out_2d = run_if_quantized_2d(api, WEIGHTS_INPUT.scaling, true, out_2d, v_plus_one, two_v, alpha_2_v, true);
            }
            api.display("4", out_2d[0][0]);

        }

        for (j, dim1) in self.outputs.iter().enumerate() {
                for (k, _dim2) in dim1.iter().enumerate() {
                    api.assert_is_equal(self.outputs[j][k], out_2d[j][k]);
                }
            }
    }
}


impl<C: Config> IOReader<DoomCircuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: DoomCircuit<CircuitField::<C>>,
    ) -> DoomCircuit<CircuitField::<C>> {
        let data: InputData = <FileReader as IOReader<DoomCircuit<_>, C>>::read_data_from_json::<
            InputData,
        >(file_path);

        // Assign inputs to assignment
        for (i, dim1) in data.input.iter().enumerate() {
            for (j, dim2) in dim1.iter().enumerate() {
                for (k, dim3) in dim2.iter().enumerate() {
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            assignment.input_arr[i][j][k][l] =
                                CircuitField::<C>::from(element.abs() as u32).neg();
                        } else {
                            assignment.input_arr[i][j][k][l] =
                                CircuitField::<C>::from(element.abs() as u32);
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
        mut assignment: DoomCircuit<CircuitField::<C>>,
    ) -> DoomCircuit<CircuitField::<C>> {
        let data: OutputData = <FileReader as IOReader<DoomCircuit<_>, C>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for (i, dim1) in data.output.iter().enumerate() {
            for (j, &element) in dim1.iter().enumerate() {
                if element < 0 {
                    assignment.outputs[i][j] =
                        CircuitField::<C>::from_u256(U256::from(element.abs() as u64)).neg();
                } else {
                    assignment.outputs[i][j] =
                        CircuitField::<C>::from_u256(U256::from(element.abs() as u64));
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
