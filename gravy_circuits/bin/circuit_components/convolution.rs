use gravy_circuits::circuit_functions::convolution_fn::{conv_shape_4, not_yet_implemented_conv, set_default_params};
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{four_d_array_to_vec, load_circuit_constant, read_4d_weights};
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array,
};
use gravy_circuits::circuit_functions::quantization::quantize_4d_vector;
use serde::Deserialize;
use std::ops::Neg;
use gravy_circuits::runner::main_runner::handle_args;


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

const DIM2OUT: usize = 16;

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
    quantized: bool,
    scaling: u64,
    is_relu: bool
}

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<Vec<Vec<i64>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<Vec<Vec<i64>>>>,
}

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/convolution_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}

declare_circuit!(ConvCircuit {
    input_arr: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, n)
    conv_out: [[[[Variable; DIM4]; DIM3]; DIM2OUT]; DIM1], // shape (m, k)
});

// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Bring the weights into the circuit as constants

        let v_plus_one: usize = 32;
        let two_v: u32 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        let weights = read_4d_weights(api, &WEIGHTS_INPUT.conv_weights[0]);
        let bias: Vec<Variable> = WEIGHTS_INPUT
            .conv_bias[0]
            .clone()
            .into_iter()
            .map(|x| load_circuit_constant(api, x))
            .collect();
        let (dilations, kernel_shape, pads, strides) = set_default_params(
            &WEIGHTS_INPUT.conv_dilation[0],
            &WEIGHTS_INPUT.conv_kernel_shape[0],
            &WEIGHTS_INPUT.conv_pads[0],
            &WEIGHTS_INPUT.conv_strides[0],
            &WEIGHTS_INPUT.conv_input_shape[0],
        );
        not_yet_implemented_conv(&WEIGHTS_INPUT.conv_input_shape[0], &WEIGHTS_INPUT.conv_group[0], &dilations);

        let input_arr = four_d_array_to_vec(self.input_arr);

        let mut out: Vec<Vec<Vec<Vec<Variable>>>> = conv_shape_4(
            api,
            input_arr,
            &WEIGHTS_INPUT.conv_input_shape[0],
            &kernel_shape,
            &strides,
            &pads,
            &weights,
            &bias,
        );

        if WEIGHTS_INPUT.quantized{
            let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
            println!("{}", scaling_factor);
            out = quantize_4d_vector(api, out, scaling_factor, WEIGHTS_INPUT.scaling as usize, v_plus_one, two_v, alpha_2_v, WEIGHTS_INPUT.is_relu);
            // panic!("Quantized not yet implemented");
        }
        else{
            out = out;
        }

        //Assert output of matrix multiplication
        for (j, dim1) in self.conv_out.iter().enumerate() {
            for (k, dim2) in dim1.iter().enumerate() {
                for (l, dim3) in dim2.iter().enumerate() {
                    for (m, _) in dim3.iter().enumerate() {
                        api.assert_is_equal(self.conv_out[j][k][l][m], out[j][k][l][m]);
                    }
                }
            }
        }
    }
}

impl<C: Config> IOReader<ConvCircuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: ConvCircuit<CircuitField::<C>>,
    ) -> ConvCircuit<CircuitField::<C>> {
        let data: InputData = <FileReader as IOReader<ConvCircuit<_>, C>>::read_data_from_json::<
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
        mut assignment: ConvCircuit<CircuitField::<C>>,
    ) -> ConvCircuit<CircuitField::<C>> {
        let data: OutputData = <FileReader as IOReader<ConvCircuit<_>, C>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for (i, dim1) in data.output.iter().enumerate() {
            for (j, dim2) in dim1.iter().enumerate() {
                for (k, dim3) in dim2.iter().enumerate() {
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            assignment.conv_out[i][j][k][l] =
                                CircuitField::<C>::from_u256(U256::from(element.abs() as u64)).neg();
                        } else {
                            assignment.conv_out[i][j][k][l] =
                                CircuitField::<C>::from_u256(U256::from(element.abs() as u64));
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
        path: "convolution".to_owned(),
    };
    handle_args::<BN254Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);
    // handle_args::<M31Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);
    // handle_args::<GF2Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);
}
