use arith::FieldForECC;
use convolution_fn::{conv_shape_4, not_yet_implemented_conv, set_default_params};
use ethnum::U256;
use expander_compiler::frontend::*;
use helper_fn::{four_d_array_to_vec, load_circuit_constant, read_4d_weights};
use io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, two_d_array_to_vec,
};
use quantization::quantize_4d_vector;
use serde::Deserialize;
use std::ops::Neg;

#[path = "../src/convolution_fn.rs"]
pub mod convolution_fn;
#[path = "../src/matrix_computation.rs"]
pub mod matrix_computation;

#[path = "../src/quantization.rs"]
pub mod quantization;

#[path = "../src/helper_fn.rs"]
pub mod helper_fn;
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

const DIM1: usize = 1; // m
const DIM2: usize = 4; // n
const DIM3: usize = 28; // n
const DIM4: usize = 28; // k

const DIM2OUT: usize = 16;

//Define structure of inputs, weights and output
#[derive(Deserialize, Clone)]
struct WeightsData {
    weights: Vec<Vec<Vec<Vec<i64>>>>,
    bias: Vec<i64>,
    strides: Vec<u32>,
    kernel_shape: Vec<u32>,
    group: Vec<u32>,
    dilation: Vec<u32>,
    pads: Vec<u32>,
    input_shape: Vec<u32>,
    quantized: bool,
    scaling: u64,
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
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/convolution_weights.json");

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
impl<C: Config> GenericDefine<C> for ConvCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Bring the weights into the circuit as constants

        let v_plus_one: usize = 32;
        let two_v: u32 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        let weights = read_4d_weights(api, &WEIGHTS_INPUT.weights);
        let bias: Vec<Variable> = WEIGHTS_INPUT
            .bias
            .clone()
            .into_iter()
            .map(|x| load_circuit_constant(api, x))
            .collect();
        let (dilations, kernel_shape, pads, strides) = set_default_params(
            &WEIGHTS_INPUT.dilation,
            &WEIGHTS_INPUT.kernel_shape,
            &WEIGHTS_INPUT.pads,
            &WEIGHTS_INPUT.strides,
            &WEIGHTS_INPUT.input_shape,
        );
        not_yet_implemented_conv(&WEIGHTS_INPUT.input_shape, &WEIGHTS_INPUT.group, &dilations);

        let input_arr = four_d_array_to_vec(self.input_arr);

        let mut out: Vec<Vec<Vec<Vec<Variable>>>> = conv_shape_4(
            api,
            input_arr,
            &WEIGHTS_INPUT.input_shape,
            &kernel_shape,
            &strides,
            &pads,
            &weights,
            &bias,
        );

        if WEIGHTS_INPUT.quantized{
            let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
            println!("{}", scaling_factor);
            out = quantize_4d_vector(api, out, scaling_factor, WEIGHTS_INPUT.scaling as usize, v_plus_one, two_v, alpha_2_v);
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

impl<C: Config> IOReader<C, ConvCircuit<C::CircuitField>> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: ConvCircuit<C::CircuitField>,
    ) -> ConvCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<C, ConvCircuit<_>>>::read_data_from_json::<
            InputData,
        >(file_path);

        // Assign inputs to assignment
        for (i, dim1) in data.input.iter().enumerate() {
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
        mut assignment: ConvCircuit<C::CircuitField>,
    ) -> ConvCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<C, ConvCircuit<_>>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for (i, dim1) in data.output.iter().enumerate() {
            for (j, dim2) in dim1.iter().enumerate() {
                for (k, dim3) in dim2.iter().enumerate() {
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            assignment.conv_out[i][j][k][l] =
                                C::CircuitField::from_u256(U256::from(element.abs() as u64)).neg();
                        } else {
                            assignment.conv_out[i][j][k][l] =
                                C::CircuitField::from_u256(U256::from(element.abs() as u64));
                        }
                    }
                }
            }
        }
        // Return the assignment
        assignment
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: String::new(),
    };
    main_runner::run_bn254::<ConvCircuit<Variable>,
                            ConvCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);

    // main_runner::debug_bn254::<ConvCircuit<Variable>,
    //                         ConvCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
    //                                                 _>(&mut file_reader);
}
