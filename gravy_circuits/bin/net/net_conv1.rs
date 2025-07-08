use gravy_circuits::circuit_functions::convolution_fn::conv_4d_run;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{four_d_array_to_vec, load_circuit_constant, read_4d_weights};
use gravy_circuits::circuit_functions::pooling::{setup_maxpooling_2d, maxpooling_2d};
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, matrix_addition_vec
};
use serde::Deserialize;
use std::ops::Neg;

use gravy_circuits::runner::main_runner::handle_args;


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
    maxpool_kernel_size: Vec<Vec<usize>>,
    maxpool_stride: Vec<Vec<usize>>,
    maxpool_padding: Vec<Vec<usize>>,
    maxpool_dilation: Vec<Vec<usize>>,
    maxpool_input_shape: Vec<Vec<usize>>,
    // return_indeces: bool,
    maxpool_ceil_mode: Vec<bool>,

    // layers: Vec<String>
}

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<Vec<Vec<i64>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<Vec<Vec<i64>>>>,
}

const BASE: u32 = 2;
const NUM_DIGITS: usize = 32; 

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/net_conv1_weights.json");


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
    input_arr: [[[[PublicVariable; 32]; 32]; 3]; 1], // shape (m, n)
    outputs: [[[[PublicVariable; 14]; 14]; 6]; 1], // shape (m, k)
});



// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n_bits = 32;

        let v_plus_one: usize = n_bits;
        let two_v: u32 = 1 << (v_plus_one - 1);
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        // Bring the weights into the circuit as constants
        let mut out = four_d_array_to_vec(self.input_arr);        
        // Conv 1
        let i = 0;
        let weights: Vec<Vec<Vec<Vec<Variable>>>> = read_4d_weights(api, &WEIGHTS_INPUT.conv_weights[i]);
        let bias: Vec<Variable> = WEIGHTS_INPUT
            .conv_bias[i]
            .clone()
            .into_iter()
            .map(|x| load_circuit_constant(api, x))
            .collect();

        
        out = conv_4d_run(api, out, weights, bias,&WEIGHTS_INPUT.conv_dilation[i], &WEIGHTS_INPUT.conv_kernel_shape[i], &WEIGHTS_INPUT.conv_pads[i], &WEIGHTS_INPUT.conv_strides[i],&WEIGHTS_INPUT.conv_input_shape[i], WEIGHTS_INPUT.scaling, &WEIGHTS_INPUT.conv_group[i], true, v_plus_one, two_v, alpha_2_v, true);
        api.display("2", out[0][0][0][0]);

        let (kernel_shape, strides, dilation, output_spatial_shape, new_pads) = setup_maxpooling_2d(&WEIGHTS_INPUT.maxpool_padding[i], &WEIGHTS_INPUT.maxpool_kernel_size[i], &WEIGHTS_INPUT.maxpool_stride[0], &WEIGHTS_INPUT.maxpool_dilation[i], WEIGHTS_INPUT.maxpool_ceil_mode[i], &WEIGHTS_INPUT.maxpool_input_shape[i]);

        // let mut table = LogUpRangeProofTable::new(nb_bits);
        // table.initial(api);
        let mut table_opt = None;

        out = maxpooling_2d(api, &out, &kernel_shape, &strides, &dilation, &output_spatial_shape, &WEIGHTS_INPUT.maxpool_input_shape[i], &new_pads, BASE, NUM_DIGITS, false, &mut table_opt);
        // panic!("{}, {}, {}, {}", out[0][0][0].len(), out[0][0].len(), out[0].len(), out.len());
        for (j, dim1) in self.outputs.iter().enumerate() {
                for (k, _dim2) in dim1.iter().enumerate() {
                    for (l, dim3) in _dim2.iter().enumerate() {
                        for (m, _dim4) in dim3.iter().enumerate() {
                            // api.display("outputs", self.outputs[j][k][l][m]);
                            // api.display("calculated", out[j][k][l][m]);

                            api.assert_is_equal(self.outputs[j][k][l][m], out[j][k][l][m]);
                            // api.assert_is_different(self.outputs[j][k][l][m], 1);

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
                            assignment.outputs[i][j][k][l] =
                                CircuitField::<C>::from(element.abs() as u32).neg();
                        } else {
                            assignment.outputs[i][j][k][l] =
                                CircuitField::<C>::from(element.abs() as u32);
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
        path: "net".to_owned(),
    };
    // println!("{:?}", WEIGHTS_INPUT.layers);
    handle_args::<BN254Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);


}
