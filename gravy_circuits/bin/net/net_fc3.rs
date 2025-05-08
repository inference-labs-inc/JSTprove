use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{read_2d_weights, two_d_array_to_vec};
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
    // conv_weights: Vec<Vec<Vec<Vec<Vec<i64>>>>>,
    // conv_bias: Vec<Vec<i64>>,
    // conv_strides: Vec<Vec<u32>>,
    // conv_kernel_shape: Vec<Vec<u32>>,
    // conv_group: Vec<Vec<u32>>,
    // conv_dilation: Vec<Vec<u32>>,
    // conv_pads: Vec<Vec<u32>>,
    // conv_input_shape: Vec<Vec<u32>>,
    // scaling: u64,
    fc_weights: Vec<Vec<Vec<i64>>>,
    fc_bias: Vec<Vec<Vec<i64>>>,
    // maxpool_kernel_size: Vec<Vec<usize>>,
    // maxpool_stride: Vec<Vec<usize>>,
    // maxpool_padding: Vec<Vec<usize>>,
    // maxpool_dilation: Vec<Vec<usize>>,
    // maxpool_input_shape: Vec<Vec<usize>>,
    // return_indeces: bool,
    // maxpool_ceil_mode: Vec<bool>,

    // layers: Vec<String>
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
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/net_fc3_weights.json");


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
    input_arr: [[PublicVariable; 84]; 1], // shape (m, n)
    outputs: [[PublicVariable; 10]; 1], // shape (m, k)
    dummy: [Variable; 2]
});



// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {

        // Bring the weights into the circuit as constants
        let mut out_2d = two_d_array_to_vec(self.input_arr);        

        let i = 0;

        let weights = read_2d_weights(api, &WEIGHTS_INPUT.fc_weights[i]);
        let bias = read_2d_weights(api, &WEIGHTS_INPUT.fc_bias[i]);

        out_2d = matrix_multplication_naive2(api, out_2d, weights);
        out_2d = matrix_addition_vec(api, out_2d, bias);
        api.display("3", out_2d[0][0]);

        for (j, dim1) in self.outputs.iter().enumerate() {
                for (k, _dim2) in dim1.iter().enumerate() {
                    api.display("out1", self.outputs[j][k]);
                    api.display("out2", out_2d[j][k]);
                    api.assert_is_equal(self.outputs[j][k], out_2d[j][k]);

                }
            }
            api.assert_is_equal(self.dummy[0],1);
            api.assert_is_equal(self.dummy[1],1);


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
            for (j, &element) in dim1.iter().enumerate() {
                if element < 0 {
                    assignment.input_arr[i][j] =
                        CircuitField::<C>::from_u256(U256::from(element.abs() as u64)).neg();
                } else {
                    assignment.input_arr[i][j] =
                        CircuitField::<C>::from_u256(U256::from(element.abs() as u64));
                }
            }
        }
        assignment.dummy[0] = CircuitField::<C>::from(1);
        assignment.dummy[1] = CircuitField::<C>::from(1);


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
        path: "net".to_owned(),
    };
    // println!("{:?}", WEIGHTS_INPUT.layers);
    handle_args::<BN254Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);


}
