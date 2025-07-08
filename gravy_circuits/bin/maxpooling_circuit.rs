use std::ops::Neg;

use expander_compiler::frontend::*;
use circuit_std_rs::logup::LogUpRangeProofTable;
use gravy_circuits::circuit_functions::helper_fn::four_d_array_to_vec;
use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::circuit_functions::pooling::{maxpooling_2d, setup_maxpooling_2d};
use serde::Deserialize;
use lazy_static::lazy_static;


const BASE: u32 = 2;
const NUM_DIGITS: usize = 32; 

const DIM1: usize = 1;
const DIM2: usize = 4;
const DIM3: usize = 28;
const DIM4: usize = 28;

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<Vec<Vec<i32>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<Vec<Vec<i32>>>>,
}
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/maxpooling_weights.json");

#[derive(Deserialize, Clone)]
struct WeightsData {
    maxpool_kernel_size: Vec<Vec<usize>>,
    maxpool_stride: Vec<Vec<usize>>,
    maxpool_padding: Vec<Vec<usize>>,
    maxpool_dilation: Vec<Vec<usize>>,
    maxpool_input_shape: Vec<Vec<usize>>,
    // return_indeces: bool,
    maxpool_ceil_mode: Vec<bool>,
}

lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}


declare_circuit!(MaxPoolCircuit {
    input_arr: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, n)
    outputs: [[[[Variable; DIM4/2]; DIM3/2]; DIM2]; DIM1], // shape (m, k)
});

impl<C: Config> Define<C> for MaxPoolCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Create a shared lookup table for digit range checks
        let nb_bits = (32 - BASE.leading_zeros()) as usize;
        let mut table = LogUpRangeProofTable::new(nb_bits);
        table.initial(api);
        let mut _table_opt = Some(&mut table);
        let inputs = four_d_array_to_vec(self.input_arr.clone());

        let (kernel_shape, strides, dilation, output_spatial_shape, new_pads) = setup_maxpooling_2d(&WEIGHTS_INPUT.maxpool_padding[0], &WEIGHTS_INPUT.maxpool_kernel_size[0], &WEIGHTS_INPUT.maxpool_stride[0], &WEIGHTS_INPUT.maxpool_dilation[0], WEIGHTS_INPUT.maxpool_ceil_mode[0], &WEIGHTS_INPUT.maxpool_input_shape[0]);

        let mut table = LogUpRangeProofTable::new(nb_bits);
        table.initial(api);
        let mut table_opt = Some(&mut table);

        let out = maxpooling_2d(api, &inputs, &kernel_shape, &strides, &dilation, &output_spatial_shape, &WEIGHTS_INPUT.maxpool_input_shape[0], &new_pads, BASE, NUM_DIGITS, false, &mut table_opt);

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





impl<C: Config> IOReader<MaxPoolCircuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: MaxPoolCircuit<CircuitField::<C>>,
    ) -> MaxPoolCircuit<CircuitField::<C>> {
        let data: InputData = <FileReader as IOReader<MaxPoolCircuit<CircuitField::<C>>, C>>::read_data_from_json::<InputData>(file_path);

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
    
        assignment
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: MaxPoolCircuit<CircuitField::<C>>,
    ) -> MaxPoolCircuit<CircuitField::<C>> {
        let data: OutputData = <FileReader as IOReader<MaxPoolCircuit<CircuitField::<C>>, C>>::read_data_from_json::<OutputData>(file_path);
    
        // assert_eq!(data.max_val.len(), BATCH_SIZE, "Expected {} outputs", BATCH_SIZE);
    
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
    
        assignment
    }
    

    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "extrema".to_owned(),
    };
    // handle_args::<M31Config, ExtremaCircuit<Variable>,ExtremaCircuit<_>,_>(&mut file_reader);
    // handle_args::<BN254Config, ExtremaCircuit<Variable>,ExtremaCircuit<_>,_>(&mut file_reader);
    handle_args::<BN254Config, MaxPoolCircuit<Variable>,MaxPoolCircuit<_>,_>(&mut file_reader);

    
}