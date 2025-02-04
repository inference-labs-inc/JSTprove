use arith::FieldForECC;
use ethnum::U256;
use expander_compiler::frontend::*;
use helper_fn::{four_d_array_to_vec, load_circuit_constant};
use io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, two_d_array_to_vec,
};
use serde::Deserialize;
use std::ops::Neg;

#[path = "../src/matrix_computation.rs"]
pub mod matrix_computation;

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
}

#[derive(Deserialize, Clone)]
struct InputData {
    input_arr: Vec<Vec<Vec<Vec<i64>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    conv_out: Vec<Vec<Vec<Vec<i64>>>>,
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

fn read_4d_weights<C: Config>(
    api: &mut API<C>,
    weights_data: &Vec<Vec<Vec<Vec<i64>>>>,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let weights: Vec<Vec<Vec<Vec<Variable>>>> = weights_data
        .clone()
        .into_iter()
        .map(|dim1| {
            dim1.into_iter()
                .map(|dim2| {
                    dim2.into_iter()
                        .map(|dim3| {
                            dim3.into_iter()
                                .map(|x| load_circuit_constant(api, x))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    weights
}

//Untested
fn set_default_params(
    dilations: &Vec<u32>,
    kernel_shape: &Vec<u32>,
    pads: &Vec<u32>,
    strides: &Vec<u32>,
    input_shape: &Vec<u32>
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    // If dilations is empty, fill it with 1s of the appropriate length
    let mut dilations_out = dilations.clone();
    let mut kernel_shape_out = kernel_shape.clone();
    let mut pads_out = pads.clone();
    let mut strides_out = strides.clone();

    if dilations.is_empty() {
        dilations_out = vec![1; input_shape[2..].len()];
    }

    // If kernel_shape is empty, fill it with W.shape()[2..]
    if kernel_shape.is_empty() {
        kernel_shape_out = input_shape[2..].to_vec();
    }

    // If pads is empty, fill it with 0s, twice the length of X.shape()[2..]
    if pads.is_empty() {
        let shape_len = input_shape[2..].len();
        pads_out = vec![0; shape_len * 2];
    }

    // If strides is empty, fill it with 1s of the appropriate length
    if strides.is_empty() {
        strides_out = vec![1; input_shape[2..].len()];
    }
    (dilations_out, kernel_shape_out, pads_out, strides_out)
}

fn not_yet_implemented_conv(input_shape: &Vec<u32>, group: &Vec<u32>, dilations: &Vec<u32>, ){
    if input_shape[1] != input_shape[1] * group[0] || input_shape[0] % group[0] != 0 {
        panic!("Shape inconsistencies");
    }
    if group[0] > 1{
        panic!("Not yet implemented for group > 1");
    }
    if (dilations[0] != 1) || (dilations.iter().min() != dilations.iter().max()){
        panic!("Not yet implemented for this dilation");
    }
    if input_shape.len() == 3{
        panic!("Not yet implemented for Input shape length 3");
    }
    if input_shape.len() == 5{
        panic!("Not yet implemented for Input shape length 5");
    }
}

fn conv_shape_4<C: Config>(api: &mut API<C>, x: Vec<Vec<Vec<Vec<Variable>>>>, input_shape: &Vec<u32>, kernel_shape: &Vec<u32>, strides: &Vec<u32>, pads: &Vec<u32>, weights: &Vec<Vec<Vec<Vec<Variable>>>>, bias: &Vec<Variable>) -> Vec<Vec<Vec<Vec<Variable>>>>{
    if pads.len() < 4{
        panic!("Pads is not long enough");
    }
    let sN = input_shape.get(0).expect("Missing input shape index 0");
    let sC = input_shape.get(1).expect("Missing input shape index 1");
    let sH = input_shape.get(2).expect("Missing input shape index 2");
    let sW = input_shape.get(3).expect("Missing input shape index 3");

    // # M, C_group, kH, kW = W.shape
    let kh = kernel_shape.get(0).expect("Missing kernel shape index 0");
    let kw = kernel_shape.get(1).expect("Missing kernel shape index 1");

    let sth = strides.get(0).expect("Missing strides index 0");
    let stw = strides.get(1).expect("Missing strides index 1");

    //Need to make sure there is no overflow/casting issues here. Dont think there should be
    let h_out = (sH - kh + pads[0] + pads[2] / sth) + 1;
    let w_out = ((sW - kw + pads[1] + pads[3]) / stw) + 1;

    let h0 = pads.get(0).expect("Missing pads 0 index");
    let w0 = pads.get(1).expect("Missing pads 1 index");

    let oh = -1 * (kh % 2) as i32;
    let ow = -1 * (kw % 2) as i32;

    let bh = -(*h0 as i32);
    let bw = -(*w0 as i32);

    let eh = h_out * sth;
    let ew = w_out * stw;

    let mut res: Vec<Vec<Vec<Vec<Variable>>>> = Vec::with_capacity(input_shape[0] as usize);
    
    let (shape_0, shape_1, shape_2, shape_3) = (input_shape[0] as usize, weights.len(), h_out, w_out as usize);

    println!("{:?}", bias);
    if !bias.is_empty(){
        for _ in 0..shape_0 {
            let mut dim2 = Vec::with_capacity(shape_1);

            for j in 0..shape_1 {
                let mut dim3 = Vec::with_capacity(h_out as usize);
                for _ in 0..shape_2 {
                    dim3.push(vec![bias[j]; shape_3 as usize]); // Fill with zeros
                }
                dim2.push(dim3);
            }
            res.push(dim2);
        }
        // res[:, :, :, :] = B.reshape((1, -1, 1, 1))  # type: ignore
    }
    else{
        let zero = api.constant(0);
        for _ in 0..shape_0 {
            let mut dim2 = Vec::with_capacity(shape_1);
            for _ in 0..shape_1 {
                let mut dim3 = Vec::with_capacity(shape_2 as usize);
                for _ in 0..shape_2 {
                    dim3.push(vec![zero; shape_3 as usize]); // Fill with zeros
                }
                dim2.push(dim3);
            }
            res.push(dim2);
        }
    }
    

    x
}

declare_circuit!(ConvCircuit {
    input_arr: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, n)
    conv_out: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1],  // shape (m, k)
});
// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable> {
    fn define(&self, api: &mut API<C>) {
        // Bring the weights into the circuit as constants

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
            &WEIGHTS_INPUT.input_shape
        );
        not_yet_implemented_conv(&WEIGHTS_INPUT.input_shape, &WEIGHTS_INPUT.group, &dilations);

        let input_arr = four_d_array_to_vec(self.input_arr);

        let out = conv_shape_4(api, input_arr, &WEIGHTS_INPUT.input_shape, &kernel_shape, &strides, &pads, &weights, &bias);




        //Assert output of matrix multiplication
        for (j, dim1) in self.conv_out.iter().enumerate() {
            for (k, dim2) in dim1.iter().enumerate() {
                for (l, dim3) in dim2.iter().enumerate() {
                    for (m, dim4) in dim3.iter().enumerate() {
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
        for (i, dim1) in data.input_arr.iter().enumerate() {
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

        for (i, dim1) in data.conv_out.iter().enumerate() {
            for (j, dim2) in dim1.iter().enumerate() {
                for (k, dim3) in dim2.iter().enumerate() {
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            assignment.conv_out[i][j][k][l] =
                                C::CircuitField::from(element.abs() as u32).neg();
                        } else {
                            assignment.conv_out[i][j][k][l] =
                                C::CircuitField::from(element.abs() as u32);
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
    // run_gf2();
    // run_m31();
    main_runner::run_bn254::<ConvCircuit<Variable>,
    ConvCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
}
