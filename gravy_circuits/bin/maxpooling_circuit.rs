use std::ops::Neg;

use expander_compiler::frontend::*;
use circuit_std_rs::logup::LogUpRangeProofTable;
use gravy_circuits::circuit_functions::helper_fn::four_d_array_to_vec;
use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::circuit_functions::extrema::assert_extremum;
use serde::Deserialize;
use lazy_static::lazy_static;


const BASE: u32 = 2;
const NUM_DIGITS: usize = 32; 

const DIM1: usize = 1;
const DIM2: usize = 4;
const DIM3: usize = 28;
const DIM4: usize = 28;



declare_circuit!(MaxPoolCircuit {
    input_arr: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, n)
    outputs: [[[[Variable; DIM4]; DIM3]; DIM2]; DIM1], // shape (m, k)
});



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
    kernel_size: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    input_shape: Vec<usize>,
    return_indeces: bool,
    ceil_mode: bool,
}

lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}

pub fn setup_maxpooling_2d_params(
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,

) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let padding = match padding.len() {
        0 => vec![0; 4],
        1 => vec![padding[0]; 4],
        2 => vec![padding[0], padding[1], padding[0], padding[1]],
        3 => panic!("Padding cannot have 3 elements"),
        4 => padding.clone(),
        _ => panic!("Padding must have between 1 and 4 elements"),
    };

    let kernel_shape  = match kernel_shape.len() {
        1 => vec![kernel_shape[0]; 2],
        2 => vec![kernel_shape[0], kernel_shape[1]],
        _ => panic!("Kernel shape must have between 1 and 2 elements"),
    };

    let dilation = match dilation.len() {
        0 => vec![1; 2],
        1 => vec![dilation[0]; 2],
        2 => vec![dilation[0], dilation[1]],
        _ => panic!("Dilation must have between 1 and 2 elements"),
    };

    let strides = match strides.len() {
        0 => vec![1; 2],
        1 => vec![strides[0]; 2],
        2 => vec![strides[0], strides[1]],
        _ => panic!("Strides must have between 1 and 2 elements"),
    };
    return (padding, kernel_shape, strides, dilation);
}

pub fn setup_maxpooling_2d(
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
    ceil_mode: bool,
    x_shape: &Vec<usize>,


)-> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<[usize; 2]>) {
    let (pads, kernel_shape, strides, dilation) = setup_maxpooling_2d_params(&padding, &kernel_shape, &strides, &dilation);
    
    let n_dims = kernel_shape.len();

    // Create new_pads as Vec<[usize; 2]>
    let mut new_pads = vec![[0; 2]; n_dims];
    for i in 0..n_dims {
        new_pads[i] = [pads[i], pads[i + n_dims]];
    }


    let input_spatial_shape = &x_shape[2..];
    let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

    for i in 0..input_spatial_shape.len() {
        let total_padding = new_pads[i][0] + new_pads[i][1];
        let kernel_extent = (kernel_shape[i] - 1) * dilation[i] + 1;
        let numerator = input_spatial_shape[i] + total_padding - kernel_extent;

        if ceil_mode {
            let mut out_dim = (numerator as f64 / strides[i] as f64).ceil() as usize + 1;
            let need_to_reduce =
                (out_dim - 1) * strides[i] >= input_spatial_shape[i] + new_pads[i][0];
            if need_to_reduce {
                out_dim -= 1;
            }
            output_spatial_shape[i] = out_dim;
        } else {
            output_spatial_shape[i] = (numerator / strides[i]) + 1;
        }
    }
    return (pads, kernel_shape, strides, dilation, output_spatial_shape, new_pads);
}

pub fn maximum<C: Config, Builder: RootAPI<C>>( 
    api: &mut Builder,
    input: Vec<Variable>,
) -> Variable{
    //Register hint to get maximum, and then assert its true using Tristans extrema code
    input[0]
}

pub fn maxpooling_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Vec<Vec<Vec<Vec<Variable>>>>,
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
    output_spatial_shape: &Vec<usize>,
    x_shape: &Vec<usize>,
    new_pads: &Vec<[usize; 2]>,
)-> Vec<Vec<Vec<Vec<Variable>>>>{
    let global_pooling = false;
    let batch = x_shape[0];
    let channels = x_shape[1];
    let height = x_shape[2];
    let width = if kernel_shape.len() > 1 { x_shape[3] } else { 1 };

    let pooled_height = output_spatial_shape[0];
    let pooled_width = if kernel_shape.len() > 1 {
        output_spatial_shape[1]
    } else {
        1
    };

    let y_dims = [batch, channels, pooled_height, pooled_width];
    let y_size = y_dims.iter().product();
    let y = vec![api.constant(0); y_size];

    let total_channels = batch * channels;

    let stride_h = if global_pooling { 1 } else { strides[0] };
    let stride_w = if global_pooling {
        1
    } else if strides.len() > 1 {
        strides[1]
    } else {
        1
    };

    let x_step = height * width;
    let y_step = pooled_height * pooled_width;

    let dilation_h = dilation[0];
    let dilation_w = if dilation.len() > 1 {
        dilation[1]
    } else {
        1
    };
    let x_data: Vec<Variable> =  x.iter()
    .flat_map(|z| z.iter())
    .flat_map(|z| z.iter())
    .flat_map(|z| z.iter())
    .copied()
    .collect();
    let mut y_data = y;

    for c in 0..total_channels {
        let x_d = c * x_step;
        let y_d = c * y_step;

        for ph in 0..pooled_height {
            let hstart = ph as isize * stride_h as isize - new_pads[0][0] as isize;
            let hend = hstart + (kernel_shape[0] * dilation_h) as isize;

            for pw in 0..pooled_width {
                let wstart = pw as isize * stride_w as isize - new_pads[1][0] as isize;
                let wend = wstart + (kernel_shape[1] * dilation_w) as isize;

                let pool_index = ph * pooled_width + pw;
                let mut max_elements: Vec<Variable> = Vec::new();

                for h in (hstart..hend).step_by(dilation_h) {
                    if h < 0 || h >= height as isize {
                        continue;
                    }
                    for w in (wstart..wend).step_by(dilation_w) {
                        if w < 0 || w >= width as isize {
                            continue;
                        }

                        let input_index = (h as usize) * width + (w as usize);
                        let val = x_data[x_d + input_index];
                        max_elements.push(val);

                    }
                }
                if max_elements.len() != 0 {
                    y_data[y_d + pool_index] = maximum(api, max_elements);
                }
            }
        }
    }

    reshape_4d(&y_data, y_dims)
}

fn reshape_4d(flat: &Vec<Variable>, dims: [usize; 4]) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let mut out =  vec![vec![vec![vec![Variable::default(); w]; h]; c]; n];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let flat_index = ((ni * c + ci) * h + hi) * w + wi;
                    out[ni][ci][hi][wi] = flat[flat_index];
                }
            }
        }
    }
    out
}


impl<C: Config> Define<C> for MaxPoolCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Create a shared lookup table for digit range checks
        let nb_bits = (32 - BASE.leading_zeros()) as usize;
        let mut table = LogUpRangeProofTable::new(nb_bits);
        table.initial(api);
        let mut table_opt = Some(&mut table);


        let inputs = four_d_array_to_vec(self.input_arr.clone());

        
        let (padding, kernel_shape, strides, dilation, output_spatial_shape, new_pads) = setup_maxpooling_2d(&WEIGHTS_INPUT.padding, &WEIGHTS_INPUT.kernel_size, &WEIGHTS_INPUT.stride, &WEIGHTS_INPUT.dilation, WEIGHTS_INPUT.ceil_mode, &WEIGHTS_INPUT.input_shape);

        let out = maxpooling_2d(api, &inputs, &padding, &kernel_shape, &strides, &dilation, &output_spatial_shape, &WEIGHTS_INPUT.input_shape, &new_pads);



        


        for (j, dim1) in self.outputs.iter().enumerate() {
            for (k, dim2) in dim1.iter().enumerate() {
                for (l, dim3) in dim2.iter().enumerate() {
                    for (m, _dim4) in dim3.iter().enumerate() {
                        api.assert_is_different(self.outputs[j][k][l][m], 1);
                        // api.assert_is_equal(, val);
                    }
                }
            }
        }

    //     for i in 0..BATCH_SIZE {
    //         let max = self.max_val[i];
    //         let candidates = &self.input_vec[i];
    //         let is_max = true;
    //         let use_lookup = false; 
    //         // let use_lookup = true;
    //         assert_extremum(
    //             api,
    //             max,
    //             candidates,
    //             BASE,
    //             NUM_DIGITS,
    //             is_max,
    //             use_lookup,
    //             &mut table_opt,
    //         );
    //     }
    }
}





impl<C: Config> IOReader<MaxPoolCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: MaxPoolCircuit<C::CircuitField>,
    ) -> MaxPoolCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<MaxPoolCircuit<C::CircuitField>, C>>::read_data_from_json::<InputData>(file_path);

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
    
        assignment
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: MaxPoolCircuit<C::CircuitField>,
    ) -> MaxPoolCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<MaxPoolCircuit<C::CircuitField>, C>>::read_data_from_json::<OutputData>(file_path);
    
        // assert_eq!(data.max_val.len(), BATCH_SIZE, "Expected {} outputs", BATCH_SIZE);
    
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
    handle_args::<MaxPoolCircuit<Variable>, MaxPoolCircuit<_>, _>(&mut file_reader);
    
}