use gravy_circuits::circuit_functions::convolution_fn::conv_4d_run;
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{arrayd_to_vec2, arrayd_to_vec4, four_d_array_to_vec, load_circuit_constant, read_2d_weights, read_4d_weights, vec2_to_arrayd, vec4_to_arrayd};
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use lazy_static::lazy_static;
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_multplication, matrix_multplication_array, matrix_multplication_naive,
    matrix_multplication_naive2, matrix_multplication_naive2_array, matrix_multplication_naive3,
    matrix_multplication_naive3_array, matrix_addition_vec
};
use gravy_circuits::circuit_functions::quantization::run_if_quantized_2d;
use serde::Deserialize;
use core::panic;
use std::ops::Neg;
use ndarray::IxDyn;


use gravy_circuits::runner::main_runner::{handle_args, ConfigurableCircuit};


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
    layers: Vec<String>,
    output_shape: Vec<usize>,
    input_shape: Vec<usize>,
    not_rescale_layers: Vec<String>,
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
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/generic_demo_weights.json");


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
    input_arr: [[[[PublicVariable]]]], // shape (m, n)
    outputs: [[PublicVariable]], // shape (m, k)
    // dummy: [PublicVariable; 10]
});



// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n_bits = 32;

        let v_plus_one: usize = n_bits;
        let two_v: u32 = 1 << (v_plus_one - 1);
        // TODO adjust for different base
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        // Bring the weights into the circuit as constants
        // let mut out: Vec<Vec<Vec<Vec<Variable>>>> = self.input_arr.to_vec();
        let mut out = vec4_to_arrayd(self.input_arr.clone());


        let mut layer_num = 0;
        let mut conv_layer_num = 0;
        let mut fc_layer_num = 0;

        while layer_num < WEIGHTS_INPUT.layers.len(){
            let layer = &WEIGHTS_INPUT.layers[layer_num];

            let mut is_rescale = true;
            for l in &WEIGHTS_INPUT.not_rescale_layers{
                if l.eq_ignore_ascii_case(layer){
                    is_rescale = false;
                }
            }

            let mut is_relu = false;
                if layer_num + 1 < WEIGHTS_INPUT.layers.len(){
                    let layer_plus_1 = &WEIGHTS_INPUT.layers[layer_num + 1];
                    if layer_plus_1.starts_with("relu") {
                        is_relu = true;
                        layer_num +=1;
                    }
                }

            if layer.starts_with("conv"){
                let i = conv_layer_num;

                let weights = read_4d_weights(api, &WEIGHTS_INPUT.conv_weights[i]);
                let bias: Vec<Variable> = WEIGHTS_INPUT
                    .conv_bias[i]
                    .clone()
                    .into_iter()
                    .map(|x| load_circuit_constant(api, x))
                    .collect();
                let temp_out = arrayd_to_vec4(out);

                let temp_out = conv_4d_run(api, temp_out, weights, bias,&WEIGHTS_INPUT.conv_dilation[i], &WEIGHTS_INPUT.conv_kernel_shape[i], &WEIGHTS_INPUT.conv_pads[i], &WEIGHTS_INPUT.conv_strides[i],&WEIGHTS_INPUT.conv_input_shape[i], WEIGHTS_INPUT.scaling, &WEIGHTS_INPUT.conv_group[i], is_rescale, v_plus_one, two_v, alpha_2_v, is_relu);
                out = vec4_to_arrayd(temp_out);
                conv_layer_num += 1;
            }
            //temporary solution
            else if layer.starts_with("reshape"){
                let reshape_shape: [usize; 2] = [1,12544];

                out = out
                    .into_shape_with_order(IxDyn(&reshape_shape))
                    .expect("Shape mismatch: Cannot reshape into the given dimensions");
            }
            else if layer.starts_with("fc"){
                let i = fc_layer_num;
                let weights = read_2d_weights(api, &WEIGHTS_INPUT.fc_weights[i]);
                let bias = read_2d_weights(api, &WEIGHTS_INPUT.fc_bias[i]);

                let mut out_2d = arrayd_to_vec2(out);

                out_2d = matrix_multplication_naive2(api, out_2d, weights);
                out_2d = matrix_addition_vec(api, out_2d, bias);
                api.display("3", out_2d[0][0]);

                out_2d = run_if_quantized_2d(api, WEIGHTS_INPUT.scaling, is_rescale, out_2d, v_plus_one, two_v, alpha_2_v, is_relu);
                out = vec2_to_arrayd(out_2d);
                fc_layer_num += 1;
            }
            layer_num += 1;
        }
        let output = arrayd_to_vec2(out);
        for (j, dim1) in self.outputs.iter().enumerate() {
                for (k, _dim2) in dim1.iter().enumerate() {
                    // api.display("out1", self.outputs[j][k]);
                    // api.display("out2", out_2d[j][k]);
                    api.assert_is_equal(self.outputs[j][k], output[j][k]);
                    // api.assert_is_different(self.outputs[j][k], 1);

                }
            }
    }
}
impl ConfigurableCircuit for ConvCircuit<Variable> {
    fn configure(&mut self) {
        // Change input and outputs as needed
        // self.input_arr[0][0][0][0] = PublicVariable::from(42);
        // Outputs
        let output_shape = WEIGHTS_INPUT.output_shape.clone();
        if output_shape.len() == 2{
            self.outputs = vec![vec![Variable::default(); output_shape[1]]; output_shape[0]];
        }
        else if output_shape.len() == 1{
            self.outputs = vec![vec![Variable::default(); output_shape[0]]];

        }
        else{
            panic!("Only output shape 2 has been implemented")
        }
        
        // Inputs
        let input_shape: Vec<usize> = WEIGHTS_INPUT.input_shape.clone();
    
        if input_shape.len() == 4{
            self.input_arr = vec![vec![vec![vec![Variable::default(); input_shape[3]]; input_shape[2]];input_shape[1]]; input_shape[0]];
        }
        else if input_shape.len() == 3{
            self.input_arr = vec![vec![vec![vec![Variable::default(); input_shape[2]]; input_shape[1]];input_shape[0]]];
        }
        else if input_shape.len() == 2{
            self.input_arr = vec![vec![vec![vec![Variable::default(); input_shape[1]]; input_shape[0]]]];
        }
        else if input_shape.len() == 1{
            self.input_arr = vec![vec![vec![vec![Variable::default(); input_shape[0]]]]];
        }
        else{
            panic!("Only 4 input shape dimensions has been implemented")
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

        assignment.input_arr = vec![vec![vec![vec![CircuitField::<C>::from(0); 28]; 28]; 4]; 1];
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

        /*
let mut out: Vec<Vec<Vec<Vec<CircuitField::<C>>>>> = Vec::new();

        for (i, dim1) in data.input.iter().enumerate() {
            let mut out_1: Vec<Vec<Vec<CircuitField::<C>>>> = Vec::new();
            for (j, dim2) in dim1.iter().enumerate() {
                let mut out_2:Vec<Vec<CircuitField::<C>>> = Vec::new();
                for (k, dim3) in dim2.iter().enumerate() {
                    let mut out_3:Vec<CircuitField::<C>> = Vec::new();
                    for (l, &element) in dim3.iter().enumerate() {
                        if element < 0 {
                            out_3.push(CircuitField::<C>::from(element.abs() as u32).neg());
                        } else {
                            out_3.push(CircuitField::<C>::from(element.abs() as u32));
                        }
                    }
                    out_2.push(out_3);
                }
                out_1.push(out_2);
            }
            out.push(out_1);
        }
        assignment.input_arr = out;

         */
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
        let mut out: Vec<Vec<CircuitField::<C>>> = Vec::new();
        for (_, dim1) in data.output.iter().enumerate() {
            let mut row: Vec<CircuitField::<C>> = Vec::new();
            for (_, &element) in dim1.iter().enumerate() {
                if element < 0 {
                    row.push(CircuitField::<C>::from_u256(U256::from(element.abs() as u64)).neg());
                } else {
                    row.push(CircuitField::<C>::from_u256(U256::from(element.abs() as u64)));
                }
            }
            out.push(row);
        }
        assignment.outputs = out;
        // Return the assignment
        assignment
    }
    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "demo_cnn".to_owned(),
    };
    println!("{:?}", WEIGHTS_INPUT.layers);

    handle_args::<BN254Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);

}
