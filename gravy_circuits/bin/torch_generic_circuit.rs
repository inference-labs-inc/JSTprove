use gravy_circuits::circuit_functions::convolution_fn::conv_4d_run;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::helper_fn::{arrayd_to_vec1, arrayd_to_vec2, arrayd_to_vec4, get_1d_circuit_inputs, get_2d_circuit_inputs, get_5d_circuit_inputs, load_circuit_constant, read_2d_weights, read_4d_weights, vec1_to_arrayd, vec2_to_arrayd, vec4_to_arrayd, vec5_to_arrayd, AnyDimVec};
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
use serde_json::Value;
use core::panic;

use ndarray::IxDyn;
use ndarray::Dimension;


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
    layers: Vec<Layer>,
    output_shape: Vec<usize>,
    input_shape: Vec<usize>,
    not_rescale_layers: Vec<String>,
    layer_input_shapes: Vec<Vec<usize>>,
    layer_output_shapes: Vec<Vec<usize>>
}

#[derive(Deserialize, Clone, Debug)]
// #[serde(deny_unknown_fields)] // Optional: remove this to allow unknowns
struct Layer{
    name: String,
    r#type: String,
    activation: String
}

#[derive(Deserialize, Clone)]
struct InputData {
    input: Value,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Value,
}

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/generic_demo_1d_weights.json");


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
    input_arr: [PublicVariable], // shape (m, n)
    outputs: [PublicVariable], // shape (m, k)
    // dummy: [PublicVariable; 10]
});



// Memorization, in a better place
impl<C: Config> Define<C> for ConvCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // TODO have this specified on python side
        // Confirm that this is hardcoded properly into the circuit, and not adjustable
        let n_bits = 32;

        // Similarly here
        let v_plus_one: usize = n_bits;
        let two_v: u32 = 1 << (v_plus_one - 1);

        // TODO adjust for different base
        let scaling_factor = 1 << WEIGHTS_INPUT.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        // Bring the weights into the circuit as constants
        // let mut out: Vec<Vec<Vec<Vec<Variable>>>> = self.input_arr.to_vec();
        let mut out = vec1_to_arrayd(self.input_arr.clone());

        api.display("{}", out[0]);
        api.display("{}", out[100]);


        let mut layer_num = 0;
        let mut conv_layer_num = 0;
        let mut fc_layer_num = 0;
        // panic!("{}", &WEIGHTS_INPUT.layers.len());

        // panic!("{:#?}", WEIGHTS_INPUT.layers); 
        assert!(WEIGHTS_INPUT.layers.len() > 0);

        while layer_num < WEIGHTS_INPUT.layers.len(){
            let layer = &WEIGHTS_INPUT.layers[layer_num].name;


            let mut is_rescale = true;
            for l in &WEIGHTS_INPUT.not_rescale_layers{
                if l.eq_ignore_ascii_case(layer){
                    is_rescale = false;
                }
            }
            

            let mut is_relu = false;
                if layer_num + 1 < WEIGHTS_INPUT.layers.len(){
                    let layer_plus_1 = &WEIGHTS_INPUT.layers[layer_num].activation;
                    if layer_plus_1.starts_with("ReLU") {
                        is_relu = true;
                        // layer_num +=1;
                    }
                }
            let raw_dim = out.raw_dim(); // Keep this alive
            let dim_view = raw_dim.as_array_view(); // Borrow from that
            let dim: &[usize] = dim_view.as_slice().unwrap(); // Now borrow is safe

            if WEIGHTS_INPUT.layer_input_shapes[layer_num] != dim{
                let reshape_shape = &WEIGHTS_INPUT.layer_input_shapes[layer_num];

                out = out
                    .into_shape_with_order(IxDyn(&reshape_shape))
                    .expect("Shape mismatch: Cannot reshape into the given dimensions");

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
        let flatten_shape: Vec<usize> = vec![WEIGHTS_INPUT.output_shape.iter().product()];

        out = out
            .into_shape_with_order(IxDyn(&flatten_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions");
        // panic!("{:?}, {:?}", out.dim(), &reshape_shape);

        let output = arrayd_to_vec1(out);
        for (j, _) in self.outputs.iter().enumerate() {
                    api.display("out1", self.outputs[j]);
                    api.display("out2", output[j]);
                    api.assert_is_equal(self.outputs[j], output[j]);
                    // api.assert_is_different(self.outputs[j], 1);
            }
    }
}



impl ConfigurableCircuit for ConvCircuit<Variable> {
    fn configure(&mut self) {
        // Change input and outputs as needed
        // Outputs
        let output_dims: usize = WEIGHTS_INPUT.output_shape.iter().product();
        self.outputs = vec![Variable::default(); output_dims];

        // Inputs
        let input_dims: usize = WEIGHTS_INPUT.input_shape.iter().product();
        self.input_arr = vec![Variable::default(); input_dims];


    }
}



impl<C: Config> IOReader<ConvCircuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: ConvCircuit<CircuitField::<C>>,
    ) -> ConvCircuit<CircuitField::<C>> {
        /*
            TODO - Can rework this code potentially to speed up witness generation...
         */
        let data: InputData = <FileReader as IOReader<ConvCircuit<_>, C>>::read_data_from_json::<
            InputData,
        >(file_path);

        let input_dims: &[usize] = &[WEIGHTS_INPUT.input_shape.iter().product()];


        assignment.input_arr = get_1d_circuit_inputs::<C>(&data.input, input_dims);
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

        let output_dims: &[usize] = &[WEIGHTS_INPUT.output_shape.iter().product()];


        assignment.outputs = get_1d_circuit_inputs::<C>(&data.output, output_dims);
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
    // println!("{:?}", WEIGHTS_INPUT.layers);

    handle_args::<BN254Config, ConvCircuit<Variable>,ConvCircuit<_>,_>(&mut file_reader);

}
