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
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use core::panic;
use std::collections::HashMap;

use ndarray::IxDyn;
use ndarray::Dimension;


use gravy_circuits::runner::main_runner::{handle_args, ConfigurableCircuit};


type WeightsData = (Architecture, W_and_B, CircuitParams);
#[derive(Deserialize, Clone, Debug)]
struct Architecture{
    inputs: Vec<ONNXIO>,
    outputs: Vec<ONNXIO>,
    architecture: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
struct W_and_B{
    w_and_b: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
struct CircuitParams{
    scale_base: u32,
    scaling: u32,
    rescale_config: HashMap<String, bool>
}

#[derive(Deserialize, Clone, Debug)]
struct ONNXLayer{
    id: usize,
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    shape: HashMap<String, Vec<usize>>,
    tensor: Option<Value>,
    params: Option<Value>,
    opset_version_number: i16,
}

#[derive(Deserialize, Clone, Debug)]
struct ONNXIO{
    name: String,
    elem_type: i16,
    shape: Vec<usize>
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
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/onnx_generic_circuit_weights.json");


//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
    static ref ARCHITECTURE: Architecture = WEIGHTS_INPUT.0.clone();
    static ref W_AND_B: W_and_B = WEIGHTS_INPUT.1.clone();
    static ref CIRCUITPARAMS: CircuitParams = WEIGHTS_INPUT.2.clone();

}
// TODO implement various inputs (maybe the handling should be done inside)
declare_circuit!(ConvCircuit {
    input_arr: [PublicVariable], // shape (m, n)
    outputs: [PublicVariable], // shape (m, k)
    dummy: [Variable; 2]
});

fn parse_value_to_array<I: DeserializeOwned>(value: Value) -> Result<I, serde_json::Error>{
    match serde_json::from_value::<I>(value) {
        Ok(vec_from_value) => {
            // Use vec_from_value here
            Ok(vec_from_value)
        }
        Err(e) => {
            eprintln!("Failed to parse JSON value: {}", e);
            Err(e)
        }
    }
}
fn transpose<T: Clone>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut transposed = vec![Vec::with_capacity(rows); cols];

    for row in matrix {
        assert_eq!(row.len(), cols, "All rows must be the same length");
        for (j, val) in row.into_iter().enumerate() {
            transposed[j].push(val);
        }
    }

    transposed
}
pub fn parse_maybe_1d_to_2d<T: DeserializeOwned + Clone>(
    value: Value,
) -> Result<Vec<Vec<T>>, serde_json::Error> {
    // Try parsing as 2D array first
    match serde_json::from_value::<Vec<Vec<T>>>(value.clone()) {
        Ok(vv) => Ok(vv),
        Err(_) => {
            // Try parsing as 1D array
            let v: Vec<T> = serde_json::from_value(value)?;
            Ok(vec![v]) // Wrap into a 2D vector
        }
    }
}


fn collect_all_shapes(layers: &[ONNXLayer], ios: &[ONNXIO]) -> HashMap<String, Vec<usize>> {
    let mut result = HashMap::new();

    // Merge from layers
    for layer in layers {
        for (key, shape) in &layer.shape {
            result.insert(key.clone(), shape.clone());
        }
    }

    // Merge from IOs
    for io in ios {
        result.insert(io.name.clone(), io.shape.clone());
    }

    result
}


// TODO all panics below need to be replaced by proper errors

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
        let scaling_factor = 1 << CIRCUITPARAMS.scaling;
        let alpha_2_v = api.mul(scaling_factor, two_v);

        // let scaling  = (CIRCUITPARAMS.scale_base as u32).pow(CIRCUITPARAMS.scaling as u32);

        // Bring the weights into the circuit as constants
        // let mut out: Vec<Vec<Vec<Vec<Variable>>>> = self.input_arr.to_vec();
        let mut out = vec1_to_arrayd(self.input_arr.clone());

        api.display("{}", out[0]);
        api.display("{}", out[100]);

        let inputs = &ARCHITECTURE.inputs;
        let outputs = &ARCHITECTURE.outputs;
        // TODO only accounts for single input for now, must account for more
        let mut shape = &inputs[0].shape;


        let mut layer_num = 0;
        assert!(ARCHITECTURE.architecture.len() > 0);

        // Load weights and biases into hashmap
        let w_and_b_map: HashMap<String, ONNXLayer> = W_AND_B.w_and_b.clone()
            .into_iter()
            .map(|layer| (layer.name.clone(), layer))
            .collect();

        let shapes_map: HashMap<String, Vec<usize>> = collect_all_shapes(&ARCHITECTURE.architecture, inputs);
        // panic!("{:?}", shapes_map);

        


        // panic!("{:?}", ARCHITECTURE.architecture);
        while layer_num < ARCHITECTURE.architecture.len(){
            let layer = ARCHITECTURE.architecture[layer_num].clone();
            let layer_name = &ARCHITECTURE.architecture[layer_num].name;
            let layer_type = &ARCHITECTURE.architecture[layer_num].op_type;
            // TODO, proper constant logic
            if layer_type.starts_with("Constant"){
                layer_num +=1;
                continue
            }

            // TODO this only works with single output shape
            // shape = match  ARCHITECTURE.architecture[layer_num].shape.get(&ARCHITECTURE.architecture[layer_num].outputs[0]){
            //     Some(input_shape) => input_shape,
            //     None => panic!("Error getting output shape for layer {}", layer_name)
            // };
            // TODO this should be done on a per input basis. For now this only works because we are looking at single input layers
            shape = match  shapes_map.get(&ARCHITECTURE.architecture[layer_num].inputs[0]){
                Some(input_shape) => input_shape,
                None => panic!("Error getting output shape for layer {}", layer_name)
            };
            // if layer_num == 0{
            //     panic!("{}", layer_name);
            // }

            // let mut is_rescale = true;
            let is_rescale = match  CIRCUITPARAMS.rescale_config.get(layer_name){
                Some(config) => config,
                None => &true
            };
            /*
                TODO This should be changed, where either here or in python, we analyze the outputs of the given file to determine if there is an activation applied to the outputs.
                TODO Also, we should diverge the outputs into different paths, depending on the outputs of the layer.
             */
             
            let mut is_relu = false;
                if layer_num + 1 < ARCHITECTURE.architecture.len(){
                    let layer_plus_1 = &ARCHITECTURE.architecture[layer_num + 1].op_type;
                    if layer_plus_1.starts_with("Relu") {
                        is_relu = true;
                        layer_num +=1;
                        // panic!("{}", layer_name);
                    }
                }
                

            let raw_dim = out.raw_dim(); // Keep this alive
            let dim_view = raw_dim.as_array_view(); // Borrow from that
            let dim: &[usize] = dim_view.as_slice().unwrap(); // Now borrow is safe

            if shape != dim{
                let reshape_shape = shape;
                // if layer_num >= 1{
                //     panic!("{:?}, {:?}, {}, {}", reshape_shape, dim, layer_name, layer_type);
                // }
                out = out
                    .into_shape_with_order(IxDyn(&reshape_shape))
                    .expect("Shape mismatch: Cannot reshape into the given dimensions");
            }
            

            if layer_type.starts_with("Conv"){
                // let i = conv_layer_num;
                let weights_input = &layer.inputs[1];
                let bias_input = &layer.inputs[2];

                // TODO these should be cleaned up a bit
                let weight_tensor = get_w_or_b(&w_and_b_map, weights_input);

                let bias_tensor_option = match  w_and_b_map.get(bias_input) {
                    Some(tensor) => tensor.tensor.clone(),
                    None => panic!("ModelError - missing weights and biases: {}", bias_input)
                };
                let bias_tensor: Vec<i64> = match bias_tensor_option {
                    Some(tensor) => parse_value_to_array(tensor).unwrap(),
                    None => panic!("ModelError - missing tensor in expected weights/bias: {}", bias_input)
                };
                let params = layer.params.unwrap();
                // TODO fix this to a function, for getting parameters, maybe have some default value as well?
                let dilation = get_param(layer_name, &"dilations", &params);
                let kernel_shape = get_param(layer_name, &"kernel_shape", &params);
                let group = vec![get_param(layer_name, &"group", &params)];
                let pads = get_param(layer_name, &"pads", &params);
                let strides = get_param(layer_name, &"strides", &params);
                // // Scale up bias tensor TODO find a better way
                // let bias_tensor: Vec<i64> = bias_tensor
                //     .iter()
                //     .map(|&val| val * scaling as i64).collect();


                let weights = read_4d_weights(api, &weight_tensor);
                let bias: Vec<Variable> = bias_tensor
                    .clone()
                    .into_iter()
                    .map(|x| load_circuit_constant(api, x))
                    .collect();

                let temp_out = arrayd_to_vec4(out);

                let in_shape = vec![temp_out.len(),temp_out[0].len(), temp_out[0][0].len(), temp_out[0][0][0].len()];
                
                // panic!("{}, {}", &is_rescale, is_relu);
                let temp_out = conv_4d_run(api, temp_out, weights, bias,&dilation, &kernel_shape, &pads, &strides, &convert_usize_to_u32(in_shape.to_vec()), CIRCUITPARAMS.scaling.into(), &group, is_rescale.clone(), v_plus_one, two_v, alpha_2_v, is_relu);
                out = vec4_to_arrayd(temp_out);
                // conv_layer_num += 1;
            }
            else if layer_type.starts_with("Gemm"){
                let weights_input = &layer.inputs[1];
                let bias_input = &layer.inputs[2];

                let weights_tensor_option = match w_and_b_map.get(weights_input) {
                    Some(tensor) => tensor.tensor.clone(),
                    None => panic!("ModelError - missing weights and biases: {}", weights_input)
                };
                let mut weight_tensor = match weights_tensor_option {
                    Some(tensor) => parse_value_to_array(tensor).unwrap(),
                    None => panic!("ModelError - missing tensor in expected weights/bias: {}", weights_input)
                };

                let bias_tensor_option = match  w_and_b_map.get(bias_input) {
                    Some(tensor) => tensor.tensor.clone(),
                    None => panic!("ModelError - missing weights and biases: {}", bias_input)
                };
                let bias_tensor: Vec<Vec<i64>> = match bias_tensor_option {
                    Some(tensor) => parse_maybe_1d_to_2d(tensor).unwrap(),
                    None => panic!("ModelError - missing tensor in expected weights/bias: {}", bias_input)
                };
                // // Scale up bias tensor
                // let bias_tensor = bias_tensor
                //     .iter()
                //     .map(|row| row.iter().map(|&val| val * scaling as i64).collect())
                //     .collect();


                let params = layer.params.unwrap();
                // TODO, not implemented yet
                let alpha: f32 = get_param(layer_name, &"alpha", &params);
                let beta: f32 = get_param(layer_name, &"beta", &params);
                // TODO No trans_a, must figure out what to do if doesnt exists
                // let trans_a: usize = get_param(layer_name, &"transA", &params);



                let trans_b: usize = get_param(layer_name, &"transB", &params);
                if trans_b == 1{
                    weight_tensor = transpose(weight_tensor);
                }



                let weights = read_2d_weights(api, &weight_tensor);
                let bias = read_2d_weights(api, &bias_tensor);
                
                let mut out_2d = arrayd_to_vec2(out);

                out_2d = matrix_multplication_naive2(api, out_2d, weights);
                out_2d = matrix_addition_vec(api, out_2d, bias);
                api.display("3", out_2d[0][0]);

                out_2d = run_if_quantized_2d(api, CIRCUITPARAMS.scaling.into(), is_rescale.clone(), out_2d, v_plus_one, two_v, alpha_2_v, is_relu);
                out = vec2_to_arrayd(out_2d);
            }
            layer_num += 1;
        }
        let flatten_shape: Vec<usize> = vec![outputs.iter()
        .map(|obj| obj.shape.iter().product::<usize>())
        .product()];

        out = out
            .into_shape_with_order(IxDyn(&flatten_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions");
        // panic!("{:?}, {:?}", out.dim(), &reshape_shape);


        let output = arrayd_to_vec1(out);
        for (j, _) in self.outputs.iter().enumerate() {
                    api.display("out1", self.outputs[j]);
                    api.display("out2", output[j]);
                    api.assert_is_equal(self.outputs[j], output[j]);
                    // api.assert_is_different(self.outputs[j], 1123412314);
            }
            api.assert_is_equal(self.dummy[0], 1);
            api.assert_is_equal(self.dummy[1], 1);

    }
}

fn get_w_or_b<I: DeserializeOwned>(w_and_b_map: &HashMap<String, ONNXLayer>, weights_input: &String) -> I{
    let weights_tensor_option = match w_and_b_map.get(weights_input) {
        Some(tensor) => tensor.tensor.clone(),
        None => panic!("ModelError - missing weights and biases: {}", weights_input)
    };
    let weight_tensor = match weights_tensor_option {
        Some(tensor) => parse_value_to_array(tensor).unwrap(),
        None => panic!("ModelError - missing tensor in expected weights/bias: {}", weights_input)
    };
    serde_json::from_value(weight_tensor).expect("Deserialization failed")
}

fn get_param<I:DeserializeOwned>(layer_name: &String, param_name: &str, params: &Value) -> I {
    match params.get(param_name){
        Some(param) => {
            let x = param.clone();
            serde_json::from_value(x.clone()).expect(&format!("❌ Failed to parse param '{}': got value {}", param_name, x))

        },
        None => panic!("ParametersError: {} is missing {}", layer_name, param_name)
    }
}

fn convert_usize_to_u32(input: Vec<usize>) -> Vec<u32> {
    input.into_iter().map(|x| x as u32).collect()
}

impl ConfigurableCircuit for ConvCircuit<Variable> {
    fn configure(&mut self) {
        // Change input and outputs as needed
        // Outputs
        // let output_dims: usize = WEIGHTS_INPUT.output_shape.iter().product();
        let output_dims: usize = ARCHITECTURE.outputs.iter()
        .map(|obj| obj.shape.iter().product::<usize>())
        .product();
        self.outputs = vec![Variable::default(); output_dims];

        // Inputs
        let input_dims: usize = ARCHITECTURE.inputs.iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product();    
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

        let input_dims: &[usize] = &[ARCHITECTURE.inputs.iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()]; 
        assignment.dummy[0] = CircuitField::<C>::from(1);
        assignment.dummy[1] = CircuitField::<C>::from(1);


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

        let output_dims: &[usize] = &[ARCHITECTURE.outputs.iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()]; 

        // let output_dims: &[usize] = &[WEIGHTS_INPUT.output_shape.iter().product()];


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
