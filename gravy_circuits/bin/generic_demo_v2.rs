use core::panic;
use std::env::consts::ARCH;
use expander_compiler::circuit::ir::common::display;
use expander_compiler::frontend::*;
use gravy_circuits::circuit_functions::convolution_fn::conv_4d_run;
use gravy_circuits::circuit_functions::helper_fn::{arrayd_to_vec1, arrayd_to_vec2, arrayd_to_vec4, arrayd_to_vec5, get_1d_circuit_inputs, load_circuit_constant, read_2d_weights, read_4d_weights, vec1_to_arrayd, vec2_to_arrayd, vec4_to_arrayd, vec5_to_arrayd};
#[allow(unused_imports)]
use gravy_circuits::circuit_functions::matrix_computation::{
    matrix_addition_vec, matrix_multplication, matrix_multplication_array,
    matrix_multplication_naive, matrix_multplication_naive2, matrix_multplication_naive2_array,
    matrix_multplication_naive3, matrix_multplication_naive3_array,
};
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::runner::main_runner::{handle_args, ConfigurableCircuit};
use lazy_static::lazy_static;
use ndarray::{Array1, Dimension};
use ndarray::{ ArrayD, IxDyn};
use rand::distributions::WeightedError;
use std::collections::HashMap;

// Serde Packages
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use gravy_circuits::circuit_functions::quantization::run_if_quantized_2d;


type WeightsData = (Architecture, WANDB, CircuitParams);
#[derive(Deserialize, Clone, Debug)]
struct Architecture{
    inputs: Vec<ONNXIO>,
    outputs: Vec<ONNXIO>,
    architecture: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
struct WANDB{
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
    static ref W_AND_B: WANDB = WEIGHTS_INPUT.1.clone();
    static ref CIRCUITPARAMS: CircuitParams = WEIGHTS_INPUT.2.clone();

}

declare_circuit!(Circuit {
    input_arr: [PublicVariable], // shape (m, n)
    outputs: [PublicVariable],   // shape (m, k)
    dummy: [Variable; 2]
});

/*
ConvLayer, ReshapeLayer, FCLayer
*/
#[derive(Debug)]
struct ConvLayer {
    name: String,
    index: usize,
    weights: Vec<Vec<Vec<Vec<i64>>>>,
    bias: Vec<i64>,
    strides: Vec<u32>,
    kernel_shape: Vec<u32>,
    group: Vec<u32>,
    dilation: Vec<u32>,
    pads: Vec<u32>,
    input_shape: Vec<usize>,
    scaling: u64,
    is_relu: bool,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: u64,
    is_rescale: bool,
    inputs: Vec<String>,
    outputs: Vec<String>,
}
#[derive(Debug)]
struct ReshapeLayer {
    name: String,
    shape: Vec<usize>,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,

}
#[derive(Debug)]
struct ConstantLayer {
    name: String,
    value: Value,
    outputs: Vec<String>,
}

#[derive(Debug)]
struct GemmLayer {
    name: String,
    index: usize,
    weights: Vec<Vec<i64>>,
    bias: Vec<Vec<i64>>,
    is_rescale: bool,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: u64,
    is_relu: bool,
    scaling: u64,
    input_shape: Vec<usize>,
    alpha: f32,
    beta: f32,
    transb: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

trait LayerOp<C: Config, Builder: RootAPI<C>> {
    fn apply(&self, api: &mut Builder, input: HashMap<String,ArrayD<Variable>>)
    // fn apply(&self, api: &mut Builder, input: ArrayD<Variable>)
        -> Result<(String,ArrayD<Variable>), String>;

    // fn build for every Layer type
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        eprintln!("Applying input shape for CONV: {}", self.name);

        eprintln!("{:?}", self.inputs);


        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        // Reshape inputs
        let layer_input = reshape_layer(layer_input, &self.input_shape);

        // Get weights and biases
        let weights = read_4d_weights(api, &self.weights);
        let bias: Vec<Variable> = self.bias
            .clone()
            .into_iter()
            .map(|x| load_circuit_constant(api, x))
            .collect();
        // Obtain scaling factors (This can move TODO)
        let scale_factor = 1 << self.scaling;
        let alpha_two_v = api.mul(self.two_v as u32, scale_factor as u32);

        // Convert inputs to correct form
        let layer_input = arrayd_to_vec4(layer_input);
        eprintln!("GOT Input:");
        // TODO there should be a better way to do this (Maybe even inside conv4drun)
        // Get input shape
        let in_shape = vec![layer_input.len() as u32,layer_input[0].len() as u32, layer_input[0][0].len() as u32, layer_input[0][0][0].len() as u32];
    
        
        // Run convolution
        let out = conv_4d_run(
            api,
            layer_input,
            weights,
            bias,
            &self.dilation,
            &self.kernel_shape,
            &self.pads,
            &self.strides,
            &in_shape,
            self.scaling,
            &self.group,
            self.is_rescale,
            self.v_plus_one,
            self.two_v,
            alpha_two_v,
            self.is_relu,
        );
        eprintln!("GOT OUTPUT");
        eprint!("PRINTING INDEX");

        Ok((self.outputs[0].clone(), vec4_to_arrayd(out)))
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GemmLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        eprintln!("Applying input shape for GemmLayer: {}", self.name);
        eprintln!("{:?}", self.inputs);


        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        // Reshape inputs
        let layer_input = reshape_layer(layer_input, &self.input_shape);

        let mut weight_tensor = self.weights.clone();
        if self.transb == 1{
            weight_tensor = transpose(weight_tensor);
        }
        let weights = read_2d_weights(api, &weight_tensor);
        let bias = read_2d_weights(api, &self.bias);
        let mut out_2d = arrayd_to_vec2(layer_input); // Potential point of failure, check with Jonathan

        let scale_factor = 1 << self.scaling;
        let alpha_two_v = api.mul(self.two_v as u32, scale_factor as u32);

        out_2d = matrix_multplication_naive2(api, out_2d, weights);
        eprintln!("out2d dimension {}, {}", out_2d.len(), out_2d[0].len());
        out_2d = matrix_addition_vec(api, out_2d, bias);
        api.display("3", out_2d[0][0]);
        eprintln!("GOT display:");
        out_2d = run_if_quantized_2d(api, CIRCUITPARAMS.scaling.into(), self.is_rescale, out_2d, self.v_plus_one, self.two_v, alpha_two_v, self.is_relu);
        eprintln!("GOT output:");
        let out = vec2_to_arrayd(out_2d);
        eprintln!("Finished");
        Ok((self.outputs[0].clone(), out))
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReshapeLayer {
    /*
    TO-DO: Implement permanent implementation, currently temp solution
     */
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        let reshape_shape = self.shape.clone();
        let mut layer_input = input.get(&self.inputs[0]).unwrap();
        let out = &layer_input.clone()
            .into_shape_with_order(IxDyn(&reshape_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions.");

        Ok((self.outputs[0].clone(), out.clone()))
    }
}
// TODO remove constants from python side. Incorporate into the layer that uses it instead
impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantLayer {
    // Passthrough
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {

        Ok((self.outputs[0].clone(), ArrayD::from_shape_vec(IxDyn(&[1]), vec![api.constant(0)]).unwrap()))
    }
}

type BoxedDynLayer<C, B> = Box<dyn LayerOp<C, B>>;

fn build_layers<C: Config, Builder: RootAPI<C>>() -> Vec<Box<dyn LayerOp<C, Builder>>> {
    let mut layers: Vec<BoxedDynLayer<C, Builder>> = vec![];
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u32 = 1 << (V_PLUS_ONE - 1);
    let alpha_two_v: u64 = ((1 << CIRCUITPARAMS.scaling) * TWO_V) as u64;

    // Load weights and biases into hashmap

    /*
    TODO: Inject weights + bias data with external functions instead of regular assignment in function.
     */

    let w_and_b_map: HashMap<String, ONNXLayer> = W_AND_B.w_and_b.clone()
        .into_iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();


    let mut skip_next_layer = false;
    // let skip_future_layers = Set();

    let inputs = &ARCHITECTURE.inputs;

    // TODO havent figured out how but this can maybe go in build layers?
    let shapes_map: HashMap<String, Vec<usize>> = collect_all_shapes(&ARCHITECTURE.architecture, inputs);
    // TODO should account for multiple inputs

    for (i, layer) in ARCHITECTURE.architecture.iter().enumerate() {
        /*
        TODO track relu through outputs instead. And skip the output layer when it comes in the graph (Maybe use a set containing the outputs)
         */
        /* if layer.name in skip_future_layers {
            skip_future_layers.pop(layer.name)
            continue
        }
        */
        if skip_next_layer{
            skip_next_layer = false;
            continue
        }
        // TODO needs to fix this approach
        let mut outputs = layer.outputs.to_vec();
        let mut is_relu = false;
        if i + 1 < ARCHITECTURE.architecture.len(){
            let layer_plus_1 = &ARCHITECTURE.architecture[i + 1].op_type;
            // let layer_plus_1 = &layer.output.name
            // May need to store all layer architecture in a hashmap or something, and access the next layer through the hashmap
            if layer_plus_1.starts_with("Relu") {
                eprintln!("Found Relu for {}", layer.name.as_str());
                is_relu = true;
                skip_next_layer = true;
                outputs = ARCHITECTURE.architecture[i + 1].outputs.clone();
                // skip_future_layers.add(layer_plus_1.name)
            }
        }

        let is_rescale = match  CIRCUITPARAMS.rescale_config.get(&layer.name){
                Some(config) => config,
                None => &true
            };

        /*
        Implement Static Polymorphism, want to generalize a build function
        no matter the type of layer we want to build and return that layer type.
        */
        match layer.op_type.as_str() {
            "Conv" => {
                let params = layer.params.clone().unwrap();
                // TODO this should be done on a per input basis. For now this only works because we are looking at single input layers
                // I think this should move inside the individual layers
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let conv = ConvLayer {
                    name: layer.name.clone(),
                    index: i,
                    weights: get_w_or_b(&w_and_b_map, &layer.inputs[1]),
                    bias: get_w_or_b(&w_and_b_map, &layer.inputs[2]),
                    strides: get_param(&layer.name, &"strides", &params),
                    kernel_shape: get_param(&layer.name, &"kernel_shape", &params),
                    group: vec![get_param(&layer.name, &"group", &params)],
                    dilation: get_param(&layer.name, &"dilations", &params),
                    pads: get_param(&layer.name, &"pads", &params),
                    input_shape: expected_shape.to_vec(),
                    scaling: CIRCUITPARAMS.scaling.into(),
                    is_relu: is_relu,
                    v_plus_one: N_BITS,
                    two_v: TWO_V,
                    /*
                    TODO - api.mul instead of hard-coding multiplication
                     */
                    alpha_two_v: alpha_two_v,
                    is_rescale: *is_rescale, //DONT KNOW IF THIS IS IDEAL TODO
                    inputs: layer.inputs.to_vec(),
                    outputs: outputs
                };

                layers.push(Box::new(conv));
            }
            "Reshape" => {
                /*
                   TODO - Implement permanent solution for what reshape layer needs like
                */
                let shape_name = layer.inputs[1].clone();
                let params = layer.params.clone().unwrap();
                
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let output_shape = shapes_map.get(&layer.outputs.to_vec()[0]);
                let reshape = ReshapeLayer {
                    name: layer.name.clone(),
                    input_shape: expected_shape.to_vec(),
                    inputs: layer.inputs.to_vec(),
                    outputs: layer.outputs.to_vec(),
                    shape: get_param_or_default(&layer.name, &shape_name, &params, output_shape)
                };
                layers.push(Box::new(reshape));
            }
            "Gemm" => {
                let params = layer.params.clone().unwrap();
                // TODO this should be done on a per input basis. For now this only works because we are looking at single input layers
                // I think this should move inside the individual layers
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let gemm = GemmLayer {
                    name: layer.name.clone(),
                    index: i,
                    weights: get_w_or_b(&w_and_b_map, &layer.inputs[1]),
                    bias: parse_maybe_1d_to_2d(get_w_or_b(&w_and_b_map, &layer.inputs[2])).unwrap(),
                    is_relu: is_relu,
                    v_plus_one: V_PLUS_ONE,
                    two_v: TWO_V,
                    alpha_two_v: alpha_two_v,
                    is_rescale: is_rescale.clone(),
                    scaling: CIRCUITPARAMS.scaling.into(), // TODO: Becomes scaling_in?
                    input_shape: expected_shape.to_vec(),
                    alpha: get_param(&layer.name, &"alpha", &params),
                    beta: get_param(&layer.name, &"beta", &params),
                    transb: get_param(&layer.name, &"transB", &params),
                    inputs: layer.inputs.to_vec(),
                    outputs: outputs

                };
                layers.push(Box::new(gemm));
            }
            "Constant" => {
                // let constant = ConstantLayer {
                //     value: get_param(&layer.name, &"value", &layer.params.clone().unwrap())
                // };
                // layers.push(Box::new(constant));
            }
            other => panic!("Unsupported layer type: {}", other),
        }
        eprintln!("layer added: {}", layer.op_type.as_str() );
    }
    layers
}

fn get_inputs<T: Clone>(v: Vec<T>, inputs: Vec<ONNXIO>) -> HashMap<String, ArrayD<T>>{
    // Step 1: Compute total number of elements required
    let total_required: usize = inputs
        .iter()
        .map(|input| input.shape.iter().product::<usize>())
        .sum();

    // Step 2: Validate that v has exactly the required number of elements
    if v.len() != total_required {
        panic!(
            "Input data length mismatch: got {}, but {} elements required by input shapes",
            v.len(),
            total_required
        );
    }

    // Step 3: Split and reshape
    let mut result = HashMap::new();
    let mut start = 0;

    for input_info in inputs {
        let num_elements: usize = input_info.shape.iter().product();
        let end = start + num_elements;

        let slice = v[start..end].to_vec(); // clone slice
        let arr = ArrayD::from_shape_vec(IxDyn(&input_info.shape), slice)
            .expect("Invalid shape for input data");

        result.insert(input_info.name.clone(), arr);
        start = end;
    }

    result
}

// Memorization, in a better place
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Getting inputs
        let mut out = get_inputs(self.input_arr.clone(), ARCHITECTURE.inputs.clone());
        
        // let mut out = out2.remove("input").unwrap().clone();



        // let mut out = vec1_to_arrayd(self.input_arr.clone());
        let layers = build_layers::<C, Builder>();
        
        assert!(ARCHITECTURE.architecture.len() > 0);

        for (i, layer) in layers.iter().enumerate() {

            eprintln!("Applying Layer {:?}", &ARCHITECTURE.architecture[i].name);
            // TODO worried about clone in here being very expensive
            let result = layer
                .apply(api, out.clone())
                .expect(&format!("Failed to apply layer {}", i));
            out.insert(result.0, result.1);

        }
        
        eprint!("Flatten output");
        let flatten_shape: Vec<usize> = vec![ARCHITECTURE.outputs.iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()];

        // TODO only support single output
        let output_name = ARCHITECTURE.outputs[0].name.clone();

        let output = out.get(&output_name).unwrap().clone()
            .into_shape_with_order(IxDyn(&flatten_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions"); 

        let output = arrayd_to_vec1(output);

        eprint!("Assert outputs match");
        for (j, _) in self.outputs.iter().enumerate() {
            api.display("out1", self.outputs[j]);
            api.display("out2", output[j]);
            api.assert_is_equal(self.outputs[j], output[j]);
            // api.assert_is_different(self.outputs[j], 13241234);
        }
        api.assert_is_equal(self.dummy[0], 1);
        api.assert_is_equal(self.dummy[1], 1);
        eprintln!("Outputs match");

    }
}

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
fn get_param<I:DeserializeOwned>(layer_name: &String, param_name: &str, params: &Value) -> I {
    match params.get(param_name){
        Some(param) => {
            let x = param.clone();
            serde_json::from_value(x.clone()).expect(&format!("❌ Failed to parse param '{}': got value {}", param_name, x))

        },
        None => panic!("ParametersError: {} is missing {}", layer_name, param_name)
    }
}


fn get_param_or_default<I: DeserializeOwned + Clone>(
    layer_name: &str,
    param_name: &str,
    params: &Value,
    default: Option<&I>,
) -> I {
    match params.get(param_name) {
        Some(param) => {
            let x = param.clone();
            match serde_json::from_value(x.clone()) {
                Ok(value) => value,
                Err(_) => {
                    eprintln!("⚠️ Warning: Failed to parse param '{}': got value {} — using default", param_name, x);
                    default.unwrap().clone()
                }
            }
        },
        None => {
            eprintln!("⚠️ Warning: ParametersError: '{}' is missing '{}' — using default", layer_name, param_name);
            default.unwrap().clone()
        }
    }
}

fn reshape_layer(input: ndarray::ArrayBase<ndarray::OwnedRepr<Variable>, ndarray::Dim<ndarray::IxDynImpl>>, input_shape: &[usize]) -> ndarray::ArrayBase<ndarray::OwnedRepr<Variable>, ndarray::Dim<ndarray::IxDynImpl>> {
    let raw_dim = input.raw_dim();
    // Keep this alive
    let dim_view = raw_dim.as_array_view();
    // Borrow from that
    let dim: &[usize] = dim_view.as_slice().unwrap();
    // Now borrow is safe

    let expected_shape = input_shape;

    eprintln!("Current Shape {:?}", dim);
    eprintln!("Expected Shape {:?}", expected_shape);

    // TODO remove this
    let mut input = input;
    
    if dim != expected_shape {
        let reshape_shape = &expected_shape;


        input = input
            .into_shape_with_order(IxDyn(&reshape_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions");
    }
    input
}
fn convert_usize_to_u32(input: Vec<usize>) -> Vec<u32> {
    input.into_iter().map(|x| x as u32).collect()
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
    weight_tensor
    // serde_json::from_value(weight_tensor).expect("Deserialization failed")
}

impl ConfigurableCircuit for Circuit<Variable> {
    fn configure(&mut self) {
        // Change input and outputs as needed
        // Outputs
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

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Circuit<CircuitField<C>> {
        /*
           TODO - Can rework this code potentially to speed up witness generation...
        */
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path);

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
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Circuit<CircuitField<C>> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path);

        let output_dims: &[usize] = &[ARCHITECTURE.outputs.iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()]; 

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

    handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(&mut file_reader);
}
