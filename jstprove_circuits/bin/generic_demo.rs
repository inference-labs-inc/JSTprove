use core::panic;
use std::any::TypeId;
use expander_compiler::frontend::*;
use jstprove_circuits::circuit_functions::layers::conv::conv_4d_run;
use jstprove_circuits::circuit_functions::utils::helper::{
    load_array_constants,
    arrayd_to_vec1, arrayd_to_vec2, arrayd_to_vec4, arrayd_to_vec5,
    get_1d_circuit_inputs,
    load_circuit_constant,
    read_2d_weights, read_4d_weights,
    vec1_to_arrayd, vec2_to_arrayd, vec4_to_arrayd, vec5_to_arrayd,
};
#[allow(unused_imports)]
use jstprove_circuits::circuit_functions::layers::gemm::{
    matrix_addition,
    matrix_multiplication,
};

use jstprove_circuits::circuit_functions::layers::maxpool::{
    setup_maxpooling_2d,
    maxpooling_2d,
};

use jstprove_circuits::circuit_functions::layers::relu::{
    relu_array,
    ReluContext,
};

use jstprove_circuits::io::io_reader::{FileReader, IOReader};
use jstprove_circuits::runner::main_runner::{handle_args, ConfigurableCircuit};
use lazy_static::lazy_static;
use ndarray::Dimension;
use ndarray::{ArrayD, IxDyn, Array2, Axis, Ix2};
use std::collections::HashMap;

// Serde Packages
use serde::de::{value, DeserializeOwned};
use serde::Deserialize;
use serde_json::Value;

use jstprove_circuits::circuit_functions::utils::quantization::{
    rescale_array, 
};

use jstprove_circuits::circuit_functions::utils::helper::IntoTensor;


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
    weights: ArrayD<i64>,
    bias: ArrayD<i64>,
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
struct FlattenLayer {
    name: String,
    axis: usize,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Debug)]
struct ReluLayer {
    name: String,
    index: usize,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
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
    weights: ArrayD<i64>,
    bias: ArrayD<i64>,
    is_rescale: bool,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: u64,
    is_relu: bool,
    scaling: u64,
    input_shape: Vec<usize>,
    alpha: f32,
    beta: f32,
    transa: usize,
    transb: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

fn get_vector_dim<T>(v: &Vec<T>) -> usize {
    if let Some(first) = v.first() {
        if let Some(inner) = any_as_vec_ref(first) {
            1 + get_vector_dim(inner)
        } else {
            1
        }
    } else {
        1
    }
}
// Helper to try casting to Vec<_>
fn any_as_vec_ref<T>(_: &T) -> Option<&Vec<T>> {
    None // Rust has no runtime reflection to inspect Vec<T>'s contents
}

#[derive(Debug)]
struct MaxPoolLayer {
    name: String,
    kernel_shape: Vec<usize>,
    strides: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    input_shape: Vec<usize>,
    shift_exponent: usize, 
    inputs: Vec<String>,
    outputs: Vec<String>,
}

trait LayerOp<C: Config, Builder: RootAPI<C>> {
    fn apply(&self, api: &mut Builder, input: HashMap<String,ArrayD<Variable>>)
        -> Result<(String,ArrayD<Variable>), String>;
    // fn build for every Layer type
}

// TODO TEST THIS
impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReluLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        // Reshape inputs
        // TODO work on removing
        // let layer_input = reshape_layer(layer_input, &self.input_shape);

        let out = layer_input;

        // TODO RELU unsupported for now. Must design relu function that takes in array instead of vectors
        let out = relu_array(api, out, self.n_bits - 1);

        Ok((self.outputs[0].clone(), out))
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(String, ArrayD<Variable>), String> {
        // Extract input
        let layer_input = input.get(&self.inputs[0])
            .ok_or("Missing input tensor")?
            .clone();

        // Convert weights
        let weights = load_array_constants(api, &self.weights);

        // Convert bias to ArrayD
        let bias: ArrayD<Variable> = vec1_to_arrayd(
            self.bias.iter().map(|x| load_circuit_constant(api, *x)).collect()
        );

        // Scaling
        let scale_factor = 1 << self.scaling;
        let alpha_two_v = api.mul(self.two_v as u32, scale_factor as u32);

        // Get shape
        let in_shape = layer_input.shape().iter().map(|&x| x as u32).collect::<Vec<_>>();

        // Convolution
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

        Ok((self.outputs[0].clone(), out))
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GemmLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(String, ArrayD<Variable>), String> {
        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        let mut input_array = layer_input
            .into_dimensionality::<Ix2>()
            .map_err(|_| format!("Expected 2D input for layer {}", self.name))?;
        let mut weights_array = load_array_constants(api, &self.weights)
        .into_dimensionality::<Ix2>()
            .map_err(|_| format!("Expected 2D input for layer {}", self.name))?;;

        input_array = check_and_apply_transpose_array(input_array, self.transa, "transa", "Gemm", &self.name);
        weights_array = check_and_apply_transpose_array(weights_array, self.transb, "transb", "Gemm", &self.name);

        let bias_array = load_array_constants(api, &self.bias);

        // Sanity check alpha and beta
        check_alpha_beta(self.alpha, "alpha", "Gemm", &self.name);
        check_alpha_beta(self.beta, "beta", "Gemm", &self.name);

        // Matrix multiplication and bias addition
        let mut result = matrix_multiplication(api, input_array.into_dyn(), weights_array.into_dyn());
        result = matrix_addition(api, result, bias_array);

        api.display("3", result[[0, 0]]);

        let mut out_array = result.into_dyn(); // back to ArrayD<Variable>
        if self.is_rescale {
            let k = CIRCUITPARAMS.scaling as usize;
            let s = self.v_plus_one.checked_sub(1).expect("v_plus_one must be at least 1");
            out_array = rescale_array(api, out_array, k, s, self.is_relu);
        }

        Ok((self.outputs[0].clone(), out_array))
    }
}

fn check_alpha_beta(val: f32, var_name: &str, layer_type: &str, layer_name: &str) {
    if val != 1.0{
        panic!("Only {} = 1 is currently supported for {} layers: {}", var_name, layer_type, layer_name);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: check_and_apply_transpose_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a transpose to a 2D array if the transpose flag is set.
///
/// # Arguments
/// - `matrix`: A 2D array (`Array2<T>`) to conditionally transpose.
/// - `flag`: 0 means no transpose, 1 means transpose.
/// - `var_name`: Name of the transpose flag variable (for error messages).
/// - `layer_type`: Name of the layer type (for error messages).
/// - `layer_name`: Name of the layer instance (for error messages).
///
/// # Panics
/// Panics if `flag` is not 0 or 1.
pub fn check_and_apply_transpose_array<T: Clone>(
    matrix: Array2<T>,
    flag: usize,
    var_name: &str,
    layer_type: &str,
    layer_name: &str,
) -> Array2<T> {
    match flag {
        0 => matrix,
        1 => matrix.reversed_axes(), // transpose
        other => panic!(
            "Unsupported {} value {} in {} layer: {}",
            var_name, other, layer_type, layer_name
        ),
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReshapeLayer {
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




fn onnx_flatten<T>(array: ArrayD<T>, axis: usize) -> ArrayD<T> {
    let shape = array.shape();
    let dim0 = shape[..axis].iter().product::<usize>();
    let dim1 = shape[axis..].iter().product::<usize>();

    array.into_shape_with_order(IxDyn(&[dim0, dim1])).unwrap()
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for FlattenLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        let reshape_axis = self.axis.clone();
        let layer_input = input.get(&self.inputs[0]).unwrap();

        let out = onnx_flatten(layer_input.clone(), reshape_axis);

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

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for MaxPoolLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {

        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        // TODO work on removing

        // let layer_input = reshape_layer(layer_input, &self.input_shape);
        let x = arrayd_to_vec4(layer_input);

        let ceil_mode = false; // or make configurable
        let (kernel, strides, dilation, out_shape, pads) = setup_maxpooling_2d(
            &self.padding, &self.kernel_shape, &self.strides,
            &self.dilation, ceil_mode, &self.input_shape,
        );

        let output = maxpooling_2d::<C, Builder>(
            api, &x, &kernel, &strides, &dilation, &out_shape,
            &self.input_shape, &pads, self.shift_exponent,
        );

        let out = vec4_to_arrayd(output);
        Ok((self.outputs[0].clone(), out))
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

        match layer.op_type.as_str() {
            "Conv" => {
                let params = layer.params.clone().unwrap();
                eprintln!("Conv");
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
                    group: vec![get_param_or_default(&layer.name, &"group", &params, Some(&1))],
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
                    bias: get_w_or_b(&w_and_b_map, &layer.inputs[2]),
                    is_relu: is_relu,
                    v_plus_one: V_PLUS_ONE,
                    two_v: TWO_V,
                    alpha_two_v: alpha_two_v,
                    is_rescale: is_rescale.clone(),
                    scaling: CIRCUITPARAMS.scaling.into(), // TODO: Becomes scaling_in?
                    input_shape: expected_shape.to_vec(),
                    alpha: get_param_or_default(&layer.name, &"alpha", &params, Some(&1.0)),
                    beta: get_param_or_default(&layer.name, &"beta", &params, Some(&1.0)),
                    transa: get_param_or_default(&layer.name, &"transA", &params, Some(&0)),
                    transb: get_param_or_default(&layer.name, &"transB", &params, Some(&0)),
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
            "MaxPool" => {
                let params = layer.params.clone().unwrap();
                let expected_shape = match shapes_map.get(&layer.inputs[0]) {
                Some(s) => s,
                None => panic!("Missing shape for MaxPool input {}", layer.name),
                };

                let maxpool = MaxPoolLayer {
                    name: layer.name.clone(),
                    kernel_shape: get_param(&layer.name, "kernel_shape", &params),
                    strides: get_param(&layer.name, "strides", &params),
                    dilation: get_param(&layer.name, "dilations", &params),
                    padding: get_param(&layer.name, "pads", &params),
                    input_shape: expected_shape.clone(),
                    shift_exponent: N_BITS - 1,
                    inputs: layer.inputs.to_vec(),
                    outputs: outputs,
                };
                layers.push(Box::new(maxpool));
            }
            "Flatten" => {
                let params = layer.params.clone().unwrap();
                
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let output_shape = shapes_map.get(&layer.outputs.to_vec()[0]);
                let flatten = FlattenLayer {
                    name: layer.name.clone(),
                    input_shape: expected_shape.to_vec(),
                    inputs: layer.inputs.to_vec(),
                    outputs: layer.outputs.to_vec(),
                    axis: get_param_or_default(&layer.name, &"axis", &params, Some(&1))
                };
                layers.push(Box::new(flatten));
            }
            // Just in case the relu is not following a gemm or conv layer 
            "Relu" =>{
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let relu = ReluLayer{
                    name: layer.name.clone(),
                    index: i,
                    input_shape: expected_shape.to_vec(),
                    inputs: layer.inputs.to_vec(),
                    outputs: outputs,
                    n_bits: N_BITS,
                };
                layers.push(Box::new(relu));
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

pub fn value_to_arrayd<T>(value: Value) -> Result<ArrayD<T>, String>
where
    T: Clone + FromJsonNumber + 'static,
{
    match value {
        // Single number -> 0D array (scalar)
        Value::Number(n) => {
            let val = T::from_json_number(&n).ok_or("Invalid number for target type")?;
            Ok(ArrayD::from_elem(IxDyn(&[]), val))
        }
        
        // Array -> determine dimensions and convert
        Value::Array(arr) => {
            if arr.is_empty() {
                return Ok(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap());
            }
            
            // Get the shape by walking through the nested structure
            let shape = get_array_shape(&Value::Array(arr.clone()))?;
            
            // Flatten all the data
            let mut data = Vec::new();
            flatten_to_typed_data::<T>(&Value::Array(arr), &mut data)?;
            
            // Create the ArrayD
            ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| format!("Shape error: {}", e))
        }
        
        _ => Err("Expected number or array".to_string()),
    }
}

fn flatten_to_typed_data<T>(value: &Value, data: &mut Vec<T>) -> Result<(), String>
where
    T: Clone + FromJsonNumber,
{
    match value {
        Value::Number(n) => {
            let val = T::from_json_number(n).ok_or("Invalid number for target type")?;
            data.push(val);
            Ok(())
        }
        Value::Array(arr) => {
            for item in arr {
                flatten_to_typed_data(item, data)?;
            }
            Ok(())
        }
        _ => Err("Expected number or array".to_string()),
    }
}

pub trait FromJsonNumber {
    fn from_json_number(n: &serde_json::Number) -> Option<Self>
    where
        Self: Sized;
}

impl FromJsonNumber for f64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_f64()
    }
}

impl FromJsonNumber for f32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_f64().map(|x| x as f32)
    }
}

impl FromJsonNumber for i32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_i64().map(|x| x as i32)
    }
}

impl FromJsonNumber for i64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_i64()
    }
}

impl FromJsonNumber for u32 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_u64().map(|x| x as u32)
    }
}

impl FromJsonNumber for u64 {
    fn from_json_number(n: &serde_json::Number) -> Option<Self> {
        n.as_u64()
    }
}

fn get_array_shape(value: &Value) -> Result<Vec<usize>, String> {
    let mut shape = Vec::new();
    let mut current = value;
    
    loop {
        match current {
            Value::Array(arr) => {
                if arr.is_empty() {
                    shape.push(0);
                    break;
                }
                shape.push(arr.len());
                current = &arr[0];
            }
            Value::Number(_) => break,
            _ => return Err("Invalid array structure".to_string()),
        }
    }
    
    Ok(shape)
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

fn get_w_or_b<I: DeserializeOwned + Clone + FromJsonNumber + 'static>(
    w_and_b_map: &HashMap<String, ONNXLayer>,
    weights_input: &String,
) -> ArrayD<I> {
    let weights_tensor_option = match w_and_b_map.get(weights_input) {
        Some(tensor) => tensor.tensor.clone(),
        None => panic!("🚨 ModelError - missing weights and biases: {}", weights_input),
    };

    match weights_tensor_option {
        Some(tensor_json) => {
            // Unwrap "value" if tensor is an object with that key
            let inner_value = match &tensor_json {
                Value::Object(map) if map.contains_key("value") => map.get("value").cloned().unwrap(),
                _ => tensor_json.clone(),
            };
            

            eprintln!(
                "🔍 Attempting to parse tensor for '{}': type = {}",
                weights_input,
                match &inner_value {
                    Value::Array(_) => "Array",
                    Value::Object(_) => "Object",
                    Value::Number(_) => "Number",
                    Value::String(_) => "String",
                    _ => "Other",
                }
            );
            return value_to_arrayd(inner_value).unwrap();
            
        }
        None => panic!("🚨 ModelError - missing tensor in expected weights/bias: {}", weights_input),
    }
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