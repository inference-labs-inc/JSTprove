#[allow(unused_imports)]
/// Standard library imports
use core::panic;
use std::collections::{HashMap, HashSet};

/// External crate imports
use lazy_static::lazy_static;
use ndarray::{Array2, ArrayD, Dimension, Ix1, Ix2, IxDyn};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

/// Internal crate imports
use jstprove_circuits::circuit_functions::utils::json_array::{
    value_to_arrayd,
    FromJsonNumber,
};

use jstprove_circuits::circuit_functions::utils::onnx_model::{
    collect_all_shapes,
    get_param,
    get_param_or_default,
    get_w_or_b,
};

use jstprove_circuits::circuit_functions::utils::shaping::{
    get_inputs,
    onnx_flatten,
};

use jstprove_circuits::circuit_functions::layers::conv::conv_4d_run;

use jstprove_circuits::circuit_functions::layers::gemm::{
    matrix_addition,
    matrix_multiplication,
};

use jstprove_circuits::circuit_functions::layers::maxpool::{
    maxpooling_2d,
    setup_maxpooling_2d,
};

use jstprove_circuits::circuit_functions::layers::relu::relu_array;

use jstprove_circuits::circuit_functions::utils::quantization::rescale_array;

use jstprove_circuits::circuit_functions::utils::tensor_ops::{
    get_nd_circuit_inputs,
    load_array_constants,
    load_circuit_constant,
};

use jstprove_circuits::io::io_reader::{FileReader, IOReader};

use jstprove_circuits::runner::main_runner::{ConfigurableCircuit, handle_args};

use jstprove_circuits::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};


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

// #[derive(Deserialize, Clone, Debug)]
// struct ONNXLayer{
//     id: usize,
//     name: String,
//     op_type: String,
//     inputs: Vec<String>,
//     outputs: Vec<String>,
//     shape: HashMap<String, Vec<usize>>,
//     tensor: Option<Value>,
//     params: Option<Value>,
//     opset_version_number: i16,
// }

// #[derive(Deserialize, Clone, Debug)]
// struct ONNXIO{
//     name: String,
//     elem_type: i16,
//     shape: Vec<usize>
// }

#[derive(Deserialize, Clone)]
struct InputData {
    input: Value,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Value,
}

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../python/models/weights/onnx_generic_circuit_weights.json");

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
    optimization_pattern: GraphPattern,
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
    optimization_pattern: GraphPattern,
    scaling: u64,
    input_shape: Vec<usize>,
    alpha: f32,
    beta: f32,
    transa: usize,
    transb: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
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

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReluLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        eprintln!("{:?}", self);
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
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        let is_relu = match self.optimization_pattern.name{
                    "Conv+Relu" => true,
                    _ => false
                };

        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        // Reshape inputs
        // TODO work on removing
        // let layer_input = reshape_layer(layer_input, &self.input_shape);

        // Convert weights
        let weights = load_array_constants(api, &self.weights);

        let bias = self.bias.mapv(|x| load_circuit_constant(api, x));
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
            is_relu,
        );

        Ok((self.outputs[0].clone(), out))
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GemmLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(String,ArrayD<Variable>), String> {
        let is_relu = match self.optimization_pattern.name{
                    "Gemm+Relu" => true,
                    _ => false
                };

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
            out_array = rescale_array(api, out_array, k, s, is_relu);
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
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(String, ArrayD<Variable>), String> {
        let layer_input = input.get(&self.inputs[0]).unwrap().clone();
        let shape = layer_input.shape();
        assert_eq!(shape.len(), 4, "Expected 4D input for max pooling, got shape: {:?}", shape);

        let ceil_mode = false;
        let (kernel, strides, dilation, out_shape, pads) = setup_maxpooling_2d(
            &self.padding,
            &self.kernel_shape,
            &self.strides,
            &self.dilation,
            ceil_mode,
            &self.input_shape,
        );

        let output = maxpooling_2d::<C, Builder>(
            api,
            &layer_input,
            &kernel,
            &strides,
            &dilation,
            &out_shape,
            &self.input_shape,
            &pads,
            self.shift_exponent,
        );

        Ok((self.outputs[0].clone(), output))
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

    // Make a Hashmap where the key is the inputs to a layer, the value must be a tuple of layer name, layer type.
    // I have the layers of an onnx model. I have certain optimizations based on combining certain layers in the computation graph
    // I want to figure out a way to identify these patterns in the graph should they appear, so that I can mark them off
    //  as I proceed through the graph layers later. Can you help me with this please?


    
    let mut skip_next_layer: HashMap<String, bool>  = HashMap::new();


    // let skip_future_layers = Set();

    let inputs = &ARCHITECTURE.inputs;

    // TODO havent figured out how but this can maybe go in build layers?
    let shapes_map: HashMap<String, Vec<usize>> = collect_all_shapes(&ARCHITECTURE.architecture, inputs);
    // TODO should account for multiple inputs


    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&ARCHITECTURE.architecture);

    


    for (i, layer) in ARCHITECTURE.architecture.iter().enumerate() {
        /*
            Track layer combo optimizations
         */

        if *skip_next_layer.get(&layer.name).unwrap_or(&false){
            continue
        }
        let outputs = layer.outputs.to_vec();

        let optimization_pattern_match = opt_patterns_by_layername.get(&layer.name);
        let (optimization_pattern, outputs, layers_to_skip) =  match optimization_skip_layers(optimization_pattern_match, outputs.clone()) {
            Some(opt) => opt,
            None => (GraphPattern::default(), outputs.clone(), vec![])
        };
        // Save layers to skip
        layers_to_skip.into_iter()
            .for_each(|item| {
                skip_next_layer.insert(item.to_string(), true);
            });
        /*
            End tracking layer combo optimizations
         */

        
        

        let is_rescale = match  CIRCUITPARAMS.rescale_config.get(&layer.name){
                Some(config) => config,
                None => &true
            };

        match layer.op_type.as_str() {
            "Conv" => {
                let params = layer.params.clone().unwrap();                
                // We can move this inside the layer op
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
                    optimization_pattern: optimization_pattern,
                    v_plus_one: N_BITS,
                    two_v: TWO_V,
                    alpha_two_v: alpha_two_v,
                    is_rescale: *is_rescale,
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
                // We can move this inside the layer op
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let gemm = GemmLayer {
                    name: layer.name.clone(),
                    index: i,
                    weights: get_w_or_b(&w_and_b_map, &layer.inputs[1]),
                    bias: get_w_or_b(&w_and_b_map, &layer.inputs[2]),
                    optimization_pattern: optimization_pattern,
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
// Memorization, in a better place
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Getting inputs
        let mut out = get_inputs(self.input_arr.clone(), ARCHITECTURE.inputs.clone());
        
        // let mut out = out2.remove("input").unwrap().clone();
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

        let output = output.as_slice().expect("Output not contiguous");

        eprint!("Assert outputs match");

        for (j, _) in self.outputs.iter().enumerate() {
            api.display("out1", self.outputs[j]);
            api.display("out2", output[j]);
            api.assert_is_equal(self.outputs[j], output[j]);
        }

        api.assert_is_equal(self.dummy[0], 1);
        api.assert_is_equal(self.dummy[1], 1);
        eprintln!("Outputs match");

    }
}







/*

Pattern matching of layers

*/

fn optimization_skip_layers(optimization_match: Option<&Vec<OptimizationMatch>>, outputs: Vec<String>) -> Option<(GraphPattern, Vec<String>, Vec<String>)> {
    match optimization_match {
        Some(opt) => {
            let pattern = opt[0].pattern;
            let mut new_outputs = Vec::new();
            let mut skipped_layers: Vec<String> = Vec::new();
            // Loop through all potential branches
            for opt_match in opt{
                // Assert all the patterns are the same
                assert!(pattern.name == opt_match.pattern.name);
                // Get final layer of pattern
                let layers = opt_match.layers.clone();
                let final_layer = layers[layers.len() - 1].clone();
                let first_layer = layers[0].clone();

                // Assert outputs match 
                eprintln!("{:?}", first_layer.outputs);
                eprintln!("{:?}", outputs);
                assert!(first_layer.outputs.iter().all(|item| outputs.contains(item)));
                new_outputs.extend(final_layer.outputs);
                skipped_layers.extend(opt_match.layers.iter().map(|layer| layer.name.clone()))
            }
            // Search the other way. Makes sure both sides of inequality holds
            // assert!(outputs.iter().all(|item| new_outputs.contains(item)));

            let set: HashSet<_> = new_outputs.into_iter().collect();
            let unique_new_outputs: Vec<String> = set.into_iter().collect();
            // let set: HashSet<_> = outputs.into_iter().collect();
            // let unique_old_outputs: Vec<String> = set.into_iter().collect();

            // assert!(unique_new_outputs.len() == unique_old_outputs.len());
    
            Some((pattern, unique_new_outputs, skipped_layers))
        },
        None => return None
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BranchMatchMode {
    Any,
    All,
}

fn build_input_to_layer_map<'a>(layers: &'a [ONNXLayer]) -> HashMap<&'a str, Vec<&'a ONNXLayer>> {
    let mut map: HashMap<&str, Vec<&ONNXLayer>> = HashMap::new();
    for layer in layers {
        for input in &layer.inputs {
            map.entry(input).or_default().push(layer);
        }
    }
    // eprint!("{:?}", map.keys());
    map
}

// TODO untested with actual branching
fn find_pattern_matches<'a>(
    layers: &'a [ONNXLayer],
    pattern: &GraphPattern,
    mode: BranchMatchMode,
) -> Vec<Vec<&'a ONNXLayer>> {
    let mut matches = Vec::new();
    for layer in layers {
        if layer.op_type == pattern.ops[0] {
            dfs(
                layer,
                pattern.ops,
                1,
                vec![layer],
                layers,
                &mut matches,
                mode,
            );
        }
    }
    matches
}
/*
Inputs:
    - current_layer: current position in the graph
    - ops: list of op names we're trying to match (e.g. ["Conv", "Relu"])
    - depth:  index in the pattern we're trying to match
    - path: vector of matched layers so far
    - all_matches: where completed match paths get collected
    - mode: "Any" (at least one path matches) or "All" (every branch must match)
*/
// Recursive DFS search across branches
fn dfs<'a>(
    current_layer: &'a ONNXLayer,
    ops: &[&'static str],
    depth: usize,
    path: Vec<&'a ONNXLayer>,
    layers: &'a [ONNXLayer],
    all_matches: &mut Vec<Vec<&'a ONNXLayer>>,
    mode: BranchMatchMode,
) {
    // Base case
    // Save full match if we reach the end of the pattern
    if depth == ops.len() {
        all_matches.push(path.clone());
        return;
    }

    // Only consider layers that:
    // - Those whose op matches the next step in the pattern (ops[depth])
    // - and that directly consume one of the outputs from the current layer
    let matching_next_layers: Vec<&ONNXLayer> = layers
        .iter()
        .filter(|l| {
            l.op_type == ops[depth]
                && l.inputs.iter().any(|inp| current_layer.outputs.contains(inp))
        })
        .collect();


    match mode {
        BranchMatchMode::Any => {
            // Try matching each of the next layers
            // Recurse with new layer and keep going
            // If any completes the pattern, add to all matches
            for next_layer in matching_next_layers {
                let mut new_path = path.clone();
                new_path.push(next_layer);
                dfs(
                    next_layer,
                    ops,
                    depth + 1,
                    new_path,
                    layers,
                    all_matches,
                    mode,
                );
            }
        }
        BranchMatchMode::All => {
            // If there are no next layers that match the next op — we abort early.
            if matching_next_layers.is_empty() {
                return;
            }

            let mut all_paths = vec![];
            for next_layer in matching_next_layers {
                let mut new_path = path.clone();
                new_path.push(next_layer);
                let mut sub_matches = Vec::new();
                dfs(
                    next_layer,
                    ops,
                    depth + 1,
                    new_path.clone(),
                    layers,
                    &mut sub_matches,
                    mode,
                );
                if !sub_matches.is_empty() {
                    all_paths.push(sub_matches);
                }
                // We explore every matching direct consumer
                // Recurse on each one
                // Keep only those that reach a complete match
            }

            // Only accept if all direct consumer branches found matching paths
            if all_paths.len() >= 1 && all_paths.iter().all(|paths| !paths.is_empty()) {
                for branch in all_paths {
                    for b in branch {
                        all_matches.push(b);
                    }
                }
            }
        }
    }
}

// TODO, somewhere must include priority in sequence, for example, conv relu batchnorm takes priority over conv relu
fn build_pattern_registry() -> Vec<GraphPattern> {
    vec![
        GraphPattern {
            name: "Conv+Relu".into(),
            ops: &["Conv", "Relu"],
        },
        GraphPattern {
            name: "Gemm+Relu".into(),
            ops: &["Gemm", "Relu"],
        },
    ]
}
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphPattern {
    pub name: &'static str,
    pub ops: &'static [&'static str],
}

#[derive(Debug, Clone)]
pub struct OptimizationMatch{
    pattern: GraphPattern,
    layers: Vec<ONNXLayer>
}

pub struct PatternMatcher {
    patterns: Vec<GraphPattern>,
}

impl PatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: build_pattern_registry(),
        }
    }

    pub fn run(&self, layers: &[ONNXLayer]) -> HashMap<std::string::String, Vec<OptimizationMatch>>{
        use std::time::SystemTime;
        let now = SystemTime::now();

        let mut all_matches: HashMap<String, Vec<OptimizationMatch>> = HashMap::new();
   
        for pat in &self.patterns {
            let matches = find_pattern_matches(layers, pat, BranchMatchMode::All);
            eprintln!("Pattern `{}` matched {} times", pat.name, matches.len());

            for m in matches{
                all_matches.entry(m[0].name.clone()).or_default().push(OptimizationMatch { pattern: *pat, layers: m.into_iter().cloned().collect()});
            }
            // eprintln!("{:?}", matches[0]);
        }
        eprintln!("{:?}", all_matches);

        match now.elapsed() {
            Ok(elapsed) => {
                // it prints '2'
                eprintln!("Model pattern match took: {} nano seconds", elapsed.as_nanos());
            }
            Err(e) => {
                // an error occurred!
                eprintln!("Error calculating time: {e:?}");
            }
        }
        all_matches
        // panic!("");
    }
}


/*

Pattern matching of layers

*/


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
        let data: InputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<InputData>(file_path);

        // compute the total number of inputs
        let input_dims: &[usize] = &[ARCHITECTURE
            .inputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()];

        assignment.dummy[0] = CircuitField::<C>::from(1);
        assignment.dummy[1] = CircuitField::<C>::from(1);

        // 1) get back an ArrayD<CircuitField<C>>
        let arr: ArrayD<CircuitField<C>> =
            get_nd_circuit_inputs::<C>(&data.input, input_dims);

        // 2) downcast to Ix1 and collect into a Vec
        let flat: Vec<CircuitField<C>> = arr
            .into_dimensionality::<Ix1>()
            .expect("Expected a 1-D array here")
            .to_vec();

        assignment.input_arr = flat;
        assignment
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Circuit<CircuitField<C>> {
        let data: OutputData =
            <FileReader as IOReader<Circuit<_>, C>>::read_data_from_json::<OutputData>(file_path);

        let output_dims: &[usize] = &[ARCHITECTURE
            .outputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .product()];

        let arr: ArrayD<CircuitField<C>> =
            get_nd_circuit_inputs::<C>(&data.output, output_dims);

        let flat: Vec<CircuitField<C>> = arr
            .into_dimensionality::<Ix1>()
            .expect("Expected a 1-D array here")
            .to_vec();

        assignment.outputs = flat;
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

    handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(&mut file_reader);
}