#[allow(unused_imports)]
/// Standard library imports
use core::panic;
use std::collections::HashMap;

use jstprove_circuits::circuit_functions::layers::constant::ConstantLayer;
use jstprove_circuits::circuit_functions::layers::flatten::FlattenLayer;
use jstprove_circuits::circuit_functions::layers::layer_ops::LayerOp;
use jstprove_circuits::circuit_functions::layers::reshape::ReshapeLayer;
use jstprove_circuits::circuit_functions::utils::graph_pattern_matching::{optimization_skip_layers, GraphPattern, PatternMatcher};
/// External crate imports
use lazy_static::lazy_static;
use ndarray::{ArrayD, Ix1, IxDyn};
use serde::Deserialize;
use serde_json::Value;

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

/// Internal crate imports
use jstprove_circuits::circuit_functions::utils::onnx_model::{
    collect_all_shapes,
    get_param,
    get_param_or_default,
    get_w_or_b,
};
use jstprove_circuits::circuit_functions::utils::shaping::get_inputs;
use jstprove_circuits::circuit_functions::layers::conv::ConvLayer;
use jstprove_circuits::circuit_functions::layers::gemm::GemmLayer;
use jstprove_circuits::circuit_functions::layers::maxpool::MaxPoolLayer;
use jstprove_circuits::circuit_functions::layers::relu::ReluLayer;

use jstprove_circuits::circuit_functions::utils::tensor_ops::get_nd_circuit_inputs;

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
    _scale_base: u32,
    scaling: u32,
    rescale_config: HashMap<String, bool>
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

    let mut skip_next_layer: HashMap<String, bool>  = HashMap::new();

    let inputs = &ARCHITECTURE.inputs;

    // TODO havent figured out how but this can maybe go in build layers?
    let shapes_map: HashMap<String, Vec<usize>> = collect_all_shapes(&ARCHITECTURE.architecture, inputs);

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
                let conv = ConvLayer::new(
                    layer.name.clone(),
                    i,
                    get_w_or_b(&w_and_b_map, &layer.inputs[1]),
                    get_w_or_b(&w_and_b_map, &layer.inputs[2]),
                    get_param(&layer.name, &"strides", &params),
                    get_param(&layer.name, &"kernel_shape", &params),
                    vec![get_param_or_default(&layer.name, &"group", &params, Some(&1))],
                    get_param(&layer.name, &"dilations", &params),
                    get_param(&layer.name, &"pads", &params),
                    expected_shape.to_vec(),
                    CIRCUITPARAMS.scaling.into(),
                    optimization_pattern,
                    N_BITS,
                    TWO_V,
                    alpha_two_v,
                    *is_rescale,
                    layer.inputs.to_vec(),
                    outputs
                );

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
                let reshape = ReshapeLayer::new(
                    layer.name.clone(),
                    expected_shape.to_vec(),
                    layer.inputs.to_vec(),
                    layer.outputs.to_vec(),
                    get_param_or_default(&layer.name, &shape_name, &params, output_shape)
                );
                layers.push(Box::new(reshape));
            }
            "Gemm" => {

                let params = layer.params.clone().unwrap();
                // We can move this inside the layer op
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let gemm = GemmLayer::new(
                    layer.name.clone(),
                    i,
                    get_w_or_b(&w_and_b_map, &layer.inputs[1]),
                    get_w_or_b(&w_and_b_map, &layer.inputs[2]),
                    is_rescale.clone(),
                    V_PLUS_ONE,
                    TWO_V,
                    alpha_two_v,
                    optimization_pattern,
                    CIRCUITPARAMS.scaling.into(), // TODO: Becomes scaling_in?
                    expected_shape.to_vec(),
                    get_param_or_default(&layer.name, &"alpha", &params, Some(&1.0)),
                    get_param_or_default(&layer.name, &"beta", &params, Some(&1.0)),
                    get_param_or_default(&layer.name, &"transA", &params, Some(&0)),
                    get_param_or_default(&layer.name, &"transB", &params, Some(&0)),
                    layer.inputs.to_vec(),
                    outputs,
                );
                layers.push(Box::new(gemm));
            }
            "Constant" => {
                let constant = ConstantLayer::new(
                    layer.name.clone(),
                    get_param(&layer.name, &"value", &layer.params.clone().unwrap()),
                    outputs
                );
                layers.push(Box::new(constant));
            }
            "MaxPool" => {
                let params = layer.params.clone().unwrap();
                let expected_shape = match shapes_map.get(&layer.inputs[0]) {
                Some(s) => s,
                None => panic!("Missing shape for MaxPool input {}", layer.name),
                };

                let maxpool = MaxPoolLayer::new(
                    layer.name.clone(),
                    get_param(&layer.name, "kernel_shape", &params),
                    get_param(&layer.name, "strides", &params),
                    get_param(&layer.name, "dilations", &params),
                    get_param(&layer.name, "pads", &params),
                    expected_shape.clone(),
                    N_BITS - 1,
                    layer.inputs.to_vec(),
                    outputs,
                );
                layers.push(Box::new(maxpool));
            }
            "Flatten" => {
                let params = layer.params.clone().unwrap();
                
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let flatten = FlattenLayer::new(
                    layer.name.clone(),
                    get_param_or_default(&layer.name, &"axis", &params, Some(&1)),
                    expected_shape.to_vec(),
                    layer.inputs.to_vec(),
                    layer.outputs.to_vec(),
                );
                layers.push(Box::new(flatten));
            }
            // Just in case the relu is not following a gemm or conv layer 
            "Relu" =>{
                let expected_shape = match shapes_map.get(&layer.inputs[0]){
                    Some(input_shape) => input_shape,
                    None => panic!("Error getting output shape for layer {}", layer.name)
                };
                let relu = ReluLayer::new(
                    layer.name.clone(),
                    i,
                    expected_shape.to_vec(),
                    layer.inputs.to_vec(),
                    outputs,
                    N_BITS,
                );
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
            result.0.into_iter().for_each(|key| {
                // out.insert(key, Arc::clone(&value)); Depending on memory constraints here
                out.insert(key, result.1.clone());
            });

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