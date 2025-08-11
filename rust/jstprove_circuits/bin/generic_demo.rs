#[allow(unused_imports)]
/// Standard library imports
use core::panic;
use std::collections::HashMap;

use jstprove_circuits::circuit_functions::layers::constant::ConstantLayer;
use jstprove_circuits::circuit_functions::layers::flatten::FlattenLayer;
use jstprove_circuits::circuit_functions::layers::layer_ops::{BuildLayerContext, LayerBuilder, LayerOp};
use jstprove_circuits::circuit_functions::layers::reshape::ReshapeLayer;
use jstprove_circuits::circuit_functions::utils::graph_pattern_matching::{optimization_skip_layers, GraphPattern, PatternMatcher};
/// External crate imports
use lazy_static::lazy_static;
use ndarray::{ArrayD, Ix1, IxDyn};

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

/// Internal crate imports
use jstprove_circuits::circuit_functions::utils::onnx_model::{
    collect_all_shapes, get_param_or_default, Architecture, CircuitParams, InputData, OutputData, WANDB
};
use jstprove_circuits::circuit_functions::utils::shaping::get_inputs;
use jstprove_circuits::circuit_functions::layers::conv::ConvLayer;
use jstprove_circuits::circuit_functions::layers::gemm::GemmLayer;
use jstprove_circuits::circuit_functions::layers::maxpool::MaxPoolLayer;
use jstprove_circuits::circuit_functions::layers::relu::ReluLayer;

use jstprove_circuits::circuit_functions::utils::tensor_ops::get_nd_circuit_inputs;
use jstprove_circuits::io::io_reader::{FileReader, IOReader};
use jstprove_circuits::runner::main_runner::{ConfigurableCircuit, handle_args};
use jstprove_circuits::circuit_functions::utils::onnx_types::ONNXLayer;


type WeightsData = (Architecture, WANDB, CircuitParams);

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
    input_arr: [PublicVariable],
    outputs: [PublicVariable],
    dummy: [Variable; 2]
});


type BoxedDynLayer<C, B> = Box<dyn LayerOp<C, B>>;

fn build_layers<C: Config, Builder: RootAPI<C>>() -> Vec<Box<dyn LayerOp<C, Builder>>> {
    let mut layers: Vec<BoxedDynLayer<C, Builder>> = vec![];
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u32 = 1 << (V_PLUS_ONE - 1);
    let alpha_two_v: u64 = ((1 << CIRCUITPARAMS.scaling) * TWO_V) as u64;

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

    let layer_context = BuildLayerContext{
        w_and_b_map: w_and_b_map.clone(),
        shapes_map: shapes_map.clone(),
        n_bits: N_BITS,
        two_v: TWO_V,
        alpha_two_v: alpha_two_v
    };

    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&ARCHITECTURE.architecture);
    for (i, original_layer) in ARCHITECTURE.architecture.iter().enumerate() {
        /*
            Track layer combo optimizations
         */
        let mut layer = original_layer.clone();
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

        layer.outputs = outputs;
        /*
            End tracking layer combo optimizations
         */
        let is_rescale = match  CIRCUITPARAMS.rescale_config.get(&layer.name){
            Some(config) => config,
            None => &true
        };
        let builder = match layer.op_type.as_str() {
            "Conv"     => ConvLayer::build,
            "Reshape"  => ReshapeLayer::build,
            "Gemm"     => GemmLayer::build,
            "Constant" => ConstantLayer::build,
            "MaxPool"  => MaxPoolLayer::build,
            "Flatten"  => FlattenLayer::build,
            "Relu"     => ReluLayer::build,
            other      => panic!("Unsupported layer type: {}", other),
        };

        let built = builder(&layer, &CIRCUITPARAMS, optimization_pattern, *is_rescale, i, &layer_context)
            .unwrap();
        layers.push(built);
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