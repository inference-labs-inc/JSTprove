use core::panic;
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
use ndarray::Dimension;
use ndarray::{ ArrayD, IxDyn};
use serde::Deserialize;
use serde_json::Value;
use gravy_circuits::circuit_functions::quantization::run_if_quantized_2d;

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
    layer_output_shapes: Vec<Vec<usize>>,
}

#[derive(Deserialize, Clone, Debug)]
// #[serde(deny_unknown_fields)] // Optional: remove this to allow unknowns
struct Layer {
    name: String,
    r#type: String, // Layer Type
    activation: String,
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
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../weights/torch_generic_circuit_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref WEIGHTS_INPUT: WeightsData = {
        let x: WeightsData =
            serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        x
    };
}

declare_circuit!(Circuit {
    input_arr: [PublicVariable], // shape (m, n)
    outputs: [PublicVariable],   // shape (m, k)
});

/*
ConvLayer, ReshapeLayer, FCLayer
*/
struct ConvLayer {
    index: usize,
    weights: Vec<Vec<Vec<Vec<i64>>>>,
    bias: Vec<i64>,
    strides: Vec<u32>,
    kernel_shape: Vec<u32>,
    group: Vec<u32>,
    dilation: Vec<u32>,
    pads: Vec<u32>,
    input_shape: Vec<u32>,
    scaling: u64,
    is_relu: bool,
    v_plus_one: usize,
    two_v: u64,
    alpha_two_v: Variable,
    is_rescale: bool,
}

struct ReshapeLayer {
    shape: Vec<usize>,
}

struct FCLayer {
    index: usize,
    weights: Vec<Vec<Vec<i64>>>,
    bias: Vec<Vec<Vec<i64>>>,
    is_rescale: bool,
    v_plus_one: usize,
    two_v: u64,
    alpha_two_v: Variable,
    is_relu: bool,
    scaling: u64,
}

trait LayerOp<C: Config, Builder: RootAPI<C>> {
    fn apply(&self, api: &mut Builder, input: ArrayD<Variable>)
        -> Result<ArrayD<Variable>, String>;
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: ArrayD<Variable>,
    ) -> Result<ArrayD<Variable>, String> {
        eprintln!("Applying input shape CONV");
        let weights = read_4d_weights(api, &WEIGHTS_INPUT.conv_weights[self.index]);
        let bias: Vec<Variable> = WEIGHTS_INPUT.conv_bias[self.index]
            .clone()
            .into_iter()
            .map(|x| load_circuit_constant(api, x))
            .collect();
        let alpha_two_v = api.mul(self.two_v as u32, self.scaling as u32);
        eprintln!("Applying biases");

        let input = arrayd_to_vec4(input);
        eprintln!("GOT Input:");
        
        eprintln!("{:?}", &self.dilation);
        eprintln!("{:?}", &self.kernel_shape);
        eprintln!("{:?}", &self.pads);
        eprintln!("{:?}", &self.strides);
        eprintln!("{:?}", &self.input_shape);
        eprintln!("{:?}", &self.scaling);
        eprintln!("{:?}", &self.group);
        eprintln!("{:?}",Variable::from(self.alpha_two_v));
        
        let out = conv_4d_run(
            api,
            input,
            weights,
            bias,
            &self.dilation,
            &self.kernel_shape,
            &self.pads,
            &self.strides,
            &self.input_shape,
            self.scaling,
            &self.group,
            self.is_rescale,
            self.v_plus_one,
            self.two_v,
            alpha_two_v,
            self.is_relu,
        );
        
        eprint!("PRINTING INDEX");
        Ok(vec4_to_arrayd(out))
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for FCLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: ArrayD<Variable>,
    ) -> Result<ArrayD<Variable>, String> {
        let weights = read_2d_weights(api, &WEIGHTS_INPUT.fc_weights[self.index]);
        let bias = read_2d_weights(api, &WEIGHTS_INPUT.fc_bias[self.index]);
        let mut out_2d = arrayd_to_vec2(input); // Potential point of failure, check with Jonathan

        let alpha_two_v = api.mul(self.two_v as u32, self.scaling as u32);

        out_2d = matrix_multplication_naive2(api, out_2d, weights);
        out_2d = matrix_addition_vec(api, out_2d, bias);
        api.display("3", out_2d[0][0]);
        eprintln!("GOT display:");
        out_2d = run_if_quantized_2d(api, WEIGHTS_INPUT.scaling, self.is_rescale, out_2d, self.v_plus_one, self.two_v, alpha_two_v, self.is_relu);
        eprintln!("GOT output:");
        let out = vec2_to_arrayd(out_2d);
        eprintln!("Finished");
        Ok(out)
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReshapeLayer {
    /*
    TO-DO: Implement permanent implementation, currently temp solution
     */
    fn apply(
        &self,
        api: &mut Builder,
        input: ArrayD<Variable>,
    ) -> Result<ArrayD<Variable>, String> {
        let reshape_shape: [usize; 2] = [1,12544];
        let mut input = input;
        input = input
            .into_shape_with_order(IxDyn(&reshape_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions.");

        Ok(input)
    }
}

type BoxedDynLayer<C, B> = Box<dyn LayerOp<C, B>>;

fn build_layers<C: Config, Builder: RootAPI<C>>() -> Vec<Box<dyn LayerOp<C, Builder>>> {
    let mut layers: Vec<BoxedDynLayer<C, Builder>> = vec![];
    let mut conv_layer_num = 0;
    let mut fc_layer_num = 0;
    const N_BITS: usize = 32;
    const V_PLUS_ONE: usize = N_BITS;
    const TWO_V: u64 = 1 << (V_PLUS_ONE - 1);
    let alpha_two_v_usize: usize = ((1 << WEIGHTS_INPUT.scaling) * TWO_V) as usize;


    for (i, layer) in WEIGHTS_INPUT.layers.iter().enumerate() {
        let is_relu = if i + 1 < WEIGHTS_INPUT.layers.len() {
            WEIGHTS_INPUT.layers[i + 1].activation.starts_with("ReLU")
        } else {
            false
        };

        let is_rescale = !WEIGHTS_INPUT
            .not_rescale_layers
            .iter()
            .any(|l| l.eq_ignore_ascii_case(&layer.name));
        match layer.r#type.as_str() {
            "conv" => {
                let conv = ConvLayer {
                    index: conv_layer_num,
                    weights: WEIGHTS_INPUT.conv_weights[conv_layer_num].clone(),
                    bias: WEIGHTS_INPUT.conv_bias[conv_layer_num].clone(),
                    strides: WEIGHTS_INPUT.conv_strides[conv_layer_num].clone(),
                    kernel_shape: WEIGHTS_INPUT.conv_kernel_shape[conv_layer_num].clone(),
                    group: WEIGHTS_INPUT.conv_group[conv_layer_num].clone(),
                    dilation: WEIGHTS_INPUT.conv_dilation[conv_layer_num].clone(),
                    pads: WEIGHTS_INPUT.conv_pads[conv_layer_num].clone(),
                    input_shape: WEIGHTS_INPUT.conv_input_shape[conv_layer_num].clone(),
                    scaling: WEIGHTS_INPUT.scaling,
                    is_relu,
                    v_plus_one: N_BITS,
                    two_v: TWO_V,
                    /*
                    TODO - api.mul instead of hard-coding multiplication
                     */
                    alpha_two_v: Variable::from(alpha_two_v_usize),
                    is_rescale,
                };

                layers.push(Box::new(conv));
                conv_layer_num += 1;

            }
            "reshape" => {
                /*
                   TODO - Implement permanent solution for what reshape layer needs like
                */
                let reshape = ReshapeLayer {
                    shape: WEIGHTS_INPUT.layer_input_shapes[i].clone(),
                };
                layers.push(Box::new(reshape));
            }
            "fc" => {
                let fc = FCLayer {
                    index: fc_layer_num,
                    weights: WEIGHTS_INPUT.fc_weights.clone(),
                    bias: WEIGHTS_INPUT.fc_bias.clone(),
                    is_relu,
                    v_plus_one: V_PLUS_ONE,
                    two_v: TWO_V,
                    alpha_two_v: Variable::from(alpha_two_v_usize),
                    is_rescale,
                    scaling: WEIGHTS_INPUT.scaling,
                };
                layers.push(Box::new(fc));
                fc_layer_num += 1;
            }
            other => panic!("Unsupported layer type: {}", other),
        }
    }

    layers
}
// Memorization, in a better place
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let mut out = vec1_to_arrayd(self.input_arr.clone());
        let layers = build_layers::<C, Builder>();
        
        assert!(WEIGHTS_INPUT.layers.len() > 0);

        for (i, layer) in layers.iter().enumerate() {

            let expected_shape = &WEIGHTS_INPUT.layer_input_shapes[i];

            let raw_dim = out.raw_dim();
            let dim_view = raw_dim.as_array_view();
            let current_shape: &[usize] = dim_view.as_slice().unwrap();

            if current_shape != expected_shape {
                let reshape_shape = &WEIGHTS_INPUT.layer_input_shapes[i];

                out = out
                    .into_shape_with_order(IxDyn(&reshape_shape))
                    .expect("Shape mismatch: Cannot reshape into the given dimensions");
                print!("Check 1");
            }

            /*
            ERROR IS OCCURING HERE
             */
            out = layer
                .apply(api, out)
                .expect(&format!("Failed to apply layer {}", i));
        }

        eprint!("Pre-vec1");
        let flatten_shape: Vec<usize> = vec![WEIGHTS_INPUT.output_shape.iter().product()];

        out = out
            .into_shape_with_order(IxDyn(&flatten_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions"); 
        let output = arrayd_to_vec1(out);
        eprint!("Pre-vec2");
        for (j, _) in self.outputs.iter().enumerate() {
            api.display("out1", self.outputs[j]);
            api.display("out2", output[j]);
            api.assert_is_equal(self.outputs[j], output[j]);
            // api.assert_is_different(self.outputs[j], 1);
        }
    }
}

impl ConfigurableCircuit for Circuit<Variable> {
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

        let input_dims: &[usize] = &[WEIGHTS_INPUT.input_shape.iter().product()];

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

    handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(&mut file_reader);
}
