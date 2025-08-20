/// Standard library imports
use std::{cmp::{max, min}, collections::HashMap};

/// External crate imports
use ndarray::{s, ArrayD};

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

use crate::circuit_functions::utils::{constants::{DILATION, GROUP, KERNEL_SHAPE, PADS, STRIDES}, onnx_model::{extract_params_and_expected_shape, get_param, get_param_or_default, get_w_or_b}};

/// Internal module imports
use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::{graph_pattern_matching::GraphPattern, quantization::rescale_array, tensor_ops::{load_array_constants, load_circuit_constant}}};

// -------- Struct --------

#[derive(Debug)]
#[allow(dead_code)]
pub struct ConvLayer {
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

// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let is_relu = match self.optimization_pattern.name{
                    "Conv+Relu" => true,
                    _ => false
                };

        let layer_input = input.get(&self.inputs[0])
        .ok_or_else(|| panic!("Missing input {}", self.inputs[0].clone())).unwrap()
    .clone();

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

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {

        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer);
        
        let conv = Self{
            name: layer.name.clone(),
            index: index,
            weights: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[1]),
            bias: get_w_or_b(&layer_context.w_and_b_map, &layer.inputs[2]),
            strides: get_param(&layer.name, STRIDES, &params),
            kernel_shape: get_param(&layer.name, KERNEL_SHAPE, &params),
            group: vec![get_param_or_default(&layer.name, GROUP, &params, Some(&1))],
            dilation: get_param(&layer.name, DILATION, &params),
            pads: get_param(&layer.name, PADS, &params),
            input_shape: expected_shape.to_vec(),
            scaling: circuit_params.scale_exponent.into(),
            optimization_pattern: optimization_pattern,
            v_plus_one: layer_context.n_bits,
            two_v: layer_context.two_v,
            alpha_two_v: layer_context.alpha_two_v,
            is_rescale: is_rescale,
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
        };
        Ok(Box::new(conv))
    }
}




/// Untested
/// Set default parameters if not set
pub fn set_default_params(
    dilations: &Vec<u32>,
    kernel_shape: &Vec<u32>,
    pads: &Vec<u32>,
    strides: &Vec<u32>,
    input_shape: &Vec<u32>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    // If dilations is empty, fill it with 1s of the appropriate length
    let mut dilations_out = dilations.clone();
    let mut kernel_shape_out = kernel_shape.clone();
    let mut pads_out = pads.clone();
    let mut strides_out = strides.clone();

    if dilations.is_empty() {
        dilations_out = vec![1; input_shape[2..].len()];
    }

    // If kernel_shape is empty, fill it with W.shape()[2..]
    if kernel_shape.is_empty() {
        kernel_shape_out = input_shape[2..].to_vec();
    }

    // If pads is empty, fill it with 0s, twice the length of X.shape()[2..]
    if pads.is_empty() {
        let shape_len = input_shape[2..].len();
        pads_out = vec![0; shape_len * 2];
    }

    // If strides is empty, fill it with 1s of the appropriate length
    if strides.is_empty() {
        strides_out = vec![1; input_shape[2..].len()];
    }
    (dilations_out, kernel_shape_out, pads_out, strides_out)
}

/// Check if any parameters are not suitable
pub fn not_yet_implemented_conv(input_shape: &Vec<u32>, group: &Vec<u32>, dilations: &Vec<u32>) {
    if input_shape[1] != input_shape[1] * group[0] || input_shape[0] % group[0] != 0 {
        panic!("Shape inconsistencies");
    }
    if group[0] > 1 {
        panic!("Not yet implemented for group > 1");
    }
    if (dilations[0] != 1) || (dilations.iter().min() != dilations.iter().max()) {
        panic!("Not yet implemented for this dilation");
    }
    if input_shape.len() == 3 {
        panic!("Not yet implemented for Input shape length 3");
    }
    if input_shape.len() == 5 {
        panic!("Not yet implemented for Input shape length 5");
    }
}

/// Setup the initial array for convolution. Incorporates bias
fn conv_shape_4_setup_res<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    bias: &ArrayD<Variable>,
    shape_0: usize,
    shape_1: usize,
    shape_2: usize,
    shape_3: usize,
) -> ArrayD<Variable> {
    let shape = vec![shape_0, shape_1, shape_2, shape_3];

    if bias.len() > 0 {
        // Create array filled with bias values
        let mut res = ArrayD::from_elem(shape, bias[[0]]);

        for i in 0..shape_0 {
            for j in 0..shape_1 {
                for k in 0..shape_2 as usize {
                    for l in 0..shape_3 {
                        res[[i, j, k, l]] = bias[[j]];
                    }
                }
            }
        }
        res
    } else {
        // Create array filled with zeros
        let zero = api.constant(0);
        ArrayD::from_elem(shape, zero)
    }
}

fn have_matching_shapes(x: &ArrayD<Variable>, y: &ArrayD<Variable>) -> bool {
    x.shape() == y.shape()
}

pub fn conv_shape_4<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: ArrayD<Variable>,
    input_shape: &Vec<u32>,
    kernel_shape: &Vec<u32>,
    strides: &Vec<u32>,
    pads: &Vec<u32>,
    weights: &ArrayD<Variable>,
    bias: &ArrayD<Variable>,
) -> ArrayD<Variable> {
    if pads.len() < 4 {
        panic!("Pads is not long enough");
    }
    let s_n = *input_shape.get(0).expect("Missing input shape index 0") as usize;
    let s_c = *input_shape.get(1).expect("Missing input shape index 1") as usize;
    let s_h = *input_shape.get(2).expect("Missing input shape index 2") as usize;
    let s_w = *input_shape.get(3).expect("Missing input shape index 3") as usize;

    // # M, C_group, kH, kW = W.shape
    let kh = *kernel_shape.get(0).expect("Missing kernel shape index 0") as usize;
    let kw = *kernel_shape.get(1).expect("Missing kernel shape index 1") as usize;

    let sth = *strides.get(0).expect("Missing strides index 0") as usize;
    let stw = *strides.get(1).expect("Missing strides index 1") as usize;
    let pad_0 = *pads.get(0).expect("Missing pads 0 index") as usize;
    let pad_1 = *pads.get(1).expect("Missing pads 1 index") as usize;
    let pad_2 = *pads.get(2).expect("Missing pads 2 index") as usize;
    let pad_3 = *pads.get(3).expect("Missing pads 3 index") as usize;

    //Need to make sure there is no overflow/casting issues here. Dont think there should be
    let h_out = ((s_h - kh + pad_0 + pad_2) / sth) + 1;
    let w_out = ((s_w - kw + pad_1 + pad_3) / stw) + 1;

    let h0 = pads.get(0).expect("Missing pads 0 index");
    let w0 = pads.get(1).expect("Missing pads 1 index");

    let oh = -1 * (kh % 2) as i32;
    let ow = -1 * (kw % 2) as i32;

    let bh = -(*h0 as i32);
    let bw = -(*w0 as i32);

    let eh = h_out * sth;
    let ew = w_out * stw;

    let mut res = conv_shape_4_setup_res(
        api,
        bias,
        s_n,
        weights.shape()[0],
        h_out as usize,
        w_out as usize,
    );

    for n in 0..s_n {
        for nw in 0..weights.shape()[0] {
            for c in 0..s_c {
                let w = weights
                    .slice(s![nw..nw + 1, c..c + 1, .., ..])
                    .into_dyn()
                    .to_owned();

                for io in (bh..eh as i32).step_by(sth as usize) {
                    let hr = (io - bh) / sth as i32;
                    if hr >= h_out as i32 {
                        continue;
                    }
                    let i = io + kh as i32 % 2;
                    let ih1 = max(0, i + oh) as usize;
                    let ih2 = min(i + oh + kh as i32, s_h as i32) as usize;

                    for jo in (bw..ew as i32).step_by(stw as usize) {
                        let wr = (jo - bw) / stw as i32;
                        if wr >= w_out as i32 {
                            continue;
                        }
                        let j = jo + kw as i32 % 2;
                        let iw1 = max(0, j + ow) as usize;
                        let iw2 = min(j + ow + kw as i32, s_w as i32) as usize;

                        let n_usize = n as usize;

                        let img = input_arr
                            .slice(s![n_usize..n_usize + 1, c..c + 1, ih1..ih2, iw1..iw2])
                            .into_dyn()
                            .to_owned();

                        if !have_matching_shapes(&img, &w) {
                            let jh1 = max(-oh - i, 0) as usize;
                            let jh2 = min(kh as i32, kh as i32 + s_h as i32 - (i + oh + kh as i32))
                                as usize;

                            let jw1 = max(-ow - j, 0) as usize;
                            let jw2 = min(kw as i32, kw as i32 + s_w as i32 - (j + ow + kw as i32))
                                as usize;
                            let w_ = w
                                .slice(s![0..1, 0..1, jh1..jh2, jw1..jw2])
                                .into_dyn()
                                .to_owned();

                            if !have_matching_shapes(&w_.clone(), &img) {
                                panic!("Unexpected shape!! img != w_, oh={oh}, ow={ow}, i={i}, j={j}, kh={kh}, kw={kw}, sH={s_h}, sW={s_w}, sth={sth}, stw={stw}")
                            }
                            // TODO check if bias is empty
                            let s = flatten_and_perform_dot(api, &img, &w_);
                            res[[n as usize, nw, hr as usize, wr as usize]] =
                                api.add(s, res[[n as usize, nw, hr as usize, wr as usize]]);
                        } else {
                            // TODO check if bias is empty
                            let s = flatten_and_perform_dot(api, &img, &w);
                            res[[n as usize, nw, hr as usize, wr as usize]] =
                                api.add(s, res[[n as usize, nw, hr as usize, wr as usize]]);
                        }
                    }
                }
            }
        }
    }

    res
}

/// Flatten vector and perform dot product
fn flatten_and_perform_dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    img: &ArrayD<Variable>,
    w_: &ArrayD<Variable>,
) -> Variable {
    let flattened_img: ArrayD<&Variable> =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[img.len()]), img.iter().collect()).unwrap();
    let flattened_w_: ArrayD<&Variable> =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[w_.len()]), w_.iter().collect()).unwrap();

    let mut sum = api.constant(0);
    for (a, b) in flattened_img.iter().zip(flattened_w_.iter()) {
        let prod = api.mul(*a, *b);
        sum = api.add(sum, prod);
    }
    sum
}
/// Run of convolution
pub fn conv_4d_run<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: ArrayD<Variable>,
    weights: ArrayD<Variable>,
    bias: ArrayD<Variable>, // formerly Vec<Variable>
    dilations_in: &Vec<u32>,
    kernel_shape_in: &Vec<u32>,
    pads_in: &Vec<u32>,
    strides_in: &Vec<u32>,
    input_shape_in: &Vec<u32>,
    scaling_in: u64,
    group_in: &Vec<u32>,
    quantized: bool,
    v_plus_one: usize,
    _two_v: T,
    _alpha_two_v: Variable,
    is_relu: bool,
) -> ArrayD<Variable> {
    let (dilations, kernel_shape, pads, strides) = set_default_params(
        dilations_in,
        kernel_shape_in,
        pads_in,
        strides_in,
        input_shape_in,
    );
    not_yet_implemented_conv(input_shape_in, group_in, &dilations);

    let out = conv_shape_4(
        api,
        input_arr,
        input_shape_in,
        &kernel_shape,
        &strides,
        &pads,
        &weights,
        &bias,
    );

    if quantized {
        let scaling_exponent = scaling_in as usize;
        let shift_exponent = v_plus_one
            .checked_sub(1)
            .expect("v_plus_one must be at least 1");
        rescale_array(api, out, scaling_exponent, shift_exponent, is_relu)
    } else {
        out
    }
}
