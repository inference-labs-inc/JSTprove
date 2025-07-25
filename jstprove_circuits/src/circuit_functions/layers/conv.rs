use std::cmp::{max, min};
use ndarray::{ArrayD, IxDyn};

use crate::circuit_functions::layers::gemm::dot;

use super::super::utils::quantization::{
    rescale_array,
};

use super::super::utils::helper::IntoTensor;

use expander_compiler::frontend::*;

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
    input_shape: &Vec<u32>,
    bias: &ArrayD<Variable>,  // assumed shape: [out_channels]
    h_out: u32,
    shape_0: usize, shape_1: usize, shape_2: u32, shape_3: usize,
) -> ArrayD<Variable> {
    let shape = IxDyn(&[shape_0, shape_1, shape_2 as usize, shape_3]);
    let mut res = ArrayD::default(shape.clone());

    if bias.len() > 0 {
        for b in 0..shape_1 {
            let bias_val = bias[[b]];
            for n in 0..shape_0 {
                for h in 0..shape_2 as usize {
                    for w in 0..shape_3 {
                        res[[n, b, h, w]] = bias_val;
                    }
                }
            }
        }
    } else {
        let zero = api.constant(0);
        for idx in res.indexed_iter_mut() {
            *idx.1 = zero;
        }
    }

    res
}

/// Check if two 4 have matching shape
fn have_matching_shapes(
    x: &Vec<Vec<Vec<Vec<Variable>>>>,
    y: &Vec<Vec<Vec<Vec<Variable>>>>,
) -> bool {
    // Check if the outermost vectors (first dimension) have the same length
    if x.len() != y.len() {
        return false;
    }

    // Check if each of the second-level vectors (second dimension) have the same length
    for (x_outer, y_outer) in x.iter().zip(y.iter()) {
        if x_outer.len() != y_outer.len() {
            return false;
        }

        // Check if each of the third-level vectors (third dimension) have the same length
        for (x_mid, y_mid) in x_outer.iter().zip(y_outer.iter()) {
            if x_mid.len() != y_mid.len() {
                return false;
            }

            // Check if each of the fourth-level vectors (fourth dimension) have the same length
            for (x_innermost, y_innermost) in x_mid.iter().zip(y_mid.iter()) {
                if x_innermost.len() != y_innermost.len() {
                    return false;
                }
            }
        }
    }

    // All dimensions match
    true
}

/// Reshapes convolution output array using `ArrayD<Variable>` instead of nested Vecs.
pub fn conv_shape_4<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: ArrayD<Variable>,         // shape: [N, C, H, W]
    input_shape: &Vec<u32>,
    kernel_shape: &Vec<u32>,            // shape: [kH, kW]
    strides: &Vec<u32>,
    pads: &Vec<u32>,
    weights: &ArrayD<Variable>,         // shape: [M, C, kH, kW]
    bias: &ArrayD<Variable>,            // shape: [M]
) -> ArrayD<Variable> {
    let s_n = input_shape[0] as usize;
    let s_c = input_shape[1] as usize;
    let s_h = input_shape[2] as usize;
    let s_w = input_shape[3] as usize;

    let k_h = kernel_shape[0] as usize;
    let k_w = kernel_shape[1] as usize;
    let stride_h = strides[0] as usize;
    let stride_w = strides[1] as usize;

    let pad_top = pads[0] as usize;
    let pad_left = pads[1] as usize;
    let pad_bottom = pads[2] as usize;
    let pad_right = pads[3] as usize;

    let h_out = (s_h + pad_top + pad_bottom - k_h) / stride_h + 1;
    let w_out = (s_w + pad_left + pad_right - k_w) / stride_w + 1;

    let m = weights.shape()[0]; // number of output channels

    let mut output = ArrayD::default(IxDyn(&[s_n, m, h_out, w_out]));

    for n in 0..s_n {
        for out_ch in 0..m {
            let bias_val = if bias.len() > 0 {
                bias[[out_ch]]
            } else {
                api.constant(0)
            };

            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = bias_val;
                    for c in 0..s_c {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;
                                if ih >= pad_top && ih < s_h + pad_top && iw >= pad_left && iw < s_w + pad_left {
                                    let ih_adj = ih - pad_top;
                                    let iw_adj = iw - pad_left;
                                    let input_val = input_arr[[n, c, ih_adj, iw_adj]];
                                    let weight_val = weights[[out_ch, c, kh, kw]];
                                    let mul = api.mul(input_val, weight_val);
                                    acc = api.add(acc, mul);
                                }
                            }
                        }
                    }
                    output[[n, out_ch, oh, ow]] = acc;
                }
            }
        }
    }

    output
}

/// Flatten vector and perform dot product
fn flatten_and_perform_dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    img: Vec<Vec<Vec<Vec<Variable>>>>,
    w_: Vec<Vec<Vec<Vec<Variable>>>>,
) -> Variable {
    // let flattened_img: &Vec<Vec<Variable>> = &img[0][0];
    // let flattened_w_: &Vec<Vec<Variable>> =  &w_[0][0];

    let flattened_img: Vec<Variable> = img
        .iter() // Iterate over the first dimension
        .flat_map(|x| x.iter()) // Iterate over the second dimension
        .flat_map(|y| y.iter()) // Iterate over the third dimension
        .flat_map(|z| z.iter())
        .cloned() // Iterate over the fourth dimension
        .collect();
    let flattened_w_: Vec<Variable> = w_
        .iter() // Iterate over the first dimension
        .flat_map(|x| x.iter()) // Iterate over the second dimension
        .flat_map(|y| y.iter()) // Iterate over the third dimension
        .flat_map(|z| z.iter())
        .cloned() // Iterate over the fourth dimension
        .collect();
    let s = dot(api, flattened_img, flattened_w_);
    s
}

/// Run of convolution
pub fn conv_4d_run<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: ArrayD<Variable>,
    weights: ArrayD<Variable>,
    bias: ArrayD<Variable>,  // formerly Vec<Variable>
    dilations_in: &Vec<u32>,
    kernel_shape_in: &Vec<u32>,
    pads_in: &Vec<u32>,
    strides_in: &Vec<u32>,
    input_shape_in: &Vec<u32>,
    scaling_in: u64,
    group_in: &Vec<u32>,
    quantized: bool,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
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
        let shift_exponent = v_plus_one.checked_sub(1)
            .expect("v_plus_one must be at least 1");
        rescale_array(api, out, scaling_exponent, shift_exponent, is_relu)
    } else {
        out
    }
}
