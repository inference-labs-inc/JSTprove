use std::cmp::{max, min};

use crate::matrix_computation::dot;
use crate::quantization::run_if_quantized_4d;
use expander_compiler::frontend::*;

//Untested
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

fn conv_shape_4_setup_res<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_shape: &Vec<u32>,
    bias: &Vec<Variable>,
    h_out: u32,
    shape_0: usize,
    shape_1: usize,
    shape_2: u32,
    shape_3: usize,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let mut res: Vec<Vec<Vec<Vec<Variable>>>> = Vec::with_capacity(input_shape[0] as usize);
    if !bias.is_empty() {
        for _ in 0..shape_0 {
            let mut dim2 = Vec::with_capacity(shape_1);

            for j in 0..shape_1 {
                let mut dim3 = Vec::with_capacity(h_out as usize);
                for _ in 0..shape_2 {
                    dim3.push(vec![bias[j]; shape_3 as usize]); // Fill with zeros
                }
                dim2.push(dim3);
            }
            res.push(dim2);
        }
        // res[:, :, :, :] = B.reshape((1, -1, 1, 1))  # type: ignore
    } else {
        let zero = api.constant(0);
        for _ in 0..shape_0 {
            let mut dim2 = Vec::with_capacity(shape_1);
            for _ in 0..shape_1 {
                let mut dim3 = Vec::with_capacity(shape_2 as usize);
                for _ in 0..shape_2 {
                    dim3.push(vec![zero; shape_3 as usize]); // Fill with zeros
                }
                dim2.push(dim3);
            }
            res.push(dim2);
        }
    }
    res
}

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

pub fn conv_shape_4<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: Vec<Vec<Vec<Vec<Variable>>>>,
    input_shape: &Vec<u32>,
    kernel_shape: &Vec<u32>,
    strides: &Vec<u32>,
    pads: &Vec<u32>,
    weights: &Vec<Vec<Vec<Vec<Variable>>>>,
    bias: &Vec<Variable>,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    if pads.len() < 4 {
        panic!("Pads is not long enough");
    }
    let s_n = input_shape.get(0).expect("Missing input shape index 0");
    let s_c = input_shape.get(1).expect("Missing input shape index 1");
    let s_h = input_shape.get(2).expect("Missing input shape index 2");
    let s_w = input_shape.get(3).expect("Missing input shape index 3");

    // # M, C_group, kH, kW = W.shape
    let kh = kernel_shape.get(0).expect("Missing kernel shape index 0");
    let kw = kernel_shape.get(1).expect("Missing kernel shape index 1");

    let sth = strides.get(0).expect("Missing strides index 0");
    let stw = strides.get(1).expect("Missing strides index 1");

    //Need to make sure there is no overflow/casting issues here. Dont think there should be
    let h_out = ((s_h - kh + pads[0] + pads[2]) / sth) + 1;
    let w_out = ((s_w - kw + pads[1] + pads[3]) / stw) + 1;

    let h0 = pads.get(0).expect("Missing pads 0 index");
    let w0 = pads.get(1).expect("Missing pads 1 index");

    let oh = -1 * (kh % 2) as i32;
    let ow = -1 * (kw % 2) as i32;

    let bh = -(*h0 as i32);
    let bw = -(*w0 as i32);

    let eh = h_out * sth;
    let ew = w_out * stw;

    let (shape_0, shape_1, shape_2, shape_3) = (
        input_shape[0] as usize,
        weights.len(),
        h_out,
        w_out as usize,
    );

    let mut res = conv_shape_4_setup_res(
        api,
        input_shape,
        bias,
        h_out,
        shape_0,
        shape_1,
        shape_2,
        shape_3,
    );
    for n in 0..*s_n {
        for nw in 0..weights.len() {
            for c in 0..*s_c {
                let c_usize: usize = c as usize;
                let w: Vec<Vec<Vec<Vec<Variable>>>> = weights[nw..nw + 1]
                    .iter()
                    .map(|row| row[c_usize..c_usize + 1].to_vec()) // Take the subvector for each row
                    .collect();
                for io in (bh..eh as i32).step_by(*sth as usize) {
                    let hr = (io - bh) / *sth as i32;
                    if hr >= h_out as i32 {
                        continue;
                    }
                    let i = io + *kh as i32 % 2; // Be careful with where the modulus is
                    let ih1 = max(0, i + oh) as usize;
                    let ih2 = min(i + oh + *kh as i32, *s_h as i32) as usize;


                    for jo in (bw..ew as i32).step_by(*stw as usize) {
                        let wr = (jo - bw) / *stw as i32;
                        if wr >= w_out as i32 {
                            continue;
                        }
                        let j = jo + *kw as i32 % 2; // Be careful with where the modulus is
                        let iw1 = max(0, j + ow) as usize;
                        let iw2 = min(j + ow + *kw as i32, *s_w as i32) as usize;

                        let n_usize = n as usize;
                        
                        let img = input_arr[n_usize..n_usize + 1]
                            .iter()
                            .map(|x| {
                                x[c_usize..c_usize + 1]
                                    .iter()
                                    .map(|y| {
                                        y[ih1..ih2]
                                            .iter()
                                            .map(|z| z[iw1..iw2].to_vec()) // Convert slice to Vec
                                            .collect::<Vec<Vec<Variable>>>()
                                    })
                                    .collect::<Vec<Vec<Vec<Variable>>>>()
                            })
                            .collect::<Vec<Vec<Vec<Vec<Variable>>>>>();

                        if !have_matching_shapes(&img, &w) {
                            let jh1 = max(-oh - i, 0) as usize;
                            let jh2 =
                                min(*kh as i32, *kh as i32 + *s_h as i32 - (i + oh + *kh as i32))
                                    as usize;

                            let jw1 = max(-ow - j, 0) as usize;
                            let jw2 =
                                min(*kw as i32, *kw as i32 + *s_w as i32 - (j + ow + *kw as i32))
                                    as usize;

                            let w_: Vec<Vec<Vec<Vec<Variable>>>> = w[0..1]
                                .iter() // Slice the first dimension (up to index 1)
                                .map(|x| {
                                    x[0..1]
                                        .iter() // Slice the second dimension (up to index 1)
                                        .map(|y| {
                                            y[jh1..jh2]
                                                .iter() // Slice the third dimension from `jh1` to `jh2`
                                                .map(|z| {
                                                    z[jw1..jw2].to_vec() // Slice the fourth dimension from `jw1` to `jw2`
                                                })
                                                .collect::<Vec<Vec<Variable>>>()
                                            // Collect into a Vec<Vec<Variable>>
                                        })
                                        .collect::<Vec<Vec<Vec<Variable>>>>() // Collect into a Vec<Vec<Vec<Variable>>>
                                })
                                .collect::<Vec<Vec<Vec<Vec<Variable>>>>>(); // Collect into a Vec<Vec<Vec<Vec<Variable>>>>

                            if !have_matching_shapes(&w_, &img) {
                                panic!("Unexpected shape!! img != w_, oh={oh}, ow={ow}, i={i}, j={j}, kh={kh}, kw={kw}, sH={s_h}, sW={s_w}, sth={sth}, stw={stw}")
                            }
                            let s = flatten_and_perform_dot(api, img, w_);
                            // api.display("res pre sum", res[n as usize][nw][hr as usize][wr as usize]);
                            res[n as usize][nw][hr as usize][wr as usize] =
                                api.add(s, res[n as usize][nw][hr as usize][wr as usize]);
                        } else {
                            let s = flatten_and_perform_dot(api, img, w.clone());
                            res[n as usize][nw][hr as usize][wr as usize] =
                                api.add(s, res[n as usize][nw][hr as usize][wr as usize]);
                        }
                    }
                }
            }
        }
    }

    res
}

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

pub fn conv_4d_run<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: Vec<Vec<Vec<Vec<Variable>>>>,
    weights: Vec<Vec<Vec<Vec<Variable>>>>,
    bias: Vec<Variable>,
    dilations_in: &Vec<u32>,
    kernel_shape_in: &Vec<u32>,
    pads_in: &Vec<u32>,
    strides_in: &Vec<u32>,
    input_shape_in: &Vec<u32>,
    scaling_in: u64,
    group_in: &Vec<u32>,
    quantized: bool,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let (dilations, kernel_shape, pads, strides) = set_default_params(
        dilations_in,
        kernel_shape_in,
        pads_in,
        strides_in,
        input_shape_in,
    );
    not_yet_implemented_conv(input_shape_in, group_in, &dilations);
    let out: Vec<Vec<Vec<Vec<Variable>>>> = conv_shape_4(
        api,
        input_arr,
        input_shape_in,
        &kernel_shape,
        &strides,
        &pads,
        &weights,
        &bias,
    );

    run_if_quantized_4d(
        api,
        scaling_in,
        quantized,
        out,
        v_plus_one,
        two_v,
        alpha_two_v,
        is_relu,
    )
}
