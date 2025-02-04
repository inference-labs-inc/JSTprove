use std::cmp::{max, min};

use expander_compiler::frontend::*;

//Untested
pub fn set_default_params(
    dilations: &Vec<u32>,
    kernel_shape: &Vec<u32>,
    pads: &Vec<u32>,
    strides: &Vec<u32>,
    input_shape: &Vec<u32>
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

pub fn not_yet_implemented_conv(input_shape: &Vec<u32>, group: &Vec<u32>, dilations: &Vec<u32>, ){
    if input_shape[1] != input_shape[1] * group[0] || input_shape[0] % group[0] != 0 {
        panic!("Shape inconsistencies");
    }
    if group[0] > 1{
        panic!("Not yet implemented for group > 1");
    }
    if (dilations[0] != 1) || (dilations.iter().min() != dilations.iter().max()){
        panic!("Not yet implemented for this dilation");
    }
    if input_shape.len() == 3{
        panic!("Not yet implemented for Input shape length 3");
    }
    if input_shape.len() == 5{
        panic!("Not yet implemented for Input shape length 5");
    }
}



fn conv_shape_4_setup_res<C: Config>(api: &mut API<C>, input_shape: &Vec<u32>, bias: &Vec<Variable>, h_out: u32, shape_0: usize, shape_1: usize, shape_2: u32, shape_3: usize) ->  Vec<Vec<Vec<Vec<Variable>>>>{
    let mut res: Vec<Vec<Vec<Vec<Variable>>>> = Vec::with_capacity(input_shape[0] as usize);
    if !bias.is_empty(){
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
    }
    else{
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

pub fn conv_shape_4<C: Config>(api: &mut API<C>, X: Vec<Vec<Vec<Vec<Variable>>>>, input_shape: &Vec<u32>, kernel_shape: &Vec<u32>, strides: &Vec<u32>, pads: &Vec<u32>, weights: &Vec<Vec<Vec<Vec<Variable>>>>, bias: &Vec<Variable>) -> Vec<Vec<Vec<Vec<Variable>>>>{
    if pads.len() < 4{
        panic!("Pads is not long enough");
    }
    let sN = input_shape.get(0).expect("Missing input shape index 0");
    let sC = input_shape.get(1).expect("Missing input shape index 1");
    let sH = input_shape.get(2).expect("Missing input shape index 2");
    let sW = input_shape.get(3).expect("Missing input shape index 3");

    // # M, C_group, kH, kW = W.shape
    let kh = kernel_shape.get(0).expect("Missing kernel shape index 0");
    let kw = kernel_shape.get(1).expect("Missing kernel shape index 1");

    let sth = strides.get(0).expect("Missing strides index 0");
    let stw = strides.get(1).expect("Missing strides index 1");

    //Need to make sure there is no overflow/casting issues here. Dont think there should be
    let h_out = (sH - kh + pads[0] + pads[2] / sth) + 1;
    let w_out = ((sW - kw + pads[1] + pads[3]) / stw) + 1;

    let h0 = pads.get(0).expect("Missing pads 0 index");
    let w0 = pads.get(1).expect("Missing pads 1 index");

    let oh = -1 * (kh % 2) as i32;
    let ow = -1 * (kw % 2) as i32;

    let bh = -(*h0 as i32);
    let bw = -(*w0 as i32);

    let eh = h_out * sth;
    let ew = w_out * stw;

    let (shape_0, shape_1, shape_2, shape_3) = (input_shape[0] as usize, weights.len(), h_out, w_out as usize);

    let res = conv_shape_4_setup_res(api, input_shape, bias, h_out, shape_0, shape_1, shape_2, shape_3);

    for n in 0..*sN{
        for nw in 0..weights.len(){
            for c in 0..*sC{
                let c_usize: usize = c as usize;
                let w: Vec<Vec<Vec<Vec<Variable>>>> = weights[nw..nw + 1]
                                .iter()
                                .map(|row| row[c_usize..c_usize + 1].to_vec())  // Take the subvector for each row
                                .collect();
                for io in (bh..eh as i32).step_by(*sth as usize) {
                    let hr = (io - bh) / *sth as i32;
                    if hr >= h_out as i32{
                        continue;
                    }
                    let i = io + *kh as i32 % 2;// Be careful with where the modulus is 

                    let ih1 = max(0, i + oh) as usize;
                    let ih2 = min(i + oh + *kh as i32, *sH as i32) as usize;

                    for jo in (bw..ew as i32).step_by(*stw as usize){
                        let wr = (jo - bw) / *stw as i32;
                        if wr >= w_out as i32{
                            continue;
                        }
                        let j = jo + *kw as i32 % 2;// Be careful with where the modulus is 
                        let iw1 = max(0, j + ow) as usize;
                        let iw2 = min(j + ow + *kw as i32, *sW as i32) as usize;

                        let n_usize = n as usize;
                        let img = X[n_usize..n_usize+1]
                        .iter()
                        .map(|x| x[c_usize..c_usize+1]
                            .iter()
                            .map(|y| y[ih1..ih2]
                                .iter()
                                .map(|z| z[iw1..iw2].to_vec()) // Convert slice to Vec
                                .collect::<Vec<Vec<Variable>>>()
                            )
                            .collect::<Vec<Vec<Vec<Variable>>>>()
                        )
                        .collect::<Vec<Vec<Vec<Vec<Variable>>>>>();
    //                     iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
    //                     img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2]

                    }


                }
            }
        }
    }

    // for n in range(sN):
    //     for nw in range(W.shape[0]):
    //         for c in range(sC):
    //             w = W[nw : nw + 1, c : c + 1]
    //             for io in range(bh, eh, sth):
    //                 hr = (io - bh) // sth
    //                 # print(hr)
    //                 if hr >= h_out:
    //                     continue
    //                 i = io + kh % 2
    //                 ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
    //                 for jo in range(bw, ew, stw):
    //                     wr = (jo - bw) // stw
    //                     if wr >= w_out:
    //                         continue
    //                     j = jo + kw % 2
    //                     iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
    //                     img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2]
    //                     if img.shape != w.shape:
    //                         jh1, jh2 = (
    //                             max(-oh - i, 0),
    //                             min(kh, kh + sH - (i + oh + kh)),
    //                         )
    //                         jw1, jw2 = (
    //                             max(-ow - j, 0),
    //                             min(kw, kw + sW - (j + ow + kw)),
    //                         )
    //                         w_ = w[:1, :1, jh1:jh2, jw1:jw2]
    //                         if img.shape != w_.shape:
    //                             raise RuntimeError(
    //                                 f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, ow={ow}, "
    //                                 f"i={i}, j={j}, kh={kh}, kw={kw}, sH={sH}, sW={sW}, sth={sth}, stw={stw}."
    //                             )
    //                         s = np.dot(img.reshape((1, -1)), w_.reshape((-1, 1)))[
    //                             0, 0
    //                         ]  # (img * w_).sum()
    //                     else:
    //                         s = np.dot(img.reshape((1, -1)), w.reshape((-1, 1)))[
    //                             0, 0
    //                         ]  # (img * w).sum()
    //                     res[n, nw, hr, wr] += s  # type: ignore
    X
}