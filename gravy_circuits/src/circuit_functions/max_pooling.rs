use expander_compiler::frontend::*;
use crate::circuit_functions::core_operations::{unconstrained_max, assert_is_max, MaxAssertionContext};

pub fn setup_maxpooling_2d_params(
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,

) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let padding = match padding.len() {
        0 => vec![0; 4],
        1 => vec![padding[0]; 4],
        2 => vec![padding[0], padding[1], padding[0], padding[1]],
        3 => panic!("Padding cannot have 3 elements"),
        4 => padding.clone(),
        _ => panic!("Padding must have between 1 and 4 elements"),
    };

    let kernel_shape  = match kernel_shape.len() {
        1 => vec![kernel_shape[0]; 2],
        2 => vec![kernel_shape[0], kernel_shape[1]],
        _ => panic!("Kernel shape must have between 1 and 2 elements"),
    };

    let dilation = match dilation.len() {
        0 => vec![1; 2],
        1 => vec![dilation[0]; 2],
        2 => vec![dilation[0], dilation[1]],
        _ => panic!("Dilation must have between 1 and 2 elements"),
    };

    let strides = match strides.len() {
        0 => vec![1; 2],
        1 => vec![strides[0]; 2],
        2 => vec![strides[0], strides[1]],
        _ => panic!("Strides must have between 1 and 2 elements"),
    };
    return (padding, kernel_shape, strides, dilation);
}

pub fn setup_maxpooling_2d(
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
    ceil_mode: bool,
    x_shape: &Vec<usize>,


)-> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<[usize; 2]>) {
    let (pads, kernel_shape, strides, dilation) = setup_maxpooling_2d_params(&padding, &kernel_shape, &strides, &dilation);
    
    let n_dims = kernel_shape.len();

    // Create new_pads as Vec<[usize; 2]>
    let mut new_pads = vec![[0; 2]; n_dims];
    for i in 0..n_dims {
        new_pads[i] = [pads[i], pads[i + n_dims]];
    }


    let input_spatial_shape = &x_shape[2..];
    let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

    for i in 0..input_spatial_shape.len() {
        let total_padding = new_pads[i][0] + new_pads[i][1];
        let kernel_extent = (kernel_shape[i] - 1) * dilation[i] + 1;
        let numerator = input_spatial_shape[i] + total_padding - kernel_extent;

        if ceil_mode {
            let mut out_dim = (numerator as f64 / strides[i] as f64).ceil() as usize + 1;
            let need_to_reduce =
                (out_dim - 1) * strides[i] >= input_spatial_shape[i] + new_pads[i][0];
            if need_to_reduce {
                out_dim -= 1;
            }
            output_spatial_shape[i] = out_dim;
        } else {
            output_spatial_shape[i] = (numerator / strides[i]) + 1;
        }
    }
    return (kernel_shape, strides, dilation, output_spatial_shape, new_pads);
}


// shift_exponent, true 
pub fn maxpooling_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Vec<Vec<Vec<Vec<Variable>>>>,
    // padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
    output_spatial_shape: &Vec<usize>,
    x_shape: &Vec<usize>,
    new_pads: &Vec<[usize; 2]>,
    shift_exponent: usize,
)-> Vec<Vec<Vec<Vec<Variable>>>>{
    let global_pooling = false;
    let batch = x_shape[0];
    let channels = x_shape[1];
    let height = x_shape[2];
    let width = if kernel_shape.len() > 1 { x_shape[3] } else { 1 };

    let pooled_height = output_spatial_shape[0];
    let pooled_width = if kernel_shape.len() > 1 {
        output_spatial_shape[1]
    } else {
        1
    };

    let y_dims = [batch, channels, pooled_height, pooled_width];
    let y_size = y_dims.iter().product();
    let y = vec![api.constant(0); y_size];

    let total_channels = batch * channels;

    let stride_h = if global_pooling { 1 } else { strides[0] };
    let stride_w = if global_pooling {
        1
    } else if strides.len() > 1 {
        strides[1]
    } else {
        1
    };

    let x_step = height * width;
    let y_step = pooled_height * pooled_width;

    let dilation_h = dilation[0];
    let dilation_w = if dilation.len() > 1 {
        dilation[1]
    } else {
        1
    };
    let x_data: Vec<Variable> =  x.iter()
    .flat_map(|z| z.iter())
    .flat_map(|z| z.iter())
    .flat_map(|z| z.iter())
    .copied()
    .collect();
    let mut y_data = y;

    let context = MaxAssertionContext::new(api, shift_exponent);
    
    for c in 0..total_channels {
        let x_d = c * x_step;
        let y_d = c * y_step;

        for ph in 0..pooled_height {
            let hstart = ph as isize * stride_h as isize - new_pads[0][0] as isize;
            let hend = hstart + (kernel_shape[0] * dilation_h) as isize;

            for pw in 0..pooled_width {
                let wstart = pw as isize * stride_w as isize - new_pads[1][0] as isize;
                let wend = wstart + (kernel_shape[1] * dilation_w) as isize;

                let pool_index = ph * pooled_width + pw;
                let mut values: Vec<Variable> = Vec::new();

                for h in (hstart..hend).step_by(dilation_h) {
                    if h < 0 || h >= height as isize {
                        continue;
                    }
                    for w in (wstart..wend).step_by(dilation_w) {
                        if w < 0 || w >= width as isize {
                            continue;
                        }

                        let input_index = (h as usize) * width + (w as usize);
                        let val = x_data[x_d + input_index];
                        values.push(val);

                    }
                }
                if values.len() != 0 {                    
                    let max = unconstrained_max(api, &values);
                    assert_is_max(api, &context, &values);
                    y_data[y_d + pool_index] = max;
                }
            }
        }
    }
    reshape_4d(&y_data, y_dims)
}

fn reshape_4d(flat: &Vec<Variable>, dims: [usize; 4]) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let mut out =  vec![vec![vec![vec![Variable::default(); w]; h]; c]; n];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let flat_index = ((ni * c + ci) * h + hi) * w + wi;
                    out[ni][ci][hi][wi] = flat[flat_index];
                }
            }
        }
    }
    out
}