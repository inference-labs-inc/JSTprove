use binius_frontend::{CircuitBuilder, Wire};

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    b: &CircuitBuilder,
    input: &[Wire],
    weights: &[Wire],
    bias: &[Wire],
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride: usize,
    rescale_bits: u32,
) -> (Vec<Wire>, usize, usize) {
    let oh = (h - kernel) / stride + 1;
    let ow = (w - kernel) / stride + 1;
    let zero = b.add_constant_64(0);

    assert_eq!(input.len(), in_channels * h * w);
    assert_eq!(weights.len(), out_channels * in_channels * kernel * kernel);
    assert_eq!(bias.len(), out_channels);

    let mut output = Vec::with_capacity(out_channels * oh * ow);

    for oc in 0..out_channels {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias[oc];
                for ic in 0..in_channels {
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            let input_val = input[ic * h * w + iy * w + ix];
                            let weight_val = weights[oc * in_channels * kernel * kernel
                                + ic * kernel * kernel
                                + ky * kernel
                                + kx];
                            let (_hi, lo) = b.smul(input_val, weight_val);
                            let shifted = b.sar(lo, rescale_bits);
                            let (sum, _carry) = b.iadd_cin_cout(acc, shifted, zero);
                            acc = sum;
                        }
                    }
                }
                output.push(acc);
            }
        }
    }

    (output, oh, ow)
}
