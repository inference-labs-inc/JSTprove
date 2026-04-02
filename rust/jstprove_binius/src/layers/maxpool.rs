use binius_frontend::{CircuitBuilder, Wire};

fn signed_max(b: &CircuitBuilder, a: Wire, x: Wire) -> Wire {
    let (diff, _borrow) = b.isub_bin_bout(a, x, b.add_constant_64(0));
    let a_geq = b.bnot(b.sar(diff, 63));
    b.select(a_geq, a, x)
}

pub fn maxpool2d(
    b: &CircuitBuilder,
    input: &[Wire],
    channels: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride: usize,
) -> (Vec<Wire>, usize, usize) {
    let oh = (h - kernel) / stride + 1;
    let ow = (w - kernel) / stride + 1;
    let mut output = Vec::with_capacity(channels * oh * ow);

    for c in 0..channels {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = input[c * h * w + oy * stride * w + ox * stride];
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        if ky == 0 && kx == 0 {
                            continue;
                        }
                        let iy = oy * stride + ky;
                        let ix = ox * stride + kx;
                        let val = input[c * h * w + iy * w + ix];
                        acc = signed_max(b, acc, val);
                    }
                }
                output.push(acc);
            }
        }
    }

    (output, oh, ow)
}
