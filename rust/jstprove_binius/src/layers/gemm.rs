use binius_frontend::{CircuitBuilder, Wire};

pub fn gemm(
    b: &CircuitBuilder,
    input: &[Wire],
    weights: &[Wire],
    bias: &[Wire],
    in_features: usize,
    out_features: usize,
    rescale_bits: u32,
) -> Vec<Wire> {
    assert_eq!(input.len(), in_features);
    assert_eq!(weights.len(), out_features * in_features);
    assert_eq!(bias.len(), out_features);

    let zero = b.add_constant_64(0);
    let mut output = Vec::with_capacity(out_features);

    for o in 0..out_features {
        let mut acc = bias[o];
        for i in 0..in_features {
            let w = weights[o * in_features + i];
            let (_hi, lo) = b.smul(input[i], w);
            let shifted = b.sar(lo, rescale_bits);
            let (sum, _carry) = b.iadd_cin_cout(acc, shifted, zero);
            acc = sum;
        }
        output.push(acc);
    }

    output
}
