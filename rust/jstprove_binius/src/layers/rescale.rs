use binius_frontend::{CircuitBuilder, Wire};

pub fn rescale(b: &CircuitBuilder, x: Wire, shift_bits: u32) -> Wire {
    b.sar(x, shift_bits)
}

pub fn rescale_batch(b: &CircuitBuilder, xs: &[Wire], shift_bits: u32) -> Vec<Wire> {
    xs.iter().map(|&x| rescale(b, x, shift_bits)).collect()
}
