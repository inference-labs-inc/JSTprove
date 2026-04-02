use binius_frontend::{CircuitBuilder, Wire};

pub fn relu(b: &CircuitBuilder, x: Wire) -> Wire {
    let sign = b.sar(x, 63);
    let mask = b.bnot(sign);
    b.band(x, mask)
}

pub fn relu_batch(b: &CircuitBuilder, xs: &[Wire]) -> Vec<Wire> {
    xs.iter().map(|&x| relu(b, x)).collect()
}
