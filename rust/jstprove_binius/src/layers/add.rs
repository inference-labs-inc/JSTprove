use binius_frontend::{CircuitBuilder, Wire};

pub fn add_elementwise(b: &CircuitBuilder, a: &[Wire], rhs: &[Wire]) -> Vec<Wire> {
    assert_eq!(a.len(), rhs.len());
    let zero = b.add_constant_64(0);
    a.iter()
        .zip(rhs.iter())
        .map(|(&x, &y)| {
            let (sum, _carry) = b.iadd_cin_cout(x, y, zero);
            sum
        })
        .collect()
}
