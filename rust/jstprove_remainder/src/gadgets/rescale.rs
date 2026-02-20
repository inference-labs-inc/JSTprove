use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::tensor::ShapedMLE;

pub struct RescaleContext {
    pub scaling_exponent: u32,
    pub alpha: i64,
    pub shift_exponent: usize,
    pub offset: i64,
}

impl RescaleContext {
    pub fn new(scaling_exponent: u32, shift_exponent: usize) -> Self {
        let alpha = 1i64 << scaling_exponent;
        let offset = 1i64 << shift_exponent;
        Self {
            scaling_exponent,
            alpha,
            shift_exponent,
            offset,
        }
    }
}

pub fn constrained_rescale<F: Field>(
    builder: &mut CircuitBuilder<F>,
    input: &ShapedMLE<F>,
    quotient_hint: &ShapedMLE<F>,
    remainder_hint: &ShapedMLE<F>,
) -> ShapedMLE<F> {
    let alpha = F::from(1u64 << 18);
    let constraint_expr = input.node.expr()
        - quotient_hint.node.expr() * alpha
        - remainder_hint.node.expr();
    let constraint = builder.add_sector(constraint_expr);
    builder.set_output(&constraint);

    quotient_hint.clone()
}

pub fn compute_rescale(value: i64, alpha: i64, offset: i64) -> (i64, i64) {
    let shifted_dividend = alpha * offset + value;
    let q_shifted = shifted_dividend / alpha;
    let remainder = shifted_dividend - alpha * q_shifted;
    let quotient = q_shifted - offset;
    (quotient, remainder)
}

pub fn compute_rescale_array(
    values: &[i64],
    alpha: i64,
    offset: i64,
) -> (Vec<i64>, Vec<i64>) {
    let mut quotients = Vec::with_capacity(values.len());
    let mut remainders = Vec::with_capacity(values.len());
    for &v in values {
        let (q, r) = compute_rescale(v, alpha, offset);
        quotients.push(q);
        remainders.push(r);
    }
    (quotients, remainders)
}
