use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::tensor::ShapedMLE;

pub struct ShiftRangeContext {
    pub shift_exponent: usize,
    pub offset: i64,
}

impl ShiftRangeContext {
    pub fn new(shift_exponent: usize) -> Self {
        let offset = 1i64 << shift_exponent;
        Self {
            shift_exponent,
            offset,
        }
    }

    pub fn shifted_range_bits(&self) -> usize {
        self.shift_exponent + 1
    }
}

pub fn constrained_max<F: Field>(
    builder: &mut CircuitBuilder<F>,
    inputs: &[ShapedMLE<F>],
    max_hint: &ShapedMLE<F>,
    deltas: &[ShapedMLE<F>],
) -> ShapedMLE<F> {
    for (i, delta) in deltas.iter().enumerate() {
        let diff = max_hint.node.expr() - inputs[i].node.expr();
        let constraint = builder.add_sector(diff - delta.node.expr());
        builder.set_output(&constraint);
    }

    if inputs.len() == 2 {
        let product = AbstractExpression::products(vec![
            deltas[0].node.id(),
            deltas[1].node.id(),
        ]);
        let zero_check = builder.add_sector(product);
        builder.set_output(&zero_check);
    }

    max_hint.clone()
}

pub fn constrained_min<F: Field>(
    builder: &mut CircuitBuilder<F>,
    inputs: &[ShapedMLE<F>],
    min_hint: &ShapedMLE<F>,
    deltas: &[ShapedMLE<F>],
) -> ShapedMLE<F> {
    for (i, delta) in deltas.iter().enumerate() {
        let diff = inputs[i].node.expr() - min_hint.node.expr();
        let constraint = builder.add_sector(diff - delta.node.expr());
        builder.set_output(&constraint);
    }

    if inputs.len() == 2 {
        let product = AbstractExpression::products(vec![
            deltas[0].node.id(),
            deltas[1].node.id(),
        ]);
        let zero_check = builder.add_sector(product);
        builder.set_output(&zero_check);
    }

    min_hint.clone()
}

pub fn constrained_clip<F: Field>(
    builder: &mut CircuitBuilder<F>,
    input: &ShapedMLE<F>,
    lo: &ShapedMLE<F>,
    hi: &ShapedMLE<F>,
    max_hint: &ShapedMLE<F>,
    min_hint: &ShapedMLE<F>,
    max_deltas: &[ShapedMLE<F>],
    min_deltas: &[ShapedMLE<F>],
) -> ShapedMLE<F> {
    let after_max = constrained_max(builder, &[input.clone(), lo.clone()], max_hint, max_deltas);
    constrained_min(builder, &[after_max, hi.clone()], min_hint, min_deltas)
}

pub fn compute_max(values: &[i64]) -> i64 {
    *values.iter().max().unwrap()
}

pub fn compute_min(values: &[i64]) -> i64 {
    *values.iter().min().unwrap()
}

pub fn compute_max_deltas(values: &[i64], max_val: i64) -> Vec<i64> {
    values.iter().map(|v| max_val - v).collect()
}

pub fn compute_min_deltas(values: &[i64], min_val: i64) -> Vec<i64> {
    values.iter().map(|v| v - min_val).collect()
}

pub fn compute_clip(value: i64, lo: i64, hi: i64) -> i64 {
    value.max(lo).min(hi)
}
