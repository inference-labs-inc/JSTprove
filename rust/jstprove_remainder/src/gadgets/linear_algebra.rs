use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::tensor::ShapedMLE;
use crate::padding::num_vars_for;

pub fn elementwise_add<F: Field>(
    builder: &mut CircuitBuilder<F>,
    a: &ShapedMLE<F>,
    b: &ShapedMLE<F>,
) -> ShapedMLE<F> {
    let expr = a.node.expr() + b.node.expr();
    let result = builder.add_sector(expr);
    ShapedMLE::new(result, a.shape.clone())
}

pub fn elementwise_sub<F: Field>(
    builder: &mut CircuitBuilder<F>,
    a: &ShapedMLE<F>,
    b: &ShapedMLE<F>,
) -> ShapedMLE<F> {
    let expr = a.node.expr() - b.node.expr();
    let result = builder.add_sector(expr);
    ShapedMLE::new(result, a.shape.clone())
}

pub fn hadamard_product<F: Field>(
    builder: &mut CircuitBuilder<F>,
    a: &ShapedMLE<F>,
    b: &ShapedMLE<F>,
) -> ShapedMLE<F> {
    let expr = AbstractExpression::products(vec![a.node.id(), b.node.id()]);
    let result = builder.add_sector(expr);
    ShapedMLE::new(result, a.shape.clone())
}

pub fn scale_by_constant<F: Field>(
    builder: &mut CircuitBuilder<F>,
    a: &ShapedMLE<F>,
    scalar: F,
) -> ShapedMLE<F> {
    let expr = AbstractExpression::scaled(a.node.expr(), scalar);
    let result = builder.add_sector(expr);
    ShapedMLE::new(result, a.shape.clone())
}

pub fn matmul<F: Field>(
    builder: &mut CircuitBuilder<F>,
    a: &ShapedMLE<F>,
    a_rows_vars: usize,
    a_cols_vars: usize,
    b: &ShapedMLE<F>,
    b_rows_vars: usize,
    b_cols_vars: usize,
) -> ShapedMLE<F> {
    let result = builder.add_matmult_node(
        &a.node,
        (a_rows_vars, a_cols_vars),
        &b.node,
        (b_rows_vars, b_cols_vars),
    );

    let out_rows = 1 << a_rows_vars;
    let out_cols = 1 << b_cols_vars;
    ShapedMLE::new(result, vec![out_rows, out_cols])
}

pub fn matmul_with_dims<F: Field>(
    builder: &mut CircuitBuilder<F>,
    a: &ShapedMLE<F>,
    a_rows: usize,
    a_cols: usize,
    b: &ShapedMLE<F>,
    b_rows: usize,
    b_cols: usize,
) -> ShapedMLE<F> {
    let a_rows_vars = num_vars_for(a_rows);
    let a_cols_vars = num_vars_for(a_cols);
    let b_rows_vars = num_vars_for(b_rows);
    let b_cols_vars = num_vars_for(b_cols);

    assert_eq!(a_cols_vars, b_rows_vars, "inner dimensions must match");

    matmul(
        builder,
        a,
        a_rows_vars,
        a_cols_vars,
        b,
        b_rows_vars,
        b_cols_vars,
    )
}
