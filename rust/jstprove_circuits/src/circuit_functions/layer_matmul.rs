use expander_compiler::frontend::*;

pub fn dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    vector_a: Vec<Variable>,
    vector_b: Vec<Variable>,
) -> Variable {
    let mut row_col_product: Variable = api.constant(0);
    for k in 0..vector_a.len() {
        let element_product = api.mul(vector_a[k], vector_b[k]);
        row_col_product = api.add(row_col_product, element_product);
    }
    row_col_product
}

/// Naive Matrix addition with vectors
pub fn matrix_addition_vec<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new(); // or [[Variable::default(); N]; M]
    for (i, row_a) in matrix_a.iter().enumerate() {
        let mut row_out: Vec<Variable> = Vec::new();
        for (j, value) in row_a.iter().enumerate() {
            row_out.push(api.add(value, matrix_b[i][j]));
        }
        out.push(row_out);
    }
    out
}

/// Matrix multiplciation of vector of vectors naive version 2
pub fn matrix_multplication_naive2<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut out_row: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out_row.push(row_col_product);
        }
        out.push(out_row);
    }
    out
}