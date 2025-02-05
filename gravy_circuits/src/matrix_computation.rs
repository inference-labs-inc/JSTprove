use expander_compiler::frontend::*;


pub fn matrix_addition<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; N]; M]) -> [[Variable; N]; M]{
    let mut array:[[Variable; N]; M]  = [[Variable::default(); N]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..N {
            array[i][j] = api.add(matrix_a[i][j], matrix_b[i][j]);
        }                       
    }
    array
}

pub fn matrix_hadamard_product<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; N]; M]) -> [[Variable; N]; M]{
    let mut array:[[Variable; N]; M]  = [[Variable::default(); N]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..N {
            array[i][j] = api.mul(matrix_a[i][j], matrix_b[i][j]);
        }                       
    }
    array
}


fn product_sub_circuit<C: Config, Builder: RootAPI<C>>(api: &mut Builder, inputs: &Vec<Variable>) -> Vec<Variable>  {
    let n = inputs.len()/2; // Assuming inputs are concatenated row and column
    // let mut out: Vec<Variable> = Vec::new();
    let mut sum = api.constant(0);

    for k in 0..n {
        let x = api.mul(inputs[k], inputs[n + k]);
        sum = api.add(sum, x);
    }
    vec![sum]
}

pub fn dot<C: Config, Builder: RootAPI<C>>(api: &mut Builder, vector_a: Vec<Variable>, vector_b: Vec<Variable>) -> Variable{
    let mut row_col_product: Variable = api.constant(0);
    for k in 0..vector_a.len() {
        let element_product = api.mul(vector_a[k], vector_b[k]);
        row_col_product = api.add(row_col_product, element_product);
    }
    row_col_product
}

pub fn matrix_multplication_array<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize, const K: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: Vec<Vec<Variable>>) -> [[Variable; K]; M]{
    let mut out = [[Variable::default(); K]; M];
    for i in 0..M {
        for j in 0..K {
            // Prepare inputs as concatenated row and column
            // api.add(C::CircuitField::from(weights[0][0] as u32),self.matrix_a[0][0]);
            let mut inputs: Vec<Variable> = Vec::new();
            for k in 0..N {
                inputs.push(matrix_a[i][k]);
            }
            for k in 0..N {
                inputs.push(matrix_b[k][j]);
            }
            // Use MemorizedSimpleCall for the row-column dot product
            out[i][j] = api.memorized_simple_call(product_sub_circuit, &inputs)[0];
            // api.assert_is_equal(self.matrix_product_ab[i][j], prod[0]);
        }
    }
    out
}

pub fn matrix_multplication_naive_array<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize, const K: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; K]; N]) -> [[Variable; K]; M]{
    let mut out = [[Variable::default(); K]; M];
    for i in 0..M {
        for j in 0..K {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out[i][j] = row_col_product;               
        }
    }
    out
}

pub fn matrix_multplication_naive2_array<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize, const K: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: Vec<Vec<Variable>>) -> [[Variable; K]; M]{
    let mut out = [[Variable::default(); K]; M];
    for i in 0..M {
        for j in 0..K {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out[i][j] = row_col_product;
        }
    }
    out
}

pub fn matrix_multplication_naive3_array<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize, const K: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..M {
        let mut row_out: Vec<Variable> = Vec::new();
        for j in 0..K {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            row_out.push(row_col_product);
        }
        out.push(row_out);
    }
    out
}
// Vector of Vectors 

pub fn matrix_multplication<C: Config, Builder: RootAPI<C>>(api: &mut Builder, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut out_rows: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            // Prepare inputs as concatenated row and column
            let mut inputs: Vec<Variable> = Vec::new();
            for k in 0..matrix_b.len() {
                inputs.push(matrix_a[i][k]);
            }
            for k in 0..matrix_b.len() {
                inputs.push(matrix_b[k][j]);
            }
            // Use MemorizedSimpleCall for the row-column dot product
            out_rows.push(api.memorized_simple_call(product_sub_circuit, &inputs)[0]);
        }
        out.push(out_rows);
    }
    out
}

pub fn matrix_multplication_naive<C: Config, Builder: RootAPI<C>>(api: &mut Builder, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();    
    for i in 0..matrix_a.len() {
        let mut out_rows: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out_rows.push(row_col_product);
        }
        out.push(out_rows)
    }
    out
}

pub fn matrix_multplication_naive2<C: Config, Builder: RootAPI<C>>(api: &mut Builder, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
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

pub fn matrix_multplication_naive3<C: Config, Builder: RootAPI<C>>(api: &mut Builder, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut row_out: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            row_out.push(row_col_product);
        }
        out.push(row_out);
    }
    out
}

pub fn two_d_array_to_vec<const M: usize, const N: usize>(matrix:[[Variable; N]; M]) -> Vec<Vec<Variable>>{
    matrix.iter()
    .map(|row| row.to_vec())
    .collect()                               
}

pub fn gemm<C: Config, const M: usize, const N: usize, const K: usize, Builder: RootAPI<C>>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; K]; N], matrix_c: [[Variable; K]; M],alpha: Variable, beta: Variable) -> [[Variable; K]; M]{
    let mut array:[[Variable; K]; M]  = [[Variable::default(); K]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..K {
            let mut gemm_ij: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                gemm_ij = api.add(gemm_ij, element_product);                                        
            }
            gemm_ij = api.mul(gemm_ij, alpha);
            let scaled_c_ij = api.mul(beta,matrix_c[i][j]);
            gemm_ij = api.add(scaled_c_ij, gemm_ij);
            array[i][j] = gemm_ij;
            }
    }
    array
}

pub fn scaled_matrix_product_sum<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize, const K: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; K]; N], matrix_c: [[Variable; K]; M],alpha: Variable) -> [[Variable; K]; M]{
    let mut array:[[Variable; K]; M]  = [[Variable::default(); K]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..K {
            let mut scaled_row_col_product_sum: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                scaled_row_col_product_sum = api.add(scaled_row_col_product_sum, element_product);                                      
            }
            scaled_row_col_product_sum = api.mul(scaled_row_col_product_sum, alpha);
            scaled_row_col_product_sum = api.add(scaled_row_col_product_sum, matrix_c[i][j]);
            array[i][j] = scaled_row_col_product_sum;        
        }
    }
    array
}

pub fn scaled_matrix_product<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize, const K: usize>(api: &mut Builder, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; K]; N], alpha: Variable) -> [[Variable; K]; M]{
    let mut array:[[Variable; K]; M]  = [[Variable::default(); K]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..K {
            let mut scaled_row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                scaled_row_col_product = api.add(scaled_row_col_product, element_product);                   
            }
            scaled_row_col_product = api.mul(scaled_row_col_product, alpha);
            array[i][j] = scaled_row_col_product;            
        }
    }
    array
}
