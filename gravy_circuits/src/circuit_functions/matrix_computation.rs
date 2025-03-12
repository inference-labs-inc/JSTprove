use expander_compiler::frontend::*;




pub fn matrix_addition<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: [[Variable; N]; M],
) -> [[Variable; N]; M] {
    let mut array: [[Variable; N]; M] = [[Variable::default(); N]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..N {
            array[i][j] = api.add(matrix_a[i][j], matrix_b[i][j]);
        }
    }
    array
}


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

pub fn matrix_hadamard_product<C: Config, Builder: RootAPI<C>, const M: usize, const N: usize>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: [[Variable; N]; M],
) -> [[Variable; N]; M] {
    let mut array: [[Variable; N]; M] = [[Variable::default(); N]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..N {
            array[i][j] = api.mul(matrix_a[i][j], matrix_b[i][j]);
        }
    }
    array
}

fn product_sub_circuit<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    inputs: &Vec<Variable>,
) -> Vec<Variable> {
    let n = inputs.len() / 2; // Assuming inputs are concatenated row and column
                              // let mut out: Vec<Variable> = Vec::new();
    let mut sum = api.constant(0);

    for k in 0..n {
        let x = api.mul(inputs[k], inputs[n + k]);
        sum = api.add(sum, x);
    }
    vec![sum]
}

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


pub fn matrix_multplication_array<
    C: Config,
    Builder: RootAPI<C>,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: Vec<Vec<Variable>>,
) -> [[Variable; K]; M] {
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

pub fn matrix_multplication_naive_array<
    C: Config,
    Builder: RootAPI<C>,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: [[Variable; K]; N],
) -> [[Variable; K]; M] {
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

pub fn matrix_multplication_naive2_array<
    C: Config,
    Builder: RootAPI<C>,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: Vec<Vec<Variable>>,
) -> [[Variable; K]; M] {
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

pub fn matrix_multplication_naive3_array<
    C: Config,
    Builder: RootAPI<C>,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
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

pub fn matrix_multplication<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
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

pub fn matrix_multplication_naive<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
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

pub fn matrix_multplication_naive3<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
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

pub fn two_d_array_to_vec<const M: usize, const N: usize>(
    matrix: [[Variable; N]; M],
) -> Vec<Vec<Variable>> {
    matrix.iter().map(|row| row.to_vec()).collect()
}

pub fn gemm<C: Config, const M: usize, const N: usize, const K: usize, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: [[Variable; K]; N],
    matrix_c: [[Variable; K]; M],
    alpha: Variable,
    beta: Variable,
) -> [[Variable; K]; M] {
    let mut array: [[Variable; K]; M] = [[Variable::default(); K]; M]; // or [[Variable::default(); N]; M]
    for i in 0..M {
        for j in 0..K {
            let mut gemm_ij: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                gemm_ij = api.add(gemm_ij, element_product);
            }
            gemm_ij = api.mul(gemm_ij, alpha);
            let scaled_c_ij = api.mul(beta, matrix_c[i][j]);
            gemm_ij = api.add(scaled_c_ij, gemm_ij);
            array[i][j] = gemm_ij;
        }
    }
    array
}

pub fn gemm_vec<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    matrix_a: Vec<Vec<Variable>>,
    matrix_b: Vec<Vec<Variable>>,
    matrix_c: Vec<Vec<Variable>>,
    alpha: Variable,
    beta: Variable,
) -> Vec<Vec<Variable>> {
    let mut array: Vec<Vec<Variable>> = Vec::new(); // or [[Variable::default(); N]; M]
    for (i,dim1) in matrix_a.iter().enumerate() {
        let mut row_1: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut gemm_ij: Variable = api.constant(0);
            for k in 0..dim1.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                gemm_ij = api.add(gemm_ij, element_product);
            }
            gemm_ij = api.mul(gemm_ij, alpha);
            let scaled_c_ij = api.mul(beta, matrix_c[i][j]);
            gemm_ij = api.add(scaled_c_ij, gemm_ij);
            row_1.push(gemm_ij);
            // array[i][j] = gemm_ij;
        }
        array.push(row_1);
    }
    array
}

pub fn scaled_matrix_product_sum<
    C: Config,
    Builder: RootAPI<C>,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: [[Variable; K]; N],
    matrix_c: [[Variable; K]; M],
    alpha: Variable,
) -> [[Variable; K]; M] {
    let mut array: [[Variable; K]; M] = [[Variable::default(); K]; M]; // or [[Variable::default(); N]; M]
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

pub fn scaled_matrix_product<
    C: Config,
    Builder: RootAPI<C>,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    api: &mut Builder,
    matrix_a: [[Variable; N]; M],
    matrix_b: [[Variable; K]; N],
    alpha: Variable,
) -> [[Variable; K]; M] {
    let mut array: [[Variable; K]; M] = [[Variable::default(); K]; M]; // or [[Variable::default(); N]; M]
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


fn matrix_multiply<C: Config>(
    builder: &mut impl RootAPI<C>,
    target_mat: &mut [Variable], // target to modify
    aux_mat: &[Variable],
    origin_mat: &[Vec<Variable>],
) {
    for (i, target_item) in target_mat.iter_mut().enumerate() {
        for (j, item) in aux_mat.iter().enumerate() {
            let mul_result = builder.mul(origin_mat[i][j], item);
            *target_item = builder.add(*target_item, mul_result);
        }
    }
}

// #[allow(clippy::needless_range_loop)]
// fn matrix_multiply_unconstrained<C: Config, Builder: RootAPI<C>>(
//     api: &mut Builder,
//     a: &Vec<Vec<Variable>>,
//     b: &Vec<Vec<Variable>>,
//     zero: &Variable
// ) -> Vec<Vec<Variable>> {
//     let m1 = a.len();
//     let n1 = a[0].len();
//     let m2 = b.len();
//     let n2 = b[0].len();

//     assert_eq!(n1, m2, "n1 ! = m2 ");

//     // initialize the result matrix
//     let mut c = vec![vec![*zero; n2]; m1];
//     // let mut c: Vec<Vec<Variable>> = Vec::new();


//     // FIXME: optimize calculating the multiplication for super large matrix.
//     for i in 0..m1 {
//         for j in 0..n2 {
//             for k in 0..n1 {
//                 let temp = api.unconstrained_mul(a[i][k], b[k][j]);
//                 c[i][j] = api.unconstrained_add(temp, c[i][j]);
//             }
//         }
//     }
//     c
// }
pub fn matrix_multiplication_std<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    first_mat: Vec<Vec<Variable>>,
    second_mat: Vec<Vec<Variable>>,
    result_mat: Vec<Vec<Variable>>
) -> Vec<Vec<Variable>> {
    // [m1,n1] represents the first matrix's dimension
    let m1 = first_mat.len();
    let n1 = first_mat[0].len();

    // [m2,n2] represents the second matrix's dimension
    let m2 = second_mat.len();
    let n2 = second_mat[0].len();
    let zero = api.constant(0);
    // let result_mat = matrix_multiply_unconstrained(api, &first_mat, &second_mat, &zero);

    // [r1,r2] represents the result matrix's dimension
    let r1 = result_mat.len();
    let r2 = result_mat[0].len();
    

    api.assert_is_equal(Variable::from(n1), Variable::from(m2));
    api.assert_is_equal(Variable::from(r1), Variable::from(m1));
    api.assert_is_equal(Variable::from(r2), Variable::from(n2));

    let loop_count = if C::CircuitField::SIZE == M31::SIZE {
        3
    } else {
        1
    };

    for _ in 0..loop_count {
        let randomness = api.get_random_value();
        let mut aux_mat = Vec::new();
        let mut challenge = randomness;

        // construct the aux matrix = [1, randomness, randomness^2, ..., randomness^（n-1）]
        aux_mat.push(Variable::from(1));
        for _ in 0..n2 - 1 {
            challenge = api.mul(challenge, randomness);
            aux_mat.push(challenge);
        }

        let mut aux_second = vec![zero; m2];
        let mut aux_first = vec![zero; m1];
        let mut aux_res = vec![zero; m1];

        // calculate second_mat * aux_mat,
        matrix_multiply(api, &mut aux_second, &aux_mat, &second_mat);
        // calculate result_mat * aux_second
        matrix_multiply(api, &mut aux_res, &aux_mat, &result_mat);
        // calculate first_mat * aux_second
        matrix_multiply(api, &mut aux_first, &aux_second, &first_mat);

        // compare aux_first with aux_res
        for i in 0..m1 {
            api.assert_is_equal(aux_first[i], aux_res[i]);
        }
    }
    result_mat
}
