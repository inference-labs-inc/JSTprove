use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use serde::Deserialize;
use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::circuit_functions::conversion::{i64_to_field, matrix_i64_to_field};

const USE_FREIVALDS: bool = true; // change to false for full verification
// const USE_FREIVALDS: bool = false; // change to true to use Freivalds' probabilistic verification

const N_ROWS_A: usize = 30; // ℓ
const N_COLS_A: usize = 30; // m
const N_ROWS_B: usize = 30; // m
const N_COLS_B: usize = 30; // n

declare_circuit!(MatMulBiasCircuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A],         // (ℓ x m)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B],         // (m x n)
    matrix_c: [[Variable; N_COLS_B]; N_ROWS_A],         // (ℓ x n) bias
    matrix_product_ab_plus_c: [[Variable; N_COLS_B]; N_ROWS_A]     // (ℓ x n) output = AB + C
});

impl<C: Config> Define<C> for MatMulBiasCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let a: Vec<Vec<Variable>> = self.matrix_a.iter().map(|row| row.to_vec()).collect();
        let b: Vec<Vec<Variable>> = self.matrix_b.iter().map(|row| row.to_vec()).collect();
        let bias: Vec<Vec<Variable>> = self.matrix_c.iter().map(|row| row.to_vec()).collect();
        let d: Vec<Vec<Variable>> = self.matrix_product_ab_plus_c.iter().map(|row| row.to_vec()).collect();

        if USE_FREIVALDS {
            let x: Vec<Variable> = (0..N_COLS_B).map(|_| api.get_random_value()).collect();

            // Bx ∈ F^m
            let mut bx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_B];
            for i in 0..N_ROWS_B {
                for j in 0..N_COLS_B {
                    let prod = api.mul(b[i][j], x[j]);
                    bx[i] = api.add(bx[i], prod);
                }
            }

            // ABx ∈ F^ℓ
            let mut abx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_A];
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_A {
                    let prod = api.mul(a[i][j], bx[j]);
                    abx[i] = api.add(abx[i], prod);
                }
            }

            // Cx ∈ F^ℓ
            let mut cx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_A];
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_B {
                    let prod = api.mul(bias[i][j], x[j]);
                    cx[i] = api.add(cx[i], prod);
                }
            }

            // Dx ∈ F^ℓ
            let mut dx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_A];
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_B {
                    let prod = api.mul(d[i][j], x[j]);
                    dx[i] = api.add(dx[i], prod);
                }
            }

            // Freivalds check: ABx + Cx == Dx
            for i in 0..N_ROWS_A {
                let lhs = api.add(abx[i], cx[i]);
                api.assert_is_equal(lhs, dx[i]);
            }

        } else {
            // Deterministic check: AB + C == D
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_B {
                    let mut sum = api.constant(CircuitField::<C>::zero());
                    for k in 0..N_COLS_A {
                        let prod = api.mul(a[i][k], b[k][j]);
                        sum = api.add(sum, prod);
                    }
                    let with_bias = api.add(sum, bias[i][j]);
                    api.assert_is_equal(with_bias, d[i][j]);
                }
            }
        }
    }
}

#[derive(Deserialize, Clone)]
struct InputData {
    matrix_a: Vec<Vec<i64>>,       // (ℓ x m)
    matrix_b: Vec<Vec<i64>>,       // (m x n)
    matrix_c: Vec<Vec<i64>>,       // (ℓ x n)
}

#[derive(Deserialize, Clone)]
struct OutputData {
    matrix_product_ab_plus_c: Vec<Vec<i64>>,  // (ℓ x n)
}

impl<C: Config> IOReader<MatMulBiasCircuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(&mut self, file_path: &str, mut assignment: MatMulBiasCircuit<CircuitField::<C>>) -> MatMulBiasCircuit<CircuitField::<C>> {
        let data: InputData = <Self as IOReader<_, C>>::read_data_from_json::<InputData>(file_path);

        let a_vals = matrix_i64_to_field::<C>(data.matrix_a);
        let b_vals = matrix_i64_to_field::<C>(data.matrix_b);
        let c_vals = matrix_i64_to_field::<C>(data.matrix_c);

        for (i, row) in a_vals.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = val;
            }
        }
        for (i, row) in b_vals.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assignment.matrix_b[i][j] = val;
            }
        }
        for (i, row) in c_vals.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assignment.matrix_c[i][j] = val;
            }
        }

        assignment
    }

    fn read_outputs(&mut self, file_path: &str, mut assignment: MatMulBiasCircuit<CircuitField::<C>>) -> MatMulBiasCircuit<CircuitField::<C>> {
        let data: OutputData = <Self as IOReader<_, C>>::read_data_from_json::<OutputData>(file_path);

        let d_vals = matrix_i64_to_field::<C>(data.matrix_product_ab_plus_c);
        for (i, row) in d_vals.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assignment.matrix_product_ab_plus_c[i][j] = val;
            }
        }

        assignment
    }

    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "matmul_bias".to_owned(),
    };
    handle_args::<BN254Config, MatMulBiasCircuit<Variable>,MatMulBiasCircuit<_>,_>(&mut file_reader);

}
