use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use serde::Deserialize;
// use ethnum::U256;
use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::circuit_functions::conversion::matrix_i64_to_field;

// const USE_FREIVALDS: bool = true; // change to false for full verification
const USE_FREIVALDS: bool = false; // change to true to use Freivalds' probabilistic verification

const N_ROWS_A: usize = 40; // l
const N_COLS_A: usize = 40; // m
const N_ROWS_B: usize = 40; // m
const N_COLS_B: usize = 40; // n

declare_circuit!(MatMulCircuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A],         // (l x m)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B],         // (m x n)
    matrix_product_ab: [[Variable; N_COLS_B]; N_ROWS_A] // (l x n), provided by prover
});


impl<C: Config> Define<C> for MatMulCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let a: Vec<Vec<Variable>> = self.matrix_a.iter().map(|row| row.to_vec()).collect();
        let b: Vec<Vec<Variable>> = self.matrix_b.iter().map(|row| row.to_vec()).collect();
        let c: Vec<Vec<Variable>> = self.matrix_product_ab.iter().map(|row| row.to_vec()).collect();

        if USE_FREIVALDS {
            // ---- Freivalds' Check ----
            let x: Vec<Variable> = (0..N_COLS_B).map(|_| api.get_random_value()).collect();

            // u = Bx lies in F^m
            let mut bx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_B];
            for i in 0..N_ROWS_B {
                for j in 0..N_COLS_B {
                    let prod = api.mul(b[i][j], x[j]);
                    bx[i] = api.add(bx[i], prod);
                }
            }

            // v = Au lies in F^l
            let mut abx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_A];
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_A {
                    let prod = api.mul(a[i][j], bx[j]);
                    abx[i] = api.add(abx[i], prod);
                }
            }

            // Cx lies F^l
            let mut cx = vec![api.constant(CircuitField::<C>::zero()); N_ROWS_A];
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_B {
                    let prod = api.mul(c[i][j], x[j]);
                    cx[i] = api.add(cx[i], prod);
                }
            }

            // Freivalds' check: Cx == v
            for i in 0..N_ROWS_A {
                api.assert_is_equal(abx[i], cx[i]);
            }
        } else {
            // ---- Full Deterministic Check: AB = C ----
            for i in 0..N_ROWS_A {
                for j in 0..N_COLS_B {
                    let mut sum = api.constant(CircuitField::<C>::zero());
                    for k in 0..N_COLS_A {
                        let prod = api.mul(a[i][k], b[k][j]);
                        sum = api.add(sum, prod);
                    }

                    api.assert_is_equal(sum, c[i][j]);
                }
            }
        }
    }
}


#[derive(Deserialize, Clone)]
struct InputData {
    matrix_a: Vec<Vec<i64>>,       // (l x m)
    matrix_b: Vec<Vec<i64>>,       // (m x n)
}

#[derive(Deserialize, Clone)]
struct OutputData {
    matrix_product_ab: Vec<Vec<i64>>, // (l x n)
}

impl<C: Config> IOReader<MatMulCircuit<CircuitField::<C>>, C> for FileReader {
    fn read_inputs(&mut self, file_path: &str, mut assignment: MatMulCircuit<CircuitField::<C>>) -> MatMulCircuit<CircuitField::<C>> {
        let data: InputData = <Self as IOReader<_, C>>::read_data_from_json::<InputData>(file_path);

        let a_vals = matrix_i64_to_field::<C>(data.matrix_a);
        let b_vals = matrix_i64_to_field::<C>(data.matrix_b);       

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

        assignment
    }

    fn read_outputs(&mut self, file_path: &str, mut assignment: MatMulCircuit<CircuitField::<C>>) -> MatMulCircuit<CircuitField::<C>> {
        let data: OutputData = <Self as IOReader<_, C>>::read_data_from_json::<OutputData>(file_path);
        
        let c_vals = matrix_i64_to_field::<C>(data.matrix_product_ab);
        for (i, row) in c_vals.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assignment.matrix_product_ab[i][j] = val;
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
        path: "matmul".to_owned(),
    };
    handle_args::<BN254Config, MatMulCircuit<Variable>,MatMulCircuit<_>,_>(&mut file_reader);

}
