use arith::FieldForECC;
use jstprove_circuits::circuit_functions::max_pool::max_pool;
use ethnum::U256;
use expander_compiler::frontend::*;
use jstprove_circuits::circuit_functions::helper_fn::four_d_array_to_vec;
use jstprove_circuits::io::io_reader::{FileReader, IOReader};
use serde::Deserialize;
use std::ops::Neg;
use jstprove_circuits::runner::main_runner;

const BATCH: usize = 1;
const CHANNEL: usize = 1;
const HEIGHT: usize = 4;
const WIDTH: usize = 4;

const OUT_HEIGHT: usize = 2;
const OUT_WIDTH: usize = 2;

#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<Vec<Vec<i64>>>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<Vec<Vec<i64>>>>,
}

declare_circuit!(MaxPoolCircuit {
    input_arr: [[[[Variable; WIDTH]; HEIGHT]; CHANNEL]; BATCH],
    output_arr: [[[[Variable; OUT_WIDTH]; OUT_HEIGHT]; CHANNEL]; BATCH],
});

impl<C: Config> Define<C> for MaxPoolCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let input = four_d_array_to_vec(self.input_arr.clone());

        let kernel_shape = vec![2, 2];
        let strides = vec![2, 2];
        let pads = vec![0, 0, 0, 0]; // No padding

        let output = max_pool(api, input, &vec![BATCH as u32, CHANNEL as u32, HEIGHT as u32, WIDTH as u32], &kernel_shape, &strides, &pads);

        for b in 0..BATCH {
            for c in 0..CHANNEL {
                for h in 0..OUT_HEIGHT {
                    for w in 0..OUT_WIDTH {
                        api.assert_is_equal(self.output_arr[b][c][h][w], output[b][c][h][w]);
                    }
                }
            }
        }
    }
}

impl<C: Config> IOReader<MaxPoolCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: MaxPoolCircuit<C::CircuitField>,
    ) -> MaxPoolCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<MaxPoolCircuit<_>, C>>::read_data_from_json::<InputData>(file_path);

        for b in 0..BATCH {
            for c in 0..CHANNEL {
                for h in 0..HEIGHT {
                    for w in 0..WIDTH {
                        let val = data.input[b][c][h][w];
                        assignment.input_arr[b][c][h][w] = if val < 0 {
                            C::CircuitField::from(val.abs() as u32).neg()
                        } else {
                            C::CircuitField::from(val as u32)
                        };
                    }
                }
            }
        }

        assignment
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: MaxPoolCircuit<C::CircuitField>,
    ) -> MaxPoolCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<MaxPoolCircuit<_>, C>>::read_data_from_json::<OutputData>(file_path);

        for b in 0..BATCH {
            for c in 0..CHANNEL {
                for h in 0..OUT_HEIGHT {
                    for w in 0..OUT_WIDTH {
                        let val = data.output[b][c][h][w];
                        assignment.output_arr[b][c][h][w] = if val < 0 {
                            C::CircuitField::from_u256(U256::from(val.abs() as u64)).neg()
                        } else {
                            C::CircuitField::from_u256(U256::from(val as u64))
                        };
                    }
                }
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
        path: "max_pool".to_owned(),
    };
    main_runner::run_bn254::<
        MaxPoolCircuit<Variable>,
        MaxPoolCircuit<<BN254Config as Config>::CircuitField>,
        _,
    >(&mut file_reader);
}


