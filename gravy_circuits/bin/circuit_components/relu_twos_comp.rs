use arith::FieldForECC;
use ethnum::U256;
use expander_compiler::frontend::*;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::runner::main_runner::handle_args;
use serde::Deserialize;
use std::ops::Neg;
use gravy_circuits::circuit_functions::relu;


/*
       ########################################################################################################
       ########################################## Define the Circuit ##########################################
       ########################################################################################################
*/

// Specify input and output structure
// This will indicate the input layer and output layer of the circuit, so be careful with how it is defined
// Later, we define how the inputs get read into the input layer
const SIZE1: usize = 28;
const SIZE2: usize = 28;
const SIZE3: usize = 16;

declare_circuit!(ReLUTwosCircuit {
    input: [[[Variable; SIZE1]; SIZE2]; SIZE3],
    output: [[[Variable; SIZE1]; SIZE2]; SIZE3],
});

impl<C: Config> Define<C> for ReLUTwosCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n_bits = 32;

        // let out = relu::relu_3d_naive(api, self.input,n_bits);
        let out = relu::relu_3d_v2(api, self.input, n_bits);
        // let out = relu::relu_3d_v3(api, self.input);

        for i in 0..SIZE3 {
            for j in 0..SIZE2 {
                for k in 0..SIZE1 {
                    api.assert_is_equal(self.output[i][j][k], out[i][j][k]);
                }
            }
        }
    }
}

/*
        ########################################################################################################
        ######################  This is where we define the inputs and outputs structure  ######################
        ########################################################################################################
*/
#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<Vec<Vec<u64>>>,
}
#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<Vec<Vec<i64>>>,
}
/*
        ########################################################################################################
        ###################  This is where we define the inputs and outputs of the function  ###################
        ########################################################################################################
*/
impl<C: Config> IOReader<ReLUTwosCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: ReLUTwosCircuit<C::CircuitField>,
    ) -> ReLUTwosCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<ReLUTwosCircuit<_>, C>>::read_data_from_json::<
            InputData,
        >(file_path);

        for (i, var_vec_vec) in data.input.iter().enumerate() {
            for (j, var_vec) in var_vec_vec.iter().enumerate() {
                for (k, &var) in var_vec.iter().enumerate() {
                    if var < 0 {
                        assignment.input[i][j][k] = C::CircuitField::from(var.abs() as u32).neg();
                    } else {
                        assignment.input[i][j][k] = C::CircuitField::from(var.abs() as u32);
                    }
                }
            }
        }
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: ReLUTwosCircuit<C::CircuitField>,
    ) -> ReLUTwosCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<ReLUTwosCircuit<_>, C>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for (i, var_vec_vec) in data.output.iter().enumerate() {
            for (j, var_vec) in var_vec_vec.iter().enumerate() {
                for (k, &var) in var_vec.iter().enumerate() {
                    assignment.output[i][j][k] = C::CircuitField::from_u256(U256::from(var));
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
        path: "relu_twos_comp".to_owned(),
    };
    handle_args::<BN254Config, ReLUTwosCircuit<Variable>,ReLUTwosCircuit<_>,_>(&mut file_reader);
    // handle_args::<M31Config, ReLUTwosCircuit<Variable>,ReLUTwosCircuit<_>,_>(&mut file_reader);
    // handle_args::<GF2Config, ReLUTwosCircuit<Variable>,ReLUTwosCircuit<_>,_>(&mut file_reader);
}
