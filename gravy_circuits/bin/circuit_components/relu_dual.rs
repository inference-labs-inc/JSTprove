use ethnum::U256;
use expander_compiler::frontend::*;
use io_reader::{FileReader, IOReader};
use serde::Deserialize;
// use std::ops::Neg;
use arith::FieldForECC;

#[path = "../../src/relu.rs"]
pub mod relu;

#[path = "../../src/io_reader.rs"]
pub mod io_reader;
#[path = "../../src/main_runner.rs"]
pub mod main_runner;

const LENGTH: usize = 256;

/*
       #######################################################################################################
       #################################### This is the block for changes ####################################
       #######################################################################################################
*/

// Specify input and output structure
// This will indicate the input layer and output layer of the circuit, so be careful with how it is defined
// Later, we define how the inputs get read into the input layer
declare_circuit!(ReLUDualCircuit {
    input: [[Variable; LENGTH]; 2],
    output: [Variable; LENGTH],
});

// Assume 0 is negative and 1 is positive
fn relu<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    sign: Variable,
) -> Variable {
    let sign_2 = api.sub(1, sign);
    api.mul(x, sign_2)
}

impl<C: Config> Define<C> for ReLUDualCircuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        for i in 0..LENGTH {
            // Iterate over each input/output pair (one per batch)
            let x = relu(api, self.input[0][i], self.input[1][i]);
            api.assert_is_equal(x, self.output[i]);

            // let bits = to_binary_constrained(api, self.input[0][i], 32)
        }
    }
}
/*
       #######################################################################################################
       #######################################################################################################
       #######################################################################################################
*/
#[derive(Deserialize, Clone)]
struct InputData {
    input: Vec<u64>,
    sign: Vec<u64>,
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    output: Vec<u64>,
}

impl<C: Config> IOReader<C, ReLUDualCircuit<C::CircuitField>> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: ReLUDualCircuit<C::CircuitField>,
    ) -> ReLUDualCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<C, ReLUDualCircuit<_>>>::read_data_from_json::<
            InputData,
        >(file_path);

        // Assign inputs to assignment
        let u8_vars = [data.input, data.sign];

        for (j, var_vec) in u8_vars.iter().enumerate() {
            for (k, &var) in var_vec.iter().enumerate() {
                assignment.input[j][k] = C::CircuitField::from_u256(U256::from(var));
                // Treat the u8 as a u64 for Field
            }
        }
        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: ReLUDualCircuit<C::CircuitField>,
    ) -> ReLUDualCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<C, ReLUDualCircuit<_>>>::read_data_from_json::<
            OutputData,
        >(file_path);

        for k in 0..LENGTH {
            assignment.output[k] = C::CircuitField::from_u256(U256::from(data.output[k]));
            // Treat the u8 as a u64 for Field
        }
        assignment
    }
}

/*
        #######################################################################################################
        #####################################  Shouldn't need to change  ######################################
        #######################################################################################################
*/

fn main() {
    let mut file_reader = FileReader {
        path: String::new(),
    };
    // run_gf2();
    // run_m31();
    main_runner::run_bn254::<ReLUDualCircuit<Variable>,
        ReLUDualCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
}
