use gravy_circuits::{io::io_reader, runner::main_runner};
use ethnum::U256;
use expander_compiler::frontend::*;
use io_reader::{FileReader, IOReader};
use serde::Deserialize;
// use std::ops::Neg;
use arith::FieldForECC;
// :)

const LENGTH: usize = 10000;

/*
       #######################################################################################################
       #################################### This is the block for changes ####################################
       #######################################################################################################
*/

// Specify input and output structure
// This will indicate the input layer and output layer of the circuit, so be careful with how it is defined
// Later, we define how the inputs get read into the input layer
declare_circuit!(Circuit {
    input: [[Variable; LENGTH]; 2],
    output: [Variable; LENGTH],
});

//This is where the circuit is defined. We can refactor out some modular components to this, but this is where it is put together
impl<C: Config> Define<C> for Circuit<Variable> {
    // Default circuit for now
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Iterate over each input/output pair (one per batch)
        for i in 0..LENGTH {
            let out = api.add(self.input[0][i].clone(), self.input[1][i].clone());
            let out2 = api.mul(&out, &out);
            let out3 = api.mul(&out2, &out2);
            let out4 = api.mul(&out3, &out3);
            // let out4 = out;

            api.assert_is_equal(out4.clone(), self.output[i].clone());
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
    inputs_1: Vec<u64>,
    inputs_2: Vec<u64>,
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    outputs: Vec<u64>,
}

impl<C: Config> IOReader<C, Circuit<C::CircuitField>> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<C::CircuitField>,
    ) -> Circuit<C::CircuitField> {
        let data: InputData =
            <FileReader as IOReader<C, Circuit<_>>>::read_data_from_json::<InputData>(file_path);

        // Assign inputs to assignment
        let u8_vars = [data.inputs_1, data.inputs_2];

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
        mut assignment: Circuit<C::CircuitField>,
    ) -> Circuit<C::CircuitField> {
        let data: OutputData =
            <FileReader as IOReader<C, Circuit<_>>>::read_data_from_json::<OutputData>(file_path);

        for k in 0..LENGTH {
            assignment.output[k] = C::CircuitField::from_u256(U256::from(data.outputs[k]));
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
    main_runner::run_bn254::<Circuit<Variable>,
    Circuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                            _>(&mut file_reader);
}
