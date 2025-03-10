use expander_compiler::frontend::*;
use io_reader::{FileReader, IOReader};
use serde::Deserialize;
// use std::ops::Neg;

#[path = "../src/io_reader.rs"]
pub mod io_reader;
#[path = "../src/main_runner.rs"]
pub mod main_runner;


declare_circuit!(Circuit {
    input_a: Variable,                       // shape (m, n)
    input_b: Variable,                       // shape (n, k)
    output: Variable, // shape (m, k)
});

//Still to factor this out
impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let out = api.add(self.input_a, self.input_b);

        api.assert_is_equal(out, self.output);
    }
}

#[derive(Deserialize, Clone)]
struct InputData {
    input_a: u32, // Shape (m, n)
    input_b: u32, // Shape (n, k)
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    output: u32,
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
       assignment.input_a = C::CircuitField::from(data.input_a);
       assignment.input_b = C::CircuitField::from(data.input_b);

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

        // Assign inputs to assignment
        assignment.output = C::CircuitField::from(data.output);
        
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
