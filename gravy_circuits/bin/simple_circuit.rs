use expander_compiler::frontend::*;
use gravy_circuits::runner::main_runner;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use serde::Deserialize;


declare_circuit!(Circuit {
    input_a: Variable,
    input_b: Variable,
    output: Variable, 
});

declare_circuit!(DummyCircuit {
    input_a: Variable,
    input_b: Variable,
    output: Variable, 
});

impl<C: Config> Define<C> for DummyCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        api.assert_is_equal(0, 10);
    }
}

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

impl<C: Config> IOReader<DummyCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: DummyCircuit<C::CircuitField>,
    ) -> DummyCircuit<C::CircuitField> {
        let data: InputData =
            <FileReader as IOReader<DummyCircuit<_>, C>>::read_data_from_json::<InputData>(file_path);

        // Assign inputs to assignment
       assignment.input_a = C::CircuitField::from(data.input_a);
       assignment.input_b = C::CircuitField::from(data.input_b);

        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: DummyCircuit<C::CircuitField>,
    ) -> DummyCircuit<C::CircuitField> {
        let data: OutputData =
            <FileReader as IOReader<DummyCircuit<_>, C>>::read_data_from_json::<OutputData>(file_path);

        // Assign inputs to assignment
        assignment.output = C::CircuitField::from(data.output);
        
        assignment
    }
    fn get_path(&self) -> &str {
        &self.path
    }
}


fn main() {
    let mut file_reader = FileReader {
        path: "simple_circuit".to_owned(),
    };
    // main_runner::run_bn254::<Circuit<Variable>,
    // Circuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
    //                         _>(&mut file_reader);

    main_runner::run_bn254_seperate::<Circuit<Variable>,
                            DummyCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                                                    _>(&mut file_reader);
}
