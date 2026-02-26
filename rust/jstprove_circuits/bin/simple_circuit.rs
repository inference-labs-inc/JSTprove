use expander_compiler::frontend::{
    BN254Config, CircuitField, Config, Define, RootAPI, Variable, declare_circuit,
};
use jstprove_circuits::io::io_reader::{FileReader, IOReader};
use jstprove_circuits::runner::errors::RunError;
use jstprove_circuits::runner::main_runner::{ConfigurableCircuit, get_args, handle_args};
use serde::Deserialize;

declare_circuit!(Circuit {
    input_a: PublicVariable,
    input_b: PublicVariable,
    nonce: PublicVariable,
    output: PublicVariable,
    dummy: [Variable; 2]
});

//Still to factor this out

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let out = api.add(self.input_a, self.input_b);
        // let out1 = api.add(self.nonce, out);
        api.assert_is_non_zero(self.nonce);

        api.assert_is_equal(out, self.output);
        for i in 0..self.dummy.len() {
            api.assert_is_zero(self.dummy[i]);
        }
    }
}

#[derive(Deserialize, Clone)]
struct InputData {
    value_a: u32,
    value_b: u32,
    nonce: u32,
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    output: u32,
}

impl ConfigurableCircuit for Circuit<Variable> {
    fn configure(&mut self) -> Result<(), jstprove_circuits::runner::errors::RunError> {
        Ok(())
    }
}

impl<C: Config> IOReader<Circuit<CircuitField<C>>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: InputData = <FileReader as IOReader<Circuit<_>, C>>::read_data_from_msgpack::<
            InputData,
        >(file_path)?;

        // Assign inputs to assignment
        assignment.input_a = CircuitField::<C>::from(data.value_a);
        assignment.input_b = CircuitField::<C>::from(data.value_b);
        assignment.nonce = CircuitField::<C>::from(data.nonce);
        assignment.dummy = [CircuitField::<C>::from(0); 2];

        // Return the assignment
        Ok(assignment)
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: Circuit<CircuitField<C>>,
    ) -> Result<Circuit<CircuitField<C>>, RunError> {
        let data: OutputData = <FileReader as IOReader<Circuit<_>, C>>::read_data_from_msgpack::<
            OutputData,
        >(file_path)?;

        // Assign inputs to assignment
        assignment.output = CircuitField::<C>::from(data.output);

        Ok(assignment)
    }
    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "simple_circuit".to_owned(),
    };

    let matches = get_args();

    if let Err(err) = handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(
        &matches,
        &mut file_reader,
        None,
    ) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
