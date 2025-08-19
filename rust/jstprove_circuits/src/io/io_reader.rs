use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::frontend::Config;
use gkr_engine::{FieldEngine, GKREngine};
use serde::de::DeserializeOwned;
use std::io::Read;

/// Implement io_reader to read inputs and outputs of the circuit.
///
/// This is primarily used for witness generation
pub trait IOReader<CircuitType, C: Config>
where
    CircuitType: Default
        +
        // DumpLoadTwoVariables<Variable> +
        DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        // + expander_compiler::frontend::Define<C>
        + Clone,
{
    fn read_data_from_json<I>(file_path: &str) -> I
    where
        I: DeserializeOwned,
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");
        // panic!("{}", type_name::<I>());
        // Deserialize the JSON into the InputData struct
        let data: I = serde_json::from_str(&contents).unwrap();

        data
    }
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType, // Mutate the concrete `Circuit` type
    ) -> CircuitType;
    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType, // Mutate the concrete `Circuit` type
    ) -> CircuitType;
    fn get_path(&self) -> &str;
}
/// To implement IOReader in each binary to read in inputs and outputs of the circuit as is needed on an individual circuit basis
pub struct FileReader {
    pub path: String,
}
