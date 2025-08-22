use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::frontend::Config;
use serde::de::DeserializeOwned;
use std::io::Read;
use gkr_engine::{GKREngine, FieldEngine};

use crate::runner::errors::RunError;


/// Implement io_reader to read inputs and outputs of the circuit.
/// 
/// This is primarily used for witness generation
pub trait IOReader<CircuitType,C: Config>
where
    CircuitType: Default
        +
        // DumpLoadTwoVariables<Variable> +
        DumpLoadTwoVariables<<<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField>
        // + expander_compiler::frontend::Define<C>
        + Clone,
{
    fn read_data_from_json<I>(file_path: &str) -> Result<I, RunError>
    where
        I: DeserializeOwned,
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).map_err(|e| RunError::Io { source: e, path: file_path.into() })?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| RunError::Io { source: e, path: file_path.into() })?;
        // Deserialize the JSON into the InputData struct
        serde_json::from_str(&contents).map_err(|e| RunError::Json(format!("{:?}", e)))
    }
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType, // Mutate the concrete `Circuit` type
    ) -> Result<CircuitType, RunError>;
    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment: CircuitType, // Mutate the concrete `Circuit` type
    ) -> Result<CircuitType, RunError>;
    fn get_path(&self) -> &str;
}
/// To implement IOReader in each binary to read in inputs and outputs of the circuit as is needed on an individual circuit basis
pub struct FileReader {
    pub path: String,
}
