use ethnum::U256;
use expander_compiler::frontend::Config;
use std::{io::Read, ops::Neg};
use arith::FieldForECC;
use serde::Deserialize;
use serde::de::DeserializeOwned;

pub trait IOReader<C: Config> {
    type CircuitType;

    fn read_data_from_json<I>(file_path: &str) -> I
    where I: DeserializeOwned
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: I = serde_json::from_str(&contents).unwrap();

        data
    }
    fn read_inputs(
        &mut self,
        file_path: &str,
        assignment: Self::CircuitType, // Mutate the concrete `Circuit` type
    ) -> Self::CircuitType;
    fn read_outputs(
        &mut self,
        file_path: &str,
        assignment:  Self::CircuitType, // Mutate the concrete `Circuit` type
    ) -> Self::CircuitType;
}

pub struct FileReader {
    pub path: String,
}