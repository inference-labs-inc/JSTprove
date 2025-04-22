use expander_compiler::frontend::*;
use circuit_std_rs::logup::LogUpRangeProofTable;
use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::circuit_functions::extrema::assert_extremum;
use serde::Deserialize;

const BASE: u32 = 10;
const BATCH_SIZE: usize = 32;
const VEC_LEN: usize = 6;
const NUM_DIGITS: usize = 3; 

declare_circuit!(ExtremaCircuit {
    input_vec: [[PublicVariable; VEC_LEN]; BATCH_SIZE],
    max_val: [PublicVariable; BATCH_SIZE],
});

impl<C: Config> Define<C> for ExtremaCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Create a shared lookup table for digit range checks
        let nb_bits = (32 - BASE.leading_zeros()) as usize;
        let mut table = LogUpRangeProofTable::new(nb_bits);
        table.initial(api);
        let mut table_opt = Some(&mut table);

        for i in 0..BATCH_SIZE {
            let max = self.max_val[i];
            let candidates = &self.input_vec[i];
            let is_max = true;
            let use_lookup = false; 
            // let use_lookup = true;
            assert_extremum(
                api,
                max,
                candidates,
                BASE,
                NUM_DIGITS,
                is_max,
                use_lookup,
                &mut table_opt,
            );
        }
    }
}



#[derive(Deserialize, Clone)]
struct InputData {
    input_vec: Vec<Vec<u32>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    max_val: Vec<u32>,
}

impl<C: Config> IOReader<ExtremaCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: ExtremaCircuit<C::CircuitField>,
    ) -> ExtremaCircuit<C::CircuitField> {
        let data: InputData = <FileReader as IOReader<ExtremaCircuit<C::CircuitField>, C>>::read_data_from_json::<InputData>(file_path);
    
        assert_eq!(data.input_vec.len(), BATCH_SIZE, "Expected {} input vectors", BATCH_SIZE);
        for (i, row) in data.input_vec.iter().enumerate() {
            assert_eq!(row.len(), VEC_LEN, "Expected input vector length {}", VEC_LEN);
            for (j, &val) in row.iter().enumerate() {
                assignment.input_vec[i][j] = C::CircuitField::from(val);
            }
        }
    
        assignment
    }

    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: ExtremaCircuit<C::CircuitField>,
    ) -> ExtremaCircuit<C::CircuitField> {
        let data: OutputData = <FileReader as IOReader<ExtremaCircuit<C::CircuitField>, C>>::read_data_from_json::<OutputData>(file_path);
    
        assert_eq!(data.max_val.len(), BATCH_SIZE, "Expected {} outputs", BATCH_SIZE);
    
        for (i, &val) in data.max_val.iter().enumerate() {
            assignment.max_val[i] = C::CircuitField::from(val);
        }
    
        assignment
    }
    

    fn get_path(&self) -> &str {
        &self.path
    }
}

fn main() {
    let mut file_reader = FileReader {
        path: "extrema".to_owned(),
    };
    // handle_args::<M31Config, ExtremaCircuit<Variable>,ExtremaCircuit<_>,_>(&mut file_reader);
    // handle_args::<BN254Config, ExtremaCircuit<Variable>,ExtremaCircuit<_>,_>(&mut file_reader);
    handle_args::<ExtremaCircuit<Variable>, ExtremaCircuit<_>, _>(&mut file_reader);
    
}