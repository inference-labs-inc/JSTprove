use std::ops::Neg;

use expander_compiler::frontend::*;
use circuit_std_rs::logup::LogUpRangeProofTable;
use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use gravy_circuits::circuit_functions::extrema::assert_extremum;
use serde::Deserialize;


const BASE: u32 = 2;
const BATCH_SIZE: usize = 32;
const VEC_LEN: usize = 6;
const NUM_DIGITS: usize = 32; 

declare_circuit!(ExtremaCircuit {
    input_vec: [[PublicVariable; VEC_LEN]; BATCH_SIZE],
    max_val: [PublicVariable; BATCH_SIZE],
});

// pub fn to_binary<F: Field>(x: F, num_outputs: usize) -> Result<Vec<F>, Error> {
//     let mut outputs = Vec::with_capacity(num_outputs);
//     let mut y = x.to_u256();
//     for _ in 0..num_outputs {
//         outputs.push(F::from_u256(y & U256::from(1u32)));
//         y >>= 1;
//     }
//     if y != U256::ZERO {
//         return Err(Error::UserError(
//             "to_binary hint input too large".to_string(),
//         ));
//     }
//     Ok(outputs)
// }

// pub fn get_max_hint_32<F: Field>(x: Vec<F>, mut out: F) -> Result<(), Error> {
//     let midpoint = (F::FIELD_SIZE/2 )as u128;
//     let mut output = x[0].to_u256();
//     let mut is_neg = if output > midpoint {
//         true
//     } else {
//         false
//     };

//     for i in 1..x.len(){
//         let y = x[i].to_u256();
//         // y is positive 
//         if y < midpoint {
//             //output is negative
//             if is_neg {
//                 is_neg = false;
//                 output = y;
//                 }
//             // y is bigger than output, otherwise output is bigger than y and do nothing
//             else if y > output {
//                 output = y;
//             }
//         }
//         else {
//             //both values are negative, otherwise change nothing
//             if is_neg && y > output{
//                 //y is bigger than outputs
//                     output = y;
//             }
//         }
//     }
//         //y is negative'
//     Ok(())
// }

pub fn get_max_unconstrained<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder, x: Vec<Variable>) -> Variable {
    let midpoint = C::CircuitField::from_u256(ethnum::U256::from(C::CircuitField::MODULUS/2));

    let mut output = api.unconstrained_add(x[0], midpoint);

    for i in 1..x.len(){
        let y = api.unconstrained_add(x[i], midpoint);
        let is_new_max = api.unconstrained_greater(y, output);
        let not_new_max = api.unconstrained_lesser_eq(y, output);
        
        let output_1 = api.unconstrained_mul(y, is_new_max);
        let output_2 = api.unconstrained_mul(output, not_new_max);

        output = api.unconstrained_add(output_1, output_2);
    }
    let midpoint_and_one = api.unconstrained_add(midpoint, 1);
    let x = api.unconstrained_add(output, midpoint_and_one);
    x
}


impl<C: Config> Define<C> for ExtremaCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        // Create a shared lookup table for digit range checks
        let nb_bits = (32 - BASE.leading_zeros()) as usize;
        let mut table = LogUpRangeProofTable::new(nb_bits);
        table.initial(api);
        let mut table_opt = Some(&mut table);
        let mut max_vals = Vec::new();

        for i in 0..BATCH_SIZE {
            let max = get_max_unconstrained(api, self.input_vec[i].to_vec());

            // let max = self.max_val[i];
            let candidates = &self.input_vec[i];
            let is_max = true;
            let use_lookup = false; 
            // let use_lookup = true;
            api.display("max", max);
            api.display("max_true", self.max_val[i]);

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
            max_vals.push(max);
        }
        for i in 0..BATCH_SIZE {
            api.assert_is_equal(self.max_val[i],max_vals[i]);
        }
    }
}



#[derive(Deserialize, Clone)]
struct InputData {
    input_vec: Vec<Vec<i32>>,
}

#[derive(Deserialize, Clone)]
struct OutputData {
    max_val: Vec<i32>,
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
                if val < 0 {
                    assignment.input_vec[i][j] =
                        C::CircuitField::from(val.abs() as u32).neg();
                } else {
                    assignment.input_vec[i][j] = C::CircuitField::from(val.abs() as u32);
                }
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
            if val < 0 {
                assignment.max_val[i] =
                    C::CircuitField::from(val.abs() as u32).neg();
            } else {
                assignment.max_val[i] = C::CircuitField::from(val.abs() as u32);
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
        path: "extrema".to_owned(),
    };
    // handle_args::<M31Config, ExtremaCircuit<Variable>,ExtremaCircuit<_>,_>(&mut file_reader);
    // handle_args::<BN254Config, ExtremaCircuit<Variable>,ExtremaCircuit<_>,_>(&mut file_reader);
    handle_args::<ExtremaCircuit<Variable>, ExtremaCircuit<_>, _>(&mut file_reader);
    
}