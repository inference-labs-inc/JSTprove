
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};
use clap::{Command, Arg};
use std::io::{self, Write};




const BATCH_SIZE: usize = 256;



declare_circuit!(Circuit {
    input: [[Variable; 18]; BATCH_SIZE],
    output: [[Variable; 18]; BATCH_SIZE],
});

impl<C: Config> Define<C> for Circuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define(&self, api: &mut API<C>) {
        // Iterate over each input/output pair (one per batch)
        for i in 0..BATCH_SIZE {
            for j in 0..18 { 
                // Compare each input variable (self.p[i][j]) with the corresponding output (self.out[i][j])
                api.assert_is_equal(self.input[i][j].clone(), self.output[i][j].clone());
            }
        }
    }
}

fn run_main<C: Config, GKRC>()
where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>,
{
    println!("TEST");
    std::io::stdout().flush();
    let matches = Command::new("File Copier")
        .version("1.0")
        .about("Copies content from input file to output file")
        .arg(
            Arg::new("input")
                .help("The input file to read from")
                .required(true)  // This argument is required
                .index(1),       // Positional argument (first argument)
        )
        .arg(
            Arg::new("output")
                .help("The output file to write to")
                .required(true)  // This argument is also required
                .index(2),       // Positional argument (second argument)
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();// "inputs/reward_input.json"
    let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"



    let n_witnesses = <GKRC::SimdCircuitField as arith::SimdField>::pack_size();
    println!("n_witnesses: {}", n_witnesses);
    let compile_result: CompileResult<C> = compile(&Circuit::default()).unwrap();

    let assignment = Circuit::<C::CircuitField>::default();

    let assignment = io_reader::input_data_from_json::<C, GKRC>(input_path, assignment);

    let assignment = io_reader::output_data_from_json::<C, GKRC>(output_path, assignment);

    let assignments = vec![assignment; n_witnesses];
    let witness = compile_result
        .witness_solver
        .solve_witnesses(&assignments)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    for x in output.iter() {
        assert_eq!(*x, true);
    }

    let mut expander_circuit = compile_result
        .layered_circuit
        .export_to_expander::<GKRC>()
        .flatten();
    let config = expander_config::Config::<GKRC>::new(
        expander_config::GKRScheme::Vanilla,
        expander_config::MPIConfig::new(),
    );

    let (simd_input, simd_public_input) = witness.to_simd::<GKRC::SimdCircuitField>();
    println!("{} {}", simd_input.len(), simd_public_input.len());

    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // prove
    expander_circuit.evaluate();
    let mut prover = gkr::Prover::new(&config);
    prover.prepare_mem(&expander_circuit);
    let (claimed_v, proof) = prover.prove(&mut expander_circuit);

    println!("Proved");
    // verify
    let verifier = gkr::Verifier::new(&config);
    assert!(verifier.verify(
        &mut expander_circuit,
        &simd_public_input,
        &claimed_v,
        &proof
    ));
    println!("Verified");

}

//#[test]
#[allow(dead_code)]
fn run_gf2() {
    run_main::<GF2Config, GF2ExtConfigSha2>();
    run_main::<GF2Config, GF2ExtConfigKeccak>();
}

//#[test]
#[allow(dead_code)]
fn run_m31() {
    run_main::<M31Config, M31ExtConfigSha2>();
    run_main::<M31Config, M31ExtConfigKeccak>();
}

//#[test]
#[allow(dead_code)]
fn run_bn254() {
    run_main::<BN254Config, BN254ConfigSha2>();
    run_main::<BN254Config, BN254ConfigKeccak>();
}



mod io_reader {
    use ethnum::U256;
    use std::{io::Read, process::Output};
    use arith::FieldForECC;
    use serde::Deserialize;

    use super::Circuit;

    use expander_compiler::frontend::*;

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct InputData {
        pub(crate) RATE_OF_DECAY: u64,
        pub(crate) RATE_OF_RECOVERY: u64,
        pub(crate) FLATTENING_COEFFICIENT: u64,
        pub(crate) PROOF_SIZE_THRESHOLD: u64,
        pub(crate) PROOF_SIZE_WEIGHT: u64,
        pub(crate) RESPONSE_TIME_WEIGHT: u64,
        pub(crate) MAXIMUM_RESPONSE_TIME_DECIMAL: u64,
        pub(crate) maximum_score: Vec<u64>,
        pub(crate) previous_score: Vec<u64>,
        pub(crate) verified: Vec<u64>,
        pub(crate) proof_size: Vec<u64>,
        pub(crate) response_time: Vec<u64>,
        pub(crate) maximum_response_time: Vec<u64>,
        pub(crate) minimum_response_time: Vec<u64>,
        pub(crate) block_number: Vec<u64>,
        pub(crate) validator_uid: Vec<u64>,
        pub(crate) miner_uid: Vec<u64>,
        pub(crate) scaling: u64,
    }

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) RATE_OF_DECAY: u64,
        pub(crate) RATE_OF_RECOVERY: u64,
        pub(crate) FLATTENING_COEFFICIENT: u64,
        pub(crate) PROOF_SIZE_THRESHOLD: u64,
        pub(crate) PROOF_SIZE_WEIGHT: u64,
        pub(crate) RESPONSE_TIME_WEIGHT: u64,
        pub(crate) MAXIMUM_RESPONSE_TIME_DECIMAL: u64,
        pub(crate) maximum_score: Vec<u64>,
        pub(crate) previous_score: Vec<u64>,
        pub(crate) verified: Vec<u64>,
        pub(crate) proof_size: Vec<u64>,
        pub(crate) response_time: Vec<u64>,
        pub(crate) maximum_response_time: Vec<u64>,
        pub(crate) minimum_response_time: Vec<u64>,
        pub(crate) block_number: Vec<u64>,
        pub(crate) validator_uid: Vec<u64>,
        pub(crate) miner_uid: Vec<u64>,
        pub(crate) scaling: u64,
    }

    pub(crate) fn input_data_from_json<C: Config, GKRC>(file_path: &str, mut assignment: Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>, 
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: InputData = serde_json::from_str(&contents).unwrap();

        // Initialize the circuit assignment
    

        // assignment = setup_output(assignment, input_data.clone());

        // Map the first 7 `u8` variables in the circuit
        let u8_vars = [
            data.RATE_OF_DECAY, data.RATE_OF_RECOVERY, data.FLATTENING_COEFFICIENT, data.PROOF_SIZE_THRESHOLD,
            data.PROOF_SIZE_WEIGHT, data.RESPONSE_TIME_WEIGHT, data.MAXIMUM_RESPONSE_TIME_DECIMAL,
        ];

        for (k, &var) in u8_vars.iter().enumerate() {
            // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
            assignment.input[0][k] = C::CircuitField::from_u256(U256::from(var)) ; // Treat the u8 as a u64 for BN254

        }

        // Map each `Vec<u64>` variable (var8 to var18) in the circuit
        let vec_vars = [
            data.maximum_score, data.previous_score, data.verified, data.proof_size,
            data.response_time, data.maximum_response_time, data.minimum_response_time, data.block_number,
            data.validator_uid, data.miner_uid
        ];
        for (k, var_vec) in vec_vars.iter().enumerate() {
            // Each `Vec<u64>` corresponds to a sequence of u64 values
            for (i, &var) in var_vec.iter().enumerate() {
                // Directly assign each u32 to the corresponding position in the `assignment.p`
                assignment.input[i][7 + k] = C::CircuitField::from_u256(U256::from(var));
            }
        }

        assignment.input[0][17] = C::CircuitField::from_u256(U256::from(data.scaling)); // Store var18 as BN254
        // Return the assignment wrapped in a Result
        assignment
    }

    pub(crate) fn output_data_from_json<C: Config, GKRC>(file_path: &str, mut assignment: Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>, 
    {
        // Read the JSON file into a string
        let mut file = std::fs::File::open(file_path).expect("Unable to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Unable to read file");


        // Deserialize the JSON into the InputData struct
        let data: OutputData = serde_json::from_str(&contents).unwrap();

        // Initialize the circuit assignment
    

        // assignment = setup_output(assignment, input_data.clone());

        // Map the first 7 `u8` variables in the circuit
        let u8_vars = [
            data.RATE_OF_DECAY, data.RATE_OF_RECOVERY, data.FLATTENING_COEFFICIENT, data.PROOF_SIZE_THRESHOLD,
            data.PROOF_SIZE_WEIGHT, data.RESPONSE_TIME_WEIGHT, data.MAXIMUM_RESPONSE_TIME_DECIMAL,
        ];

        for (k, &var) in u8_vars.iter().enumerate() {
            // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
            assignment.output[0][k] = C::CircuitField::from_u256(U256::from(var)) ; // Treat the u8 as a u64 for BN254

        }

        // Map each `Vec<u64>` variable (var8 to var18) in the circuit
        let vec_vars = [
            data.maximum_score, data.previous_score, data.verified, data.proof_size,
            data.response_time, data.maximum_response_time, data.minimum_response_time, data.block_number,
            data.validator_uid, data.miner_uid
        ];
        for (k, var_vec) in vec_vars.iter().enumerate() {
            // Each `Vec<u64>` corresponds to a sequence of u64 values
            for (i, &var) in var_vec.iter().enumerate() {
                // Directly assign each u32 to the corresponding position in the `assignment.p`
                assignment.output[i][7 + k] = C::CircuitField::from_u256(U256::from(var));
            }
        }

        assignment.output[0][17] = C::CircuitField::from_u256(U256::from(data.scaling)); // Store var18 as BN254
        // Return the assignment wrapped in a Result
        assignment
    }
}



fn main(){
    run_gf2();
    run_m31();
    run_bn254();
}