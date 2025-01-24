
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};
use clap::{Command, Arg};
use peakmem_alloc::*;
use std::alloc::System;
use std::mem;
use std::time::{Instant};

// :)
#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;
const LENGTH: usize = 256;

/*
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
 */

// Specify input and output structure
// This will indicate the input layer and output layer of the circuit, so be careful with how it is defined
// Later, we define how the inputs get read into the input layer
declare_circuit!(ReLUCircuit {
    input: [[Variable; LENGTH]; 2],
    output: [Variable; LENGTH],
});

// Assume 0 is negative and 1 is positive
fn relu<C: Config>(api: &mut API<C>, x: Variable, sign: Variable) -> Variable {
    let sign_2 = api.sub(1, sign);
    api.mul(x,sign_2)
}

impl<C: Config> Define<C> for ReLUCircuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define(&self, api: &mut API<C>) {
            for i in 0..LENGTH {
                // Iterate over each input/output pair (one per batch)
                let x = relu(api, self.input[0][i], self.input[1][i]);
                api.assert_is_equal(x, self.output[i]);

                // let bits = to_binary_constrained(api, self.input[0][i], 32)
        }
    }
}
/*
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
 */

mod io_reader {
    use ethnum::U256;
    use std::io::Read;
    use arith::FieldForECC;
    use serde::Deserialize;

    use crate::LENGTH;

    use super::ReLUCircuit;

    use expander_compiler::frontend::*;
    /*
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
    */

    //This is the data structure for the input data to be read in from the json file
    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct InputData {
        pub(crate) inputs_1: Vec<u64>,
        pub(crate) inputs_2: Vec<u64>,
    }

    //This is the data structure for the output data to be read in from the json file
    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) outputs: Vec<u64>,
    }

    // Read in input data from json file. Here, we focus on reading the inputs into the input layer of the circuit in a way that makes sense to us
    pub(crate) fn input_data_from_json<C: Config, GKRC>(file_path: &str, mut assignment: ReLUCircuit<<C as Config>::CircuitField>) -> ReLUCircuit<<C as expander_compiler::frontend::Config>::CircuitField>
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


        // Assign inputs to assignment
        let u8_vars = [
            data.inputs_1, data.inputs_2
        ];

        for (j, var_vec) in u8_vars.iter().enumerate() {
            for (k, &var) in var_vec.iter().enumerate() {
                assignment.input[j][k] = C::CircuitField::from_u256(U256::from(var)) ; // Treat the u8 as a u64 for Field
            }
        }
        // Return the assignment
        assignment
    }

    // Read in output data from json file. Here, we focus on reading the outputs into the output layer of the circuit in a way that makes sense to us
    pub(crate) fn output_data_from_json<C: Config, GKRC>(file_path: &str, mut assignment: ReLUCircuit<<C as Config>::CircuitField>) -> ReLUCircuit<<C as expander_compiler::frontend::Config>::CircuitField>
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

        // Assign inputs to assignment

        for k in 0..LENGTH {
            assignment.output[k] = C::CircuitField::from_u256(U256::from(data.outputs[k])) ; // Treat the u8 as a u64 for Field
        }
        assignment
    }
    /*
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
    */
}

fn run_main<C: Config, GKRC>()
where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>,
{
    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now(); 
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
    let compile_result: CompileResult<C> = compile(&ReLUCircuit::default()).unwrap();

    let assignment = ReLUCircuit::<C::CircuitField>::default();

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

    println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}", 
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!("Time elapsed: {}.{} seconds", duration.as_secs(), duration.subsec_millis())

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

fn main(){
    run_gf2();
    run_m31();
    run_bn254();
}