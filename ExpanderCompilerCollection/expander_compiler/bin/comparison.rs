
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};
use clap::{Command, Arg};
use extra::UnconstrainedAPI;

use peakmem_alloc::*;
use std::alloc::System;
use std::fmt::write;
use std::{array, mem};
use std::time::{Instant};

use csv::{WriterBuilder, ReaderBuilder};
use serde::Serialize;
use std::{error::Error, fs::OpenOptions, io::Write};

#[derive(Serialize)]
struct Metrics {
    experiment_name: String,
    proof_size: usize,
    max_mem: f32,
    proof_time: u128,
}


fn write_metrics(metrics: Vec<Metrics>){
    let file_path = "analysis/metrics_output.csv";
    
    // Open the file with the option to append (create if it doesn't exist)
    let file = OpenOptions::new()
        .create(true)       // Create the file if it doesn't exist
        .append(true)       // Append to the file
        .open(file_path).unwrap();

    let mut wtr = WriterBuilder::new().has_headers(false).from_writer(file);

    // If the file is empty (no headers), write headers
    let metadata = std::fs::metadata(file_path).unwrap();
    if metadata.len() == 0 {
        wtr.write_record(&["experiment_name", "proof_size", "max_mem", "proof_time"]);
    }


    // Loop through each setting, collect metrics, and append to CSV
    for m in metrics {

        wtr.serialize(m);
    }

    wtr.flush();
    println!("Metrics have been appended to {}", file_path);
}


#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;
// static GLOBAL_PARTIAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

const LENGTH: usize = 256;
//0 is variable length vector, 1 is fixed length vector, 2 is array 
const arraytype: usize = 3;

// 0 is unconstrained default provided by ecc, 1 is our constrained version
const constrained: usize = 1;

const N:usize = 32;



declare_circuit!(Circuit {
    input: [[Variable; LENGTH]; 2],
    output: [Variable; LENGTH],
});

// Version 1
fn to_binary<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    let mut res = Vec::new();
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res.push(api.unconstrained_bit_and(y, 1));
    }
    res
}

fn from_binary<C: Config>(api: &mut API<C>, bits: Vec<Variable>) -> Variable {
    let mut res = api.constant(0);
    for i in 0..bits.len() {
        let coef = 1 << i;
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    res
}

// Version 2
fn to_binary_constrained<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    let mut res = Vec::new();
    let mut count = api.constant(0);
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res.push(api.unconstrained_bit_and(y, 1));
        //Check that value is binary
        let bin_check = api.sub(1,res[i]);
        let bin_check2 = api.mul(bin_check,res[i]);
        api.assert_is_zero(bin_check2);

        //keep running total
        // Not sure if this next line can be used?
        let coef = 1 << i;
        let cur = api.mul(coef, res[i]);
        count = api.add(count, cur);
    }
    api.assert_is_equal(count, x);
    res
}

// Version 1
// fn to_binary<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
fn to_binary_fixed_length<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    // let mut res = Vec::new();
    let mut res= vec![x; n_bits];
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        // res.push(api.unconstrained_bit_and(y, 1));
        res[i] = api.unconstrained_bit_and(y, 1);
        
    }
    res
}

fn from_binary_fixed_length<C: Config>(api: &mut API<C>, bits: Vec<Variable> ) -> Variable {
    let mut res = api.constant(0);
    for i in 0..bits.len() {
        let coef = 1 << i;
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    res
}

// Version 2
fn to_binary_constrained_fixed_length<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> [Variable; N]{
    let mut res = [x; N];
    let mut count = api.constant(0);
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res[i] = api.unconstrained_bit_and(y, 1);

        //Check that value is binary
        let bin_check = api.sub(1,res[i]);
        let bin_check2 = api.mul(bin_check,res[i]);
        api.assert_is_zero(bin_check2);

        //keep running total
        // Not sure if this next line can be used?
        let coef = 1 << i;
        let cur = api.mul(coef, res[i]);
        count = api.add(count, cur);
    }
    api.assert_is_equal(count, x);
    res
}
// Array
fn to_binary_constrained_array<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> [Variable; N] {
    let mut res =[x; N];
    let mut count = api.constant(0);
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res[i] = api.unconstrained_bit_and(y, 1);

        //Check that value is binary
        let bin_check = api.sub(1,res[i]);
        let bin_check2 = api.mul(bin_check,res[i]);
        api.assert_is_zero(bin_check2);

        //keep running total
        // Not sure if this next line can be used?
        let coef = 1 << i;
        let cur = api.mul(coef, res[i]);
        count = api.add(count, cur);
    }
    api.assert_is_equal(count, x);
    res
}

fn to_binary_array<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> [Variable; N] {
    // let mut res = Vec::new();
    let mut res = [x; N];
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        // res.push(api.unconstrained_bit_and(y, 1));
        res[i] = api.unconstrained_bit_and(y, 1);
        
    }
    res
}

fn from_binary_array<C: Config>(api: &mut API<C>, bits: [Variable; N] ) -> Variable {
    let mut res = api.constant(0);
    for i in 0..bits.len() {
        let coef = 1 << i;
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    res
}
// Capacity Vector
fn to_binary_constrained_capacity<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    let mut res = Vec::with_capacity(n_bits);
    let mut count = api.constant(0);
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res.push(api.unconstrained_bit_and(y, 1));

        //Check that value is binary
        let bin_check = api.sub(1,res[i]);
        let bin_check2 = api.mul(bin_check,res[i]);
        api.assert_is_zero(bin_check2);

        //keep running total
        // Not sure if this next line can be used?
        let coef = 1 << i;
        let cur = api.mul(coef, res[i]);
        count = api.add(count, cur);
    }
    api.assert_is_equal(count, x);
    res
}

fn to_binary_capacity<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    // let mut res = Vec::new();
    let mut res = Vec::with_capacity(n_bits);
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        // res.push(api.unconstrained_bit_and(y, 1));
        res.push(api.unconstrained_bit_and(y, 1));
        
    }
    res
}

fn from_binary_capacity<C: Config>(api: &mut API<C>, bits: Vec<Variable> ) -> Variable {
    let mut res = api.constant(0);
    for i in 0..bits.len() {
        let coef = 1 << i;
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    res
}


impl<C: Config> Define<C> for Circuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define(&self, api: &mut API<C>) {
        if arraytype == 0{
            for i in 0..LENGTH {
                // Iterate over each input/output pair (one per batch)
                if constrained == 0{
                    let bits = to_binary(api, self.input[0][i], 32);
                    let x = from_binary(api, bits);
                    api.assert_is_equal(x, self.input[0][i]);
                }
                else{
                    let bits = to_binary_constrained(api, self.input[0][i], 32);
                }
            }
        }
        else if arraytype == 1{
            for i in 0..LENGTH {
                // Iterate over each input/output pair (one per batch)
                if constrained == 0{
                    let bits = to_binary_fixed_length(api, self.input[0][i], 32);
                    let x = from_binary_fixed_length(api, bits);
                    api.assert_is_equal(x, self.input[0][i]);
                }
                else{
                    let bits = to_binary_constrained_fixed_length(api, self.input[0][i], 32);
                }
            }
        }
        else if arraytype == 2{
            for i in 0..LENGTH {
                // Iterate over each input/output pair (one per batch)
                if constrained == 0{
                    let bits = to_binary_array(api, self.input[0][i], 32);
                    let x = from_binary_array(api, bits);
                    api.assert_is_equal(x, self.input[0][i]);
                }
                else{
                    let bits = to_binary_constrained_array(api, self.input[0][i], 32);
                }
            }
        }
        else if arraytype == 3{
            for i in 0..LENGTH {
                // Iterate over each input/output pair (one per batch)
                if constrained == 0{
                    let bits = to_binary_capacity(api, self.input[0][i], 32);
                    let x = from_binary_capacity(api, bits);
                    api.assert_is_equal(x, self.input[0][i]);
                }
                else{
                    let bits = to_binary_constrained_capacity(api, self.input[0][i], 32);
                }
            }
        }
    }
}

mod io_reader {
    use ethnum::U256;
    use std::io::Read;
    use arith::FieldForECC;
    use serde::Deserialize;

    use crate::LENGTH;

    use super::Circuit;

    use expander_compiler::frontend::*;

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct InputData {
        pub(crate) inputs_1: Vec<u64>,
        pub(crate) inputs_2: Vec<u64>,
    }

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) outputs: Vec<i64>,
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


        // Assign inputs to assignment
        let u8_vars = [
            data.inputs_1, data.inputs_2
        ];

        for (j, var_vec) in u8_vars.iter().enumerate() {
            for (k, &var) in var_vec.iter().enumerate() {
            // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
                assignment.input[j][k] = C::CircuitField::from_u256(U256::from(var)) ; // Treat the u8 as a u64 for BN254
            }

        }
        // Return the assignment
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

        // Assign inputs to assignment

        for k in 0..LENGTH {
            // For each u8 variable, store it directly as a `u64` in the BN254 field (BN254 can handle u64)
            if data.outputs[k] == -1 {
                assignment.output[k] = C::CircuitField::from_u256(U256::from(0 as u8)) -  C::CircuitField::from_u256(U256::from(1 as u8)); // Treat the u8 as a u64 for BN254
            }
            else{
                assignment.output[k] = C::CircuitField::from_u256(U256::from(data.outputs[k] as u64)) ; // Treat the u8 as a u64 for BN254
            }
        }
        assignment
    }
}

fn run_main<C: Config, GKRC>(experiment_name: &String)
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
    let duration = start.elapsed();
    let memory = GLOBAL.get_peak_memory();
    let size = mem::size_of_val(&proof) + mem::size_of_val(&claimed_v);
    
    println!("Verified");
    println!("Size of proof: {} bytes", size);
    println!(
        "Peak Memory used Overall : {:.2}", 
        memory as f64 / (1024.0 * 1024.0)
    );
    
    println!("Time elapsed: {}.{} seconds", duration.as_secs(), duration.subsec_millis());
    let metrics = Metrics{
        experiment_name: experiment_name.to_string(),
        proof_size: size,
        max_mem: memory as f32 / (1024.0 * 1024.0),
        proof_time: duration.as_millis()
    };

    write_metrics(vec![metrics]);
}

//#[test]
#[allow(dead_code)]
fn run_gf2(experiment_name:&String) {
    run_main::<GF2Config, GF2ExtConfigSha2>(&experiment_name);
    run_main::<GF2Config, GF2ExtConfigKeccak>(&experiment_name);
}

//#[test]
#[allow(dead_code)]
fn run_m31(experiment_name:&String) {
    run_main::<M31Config, M31ExtConfigSha2>(&experiment_name);
    run_main::<M31Config, M31ExtConfigKeccak>(&experiment_name);
}

//#[test]
#[allow(dead_code)]
fn run_bn254(experiment_name:&String) {
    run_main::<BN254Config, BN254ConfigSha2>(&experiment_name);
    run_main::<BN254Config, BN254ConfigKeccak>(&experiment_name);
}

fn main(){
    let mut array_type = "";
    if arraytype == 0{
        array_type = "variablevector";
    }
    else if arraytype == 1 {
        array_type = "fixedvector";
    }
    else if arraytype == 2{
        array_type = "array";
    }
    else {
        array_type = "vectorcapacity";
    }
    let mut con = "";
    if constrained == 0 {
        con = "ecc";
    }
    else{
        con = "IL";
    }
    let experiment_name1: String = format!("to_binary_m31_{}_{}",array_type,con);
    
    // run_gf2();
    run_m31(&experiment_name1);
    let experiment_name2: String = format!("to_binary_bn254_{}_{}",array_type,con);
    run_bn254(&experiment_name2);
}