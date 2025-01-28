use ark_std::test_rng;
use ethnum::intrinsics::signed;
use ethnum::u256;
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
use extra::UnconstrainedAPI;


#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;
const SIZE1: usize = 4;
const SIZE2: usize = 8;
const SIZE3: usize = 16;


/*
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
 */

// Specify input and output structure
// This will indicate the input layer and output layer of the circuit, so be careful with how it is defined
// Later, we define how the inputs get read into the input layer

// Version 1
fn to_binary_ecc<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
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
fn to_binary<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
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
//Version 3

//What do we want?
//If x > p/2
// Then, take the negation of x

fn to_int_for_twos_comp<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Variable{

    let half = api.unconstrained_pow(2,n_bits as u32);
    //is 1 if x is neg, 0 if x is pos
    let sign = api.unconstrained_greater_eq(x, half);
    // Still need to add the multiplication of negative 1
    // let bit_max = api.unconstrained_pow(2,n_bits as u32);

    let twos_comp_add = api.unconstrained_add(x, half);
    let x_if_negative = api.unconstrained_mul(sign, twos_comp_add);

    let regular_x_add = api.unconstrained_bit_xor(1, sign);
    let x_if_positive = api.unconstrained_mul(regular_x_add, x);

    let new_x = api.unconstrained_add(x_if_negative, x_if_positive);

    return new_x;

}

fn to_binary_2s<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    // let mut res = Vec::new();
    
    
    // The following code to generate new_x is tested and confirmed works
    let new_x = to_int_for_twos_comp(api, x, n_bits);

    let bits = to_binary_ecc(api, new_x, n_bits );

    // bits.push(sign);

    bits
}

fn binary_check<C: Config>(api: &mut API<C>, bits: &Vec<Variable>) {
    // let mut out: Vec<Variable> = Vec::new();
    for i in 0..bits.len(){
        let bin_check = api.sub(1,bits[i]);
        let x = api.mul(bin_check,bits[i]);
        api.assert_is_zero(x);
        // out.push(api.mul(bin_check,bits[i]));
    }
}

fn from_binary_2s<C: Config>(api: &mut API<C>, bits: &Vec<Variable>, n_bits: usize) -> Variable {
    let mut res = api.constant(0);
    let length = n_bits - 1;
    for i in 0..length {
        let coef = 1 << i;
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    binary_check(api, bits);
    let cur = api.mul(1 << length,bits[length]);
    // let temp = api.constant(0);
    let out = api.neg(cur);
    api.add(out, res)
}



fn get_sign_simple<C: Config>(api: &mut API<C>, x: &Vec<Variable>) -> Vec<Variable> {
    let mut out: Vec<Variable> = Vec::with_capacity(x.len());
    for i in 0..x.len(){
        out.push(to_binary(api, x[i], 32)[31]);
    }
    out
}


// Assume 1 is negative and 0 is positive
fn relu_single<C: Config>(api: &mut API<C>, x: Variable, sign: Variable) -> Variable {
    let sign_2 = api.sub(1, sign);
    api.mul(x,sign_2)
}

fn relu_simple_call<C: Config>(api: &mut API<C>, x: &Vec<Variable>) -> Vec<Variable> {
    let mut out = Vec::new();
    let length = x.len()/2;
    for k in 0..length{
        out.push(relu_single(api, x[k], x[k+length]));
        // let sign_2 = api.sub(1, x[k + length]);
        // out.push(api.mul(x[k],sign_2));
    }
    out
}

fn relu_simple_call_2<C: Config>(api: &mut API<C>, x: &Vec<Variable>) -> Vec<Variable> {
    let mut out = Vec::new();
    let sign = api.memorized_simple_call(get_sign_simple, x);
    for k in 0..x.len(){
        out.push(relu_single(api, x[k], sign[k]));
        // let sign_2 = api.sub(1, x[k + length]);
        // out.push(api.mul(x[k],sign_2));
    }
    out
}


//V1 loops through each layer of the matrix and computes relu on individual basis
fn relu_twos_v1<C: Config, const X: usize, const Y: usize, const Z: usize>(api: &mut API<C>, input: [[[Variable; Z]; Y]; X], n_bits: usize) -> Vec<Vec<Vec<Variable>>> {
    // let mut out: Vec<Vec<Vec<Variable>>> = Vec::new();
    let mut out: Vec<Variable> = Vec::new();

    for i in 0..input.len() {
        // let mut out_1:Vec<Vec<Variable>> = Vec::new();
        for j in 0..input[i].len(){
            // let mut out_2:Vec<Variable> = Vec::new();
            for k in 0..input[i][j].len(){
                // Iterate over each input/output pair (one per batch)

                // Get the twos compliment binary representation
                let bits = to_binary_2s(api, input[i][j][k], n_bits);
                //Add constraints to ensure that the bitstring is correct
                let total = from_binary_2s(api, &bits, n_bits);
                api.assert_is_equal(total, input[i][j][k]);

                // Perform relu using sign bit
                let x = relu_single(api, input[i][j][k], bits[n_bits - 1]);
                // out_2.push(x);
                out.push(x);
            }
            // out_1.push(out_2);
        }
        // out.push(out_1)
    }
    // out;
    let (layers, rows, cols) = (X,Y,Z);

    // First chunk into rows (each row has `cols` elements)
    let reshaped_3d: Vec<Vec<Vec<Variable>>> = out
        .chunks(cols)  // Chunks into rows of size `cols`
        .map(|chunk| chunk.to_vec())  // Convert rows into Vec<Variable>
        .collect::<Vec<Vec<Variable>>>()  // Collect the rows into a Vec<Vec<Variable>>
        .chunks(rows)  // Now chunk the rows into layers of size `rows`
        .map(|chunk| chunk.to_vec())  // Convert each layer of rows into Vec<Vec<Variable>>
        .collect();
    reshaped_3d
}

//V1 loops through each layer of the matrix and computes relu on individual basis
fn test_twos_comp_int_verion<C: Config, const X: usize, const Y: usize, const Z: usize>(api: &mut API<C>, input: [[[Variable; Z]; Y]; X], output: [[[Variable; Z]; Y]; X], n_bits: usize) -> [[[Variable; Z]; Y]; X] {
    for i in 0..input.len() {
        for j in 0..input[i].len(){
            for k in 0..input[i][j].len(){
                let new_x = to_int_for_twos_comp(api, input[i][j][k], n_bits);
                api.assert_is_equal(new_x, output[i][j][k]);
            }
        }
    }
    output
}

// memorized_simple call on last dimension of array
fn relu_twos_v2<C: Config, const X: usize, const Y: usize, const Z: usize>(api: &mut API<C>, input: [[[Variable; Z]; Y]; X], output: [[[Variable; Z]; Y]; X]) -> [[[Variable; Z]; Y]; X] {
    for i in 0..input.len() {
        for j in 0..input[i].len(){
            let mut vec1 = input[i][j].to_vec();
            let mut vec2 = api.memorized_simple_call(get_sign_simple, &input[i][j].to_vec());
            vec1.append(&mut vec2);
            let x = api.memorized_simple_call(relu_simple_call, &vec1);
            // let x = relu_simple_call(api, &vec1);


            for k in 0..input[i][j].len(){
                api.assert_is_equal(x[k], output[i][j][k]);

            }
        }
    }
    output
}

//Appears slightly worse than relu twos v2
fn relu_twos_v2_5<C: Config, const X: usize, const Y: usize, const Z: usize>(api: &mut API<C>, input: [[[Variable; Z]; Y]; X], output: [[[Variable; Z]; Y]; X]) -> [[[Variable; Z]; Y]; X] {
    for i in 0..input.len() {
        for j in 0..input[i].len(){
            let mut vec1 = input[i][j].to_vec();
            // let mut vec2 = api.memorized_simple_call(get_sign_simple, &input[i][j].to_vec());
            // vec1.append(&mut vec2);
            let x = api.memorized_simple_call(relu_simple_call_2, &input[i][j].to_vec());
            // let x = relu_simple_call(api, &vec1);


            for k in 0..input[i][j].len(){
                api.assert_is_equal(x[k], output[i][j][k]);

            }
        }
    }
    output
}


//V3 involves flattening the matrix entirely and feeding into memorized_simple call
fn relu_twos_v3<C: Config, const X: usize, const Y: usize, const Z: usize>(api: &mut API<C>, input: [[[Variable; Z]; Y]; X], output: [[[Variable; Z]; Y]; X]) -> [[[Variable; Z]; Y]; X] {
// Flatten the two 3D matrices
        let flattened: Vec<Variable> = input.iter()
            .flat_map(|i| i.iter())
            .flat_map(|j| j.iter())
            .cloned()
            .collect();

        // Determine the dimensions of the 3D matrix
        let out = api.memorized_simple_call(relu_simple_call_2, &flattened);

        let mut iter = out.into_iter();

        for i in 0..output.len() {
            for j in 0..output[i].len(){
                for k in 0..output[i][j].len(){
                    api.assert_is_equal(iter.next().unwrap(), output[i][j][k]);
                }
        }
    }
    output
}


declare_circuit!(Circuit {
    input: [[[Variable; SIZE1]; SIZE2]; SIZE3],
    output: [[[Variable; SIZE1]; SIZE2]; SIZE3],
});
impl<C: Config> Define<C> for Circuit<Variable> {
    // Default circuit for now, ensures input and output are equal
    fn define(&self, api: &mut API<C>) {
        let n_bits = 32;
        
        let out = relu_twos_v1(api, self.input, n_bits);
        // let out = test_twos_comp_int_verion(api, self.input, self.output, n_bits);
        // let out = relu_twos_v2(api, self.input, self.output, n_bits);
        // let out = relu_twos_v2_5(api, self.input, self.output, n_bits);
        // let out = relu_twos_v3(api, self.input, self.output,n_bits);

    
        for i in 0..SIZE3{
            for j in 0..SIZE2{
                for k in 0..SIZE1{
                    api.assert_is_equal(self.output[i][j][k], out[i][j][k]);
                }
            }
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
    use std::{io::Read, ops::Neg};
    use arith::FieldForECC;
    use serde::Deserialize;

    use super::Circuit;

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
        pub(crate) inputs_1: Vec<Vec<Vec<i64>>>,
    }

    //This is the data structure for the output data to be read in from the json file
    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) outputs: Vec<Vec<Vec<u64>>>,
    }

    // Read in input data from json file. Here, we focus on reading the inputs into the input layer of the circuit in a way that makes sense to us
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
        

        for (i,var_vec_vec) in data.inputs_1.iter().enumerate(){
            for (j, var_vec) in var_vec_vec.iter().enumerate(){
                for (k, &var) in var_vec.iter().enumerate(){
                    if var < 0{
                        assignment.input[i][j][k] = C::CircuitField::from(var.abs() as u32).neg();
                    }
                    else{
                        assignment.input[i][j][k] = C::CircuitField::from(var.abs() as u32);
                    }
                    // assignment.input[i][j][k] = C::CircuitField::from_u256(U256::from(var)); 
                }
            }
        }
        assignment
    }

    // Read in output data from json file. Here, we focus on reading the outputs into the output layer of the circuit in a way that makes sense to us
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

        for (i,var_vec_vec) in data.outputs.iter().enumerate(){
            for (j, var_vec) in var_vec_vec.iter().enumerate(){
                for (k, &var) in var_vec.iter().enumerate(){
                    assignment.output[i][j][k] = C::CircuitField::from_u256(U256::from(var));

                }
            }
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
    // run_main::<BN254Config, BN254ConfigKeccak>();
}

fn main(){
    // run_gf2();
    // run_m31();
    run_bn254();
}