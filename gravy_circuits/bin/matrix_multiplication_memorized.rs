
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};


use clap::{Command, Arg};
use peakmem_alloc::*;
use std::alloc::System;
use std::mem;
use std::time::Instant;
use serde::Deserialize;
use lazy_static::lazy_static;



#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;

/* 
Part 2 (memorization), Step 1: vanilla matrix multiplication of two matrices of compatible dimensions.
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix product ab has shape (m, k)
*/

const N_ROWS_A: usize = 1; // m
const N_COLS_A: usize = 1568; // n
// const N_ROWS_B: usize = N_COLS_A; // n
const N_COLS_B: usize = 256; // k

//Define structure of inputs, weights and output
#[derive(Deserialize)]
#[derive(Clone)]
struct WeightsData {
    matrix_b: Vec<Vec<u64>>,
} 

#[derive(Deserialize)]
#[derive(Clone)]
struct InputData {
    matrix_a: Vec<Vec<u64>>,
} 

#[derive(Deserialize)]
#[derive(Clone)]
struct OutputData {
    matrix_product_ab: Vec<Vec<u64>>,
} 

// This reads the weights json into a string
const MATRIX_WEIGHTS_FILE: &str = include_str!("../../../weights/matrix_multiplication_memorized_weights.json");

//lazy static macro, forces this to be done at compile time (and allows for a constant of this weights variable)
// Weights will be read in
lazy_static! {
    static ref weights: Vec<Vec<u64>> = {
        let x: WeightsData = serde_json::from_str(MATRIX_WEIGHTS_FILE).expect("JSON was not well-formatted");
        
        let mut y: Vec<Vec<u64>> = Vec::new();
        for (_, row) in x.matrix_b.iter().enumerate() {
            let mut z: Vec<u64> = Vec::new();
            for (_, &element) in row.iter().enumerate() {
                z.push(element);
            }
            // println!("{}", row.len());
            y.push(z);
        }
        // println!("{}", y.len());
        y
        };

}

#[allow(dead_code)]
fn product_sub_circuit<C: Config>(api: &mut API<C>, inputs: &Vec<Variable>) -> Vec<Variable>  {
    let n = inputs.len()/2; // Assuming inputs are concatenated row and column
    // let mut out: Vec<Variable> = Vec::new();
    let mut sum = api.constant(0);

    for k in 0..n {
        let x = api.mul(inputs[k], inputs[n + k]);
        sum = api.add(sum, x);
    }
    vec![sum]
}

//Arrays
#[allow(dead_code)]
fn matrix_multplication_array<C: Config, const M: usize, const N: usize, const K: usize>(api: &mut API<C>, matrix_a: [[Variable; N]; M], matrix_b: Vec<Vec<Variable>>) -> [[Variable; K]; M]{
    let mut out = [[Variable::default(); K]; M];
    for i in 0..M {
        for j in 0..K {
            // Prepare inputs as concatenated row and column
            // api.add(C::CircuitField::from(weights[0][0] as u32),self.matrix_a[0][0]);
            let mut inputs: Vec<Variable> = Vec::new();
            for k in 0..N {
                inputs.push(matrix_a[i][k]);
            }
            for k in 0..N {
                inputs.push(matrix_b[k][j]);
            }
            // Use MemorizedSimpleCall for the row-column dot product
            out[i][j] = api.memorized_simple_call(product_sub_circuit, &inputs)[0];
            // api.assert_is_equal(self.matrix_product_ab[i][j], prod[0]);
        }
    }
    out
}

#[allow(dead_code)]
fn matrix_multplication_naive_array<C: Config, const M: usize, const N: usize, const K: usize>(api: &mut API<C>, matrix_a: [[Variable; N]; M], matrix_b: [[Variable; K]; N]) -> [[Variable; K]; M]{
    let mut out = [[Variable::default(); K]; M];
    for i in 0..M {
        for j in 0..K {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out[i][j] = row_col_product;               
        }
    }
    out
}

#[allow(dead_code)]
fn matrix_multplication_naive2_array<C: Config, const M: usize, const N: usize, const K: usize>(api: &mut API<C>, matrix_a: [[Variable; N]; M], matrix_b: Vec<Vec<Variable>>) -> [[Variable; K]; M]{
    let mut out = [[Variable::default(); K]; M];
    for i in 0..M {
        for j in 0..K {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out[i][j] = row_col_product;
        }
    }
    out
}

#[allow(dead_code)]
fn matrix_multplication_naive3_array<C: Config, const M: usize, const N: usize, const K: usize>(api: &mut API<C>, matrix_a: [[Variable; N]; M], matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..M {
        let mut row_out: Vec<Variable> = Vec::new();
        for j in 0..K {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..N {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            row_out.push(row_col_product);
        }
        out.push(row_out);
    }
    out
}
// Vector of Vectors 

#[allow(dead_code)]
fn matrix_multplication<C: Config>(api: &mut API<C>, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut out_rows: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            // Prepare inputs as concatenated row and column
            // api.add(C::CircuitField::from(weights[0][0] as u32),self.matrix_a[0][0]);
            let mut inputs: Vec<Variable> = Vec::new();
            for k in 0..matrix_b.len() {
                inputs.push(matrix_a[i][k]);
            }
            for k in 0..matrix_b.len() {
                inputs.push(matrix_b[k][j]);
            }
            // Use MemorizedSimpleCall for the row-column dot product
            out_rows.push(api.memorized_simple_call(product_sub_circuit, &inputs)[0]);
            // api.assert_is_equal(self.matrix_product_ab[i][j], prod[0]);
        }
        out.push(out_rows);
    }
    out
}

#[allow(dead_code)]
fn matrix_multplication_naive<C: Config>(api: &mut API<C>, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();    
    for i in 0..matrix_a.len() {
        let mut out_rows: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out_rows.push(row_col_product);
        }
        out.push(out_rows)
    }
    out
}

#[allow(dead_code)]
fn matrix_multplication_naive2<C: Config>(api: &mut API<C>, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut out_row: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            out_row.push(row_col_product);
        }
        out.push(out_row);
    }
    out
}


#[allow(dead_code)]
fn matrix_multplication_naive3<C: Config>(api: &mut API<C>, matrix_a: Vec<Vec<Variable>>, matrix_b: Vec<Vec<Variable>>) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for i in 0..matrix_a.len() {
        let mut row_out: Vec<Variable> = Vec::new();
        for j in 0..matrix_b[0].len() {
            let mut row_col_product: Variable = api.constant(0);
            for k in 0..matrix_b.len() {
                let element_product = api.mul(matrix_a[i][k], matrix_b[k][j]);
                row_col_product = api.add(row_col_product, element_product);
            }
            row_out.push(row_col_product);
        }
        out.push(row_out);
    }
    out
}

fn two_d_array_to_vec<const M: usize, const N: usize>(matrix:[[Variable; N]; M]) -> Vec<Vec<Variable>>{
    matrix.iter()
    .map(|row| row.to_vec())
    .collect()
                                    
}

declare_circuit!(Circuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_product_ab: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});
// Memorization, in a better place
impl<C: Config> Define<C> for Circuit<Variable,> {
    fn define(&self, api: &mut API<C>) {
        // Bring the weights into the circuit as constants

        let weights_matrix_multiplication: Vec<Vec<Variable>> = weights.clone()
            .into_iter()
            .map(|row| row.into_iter().map(|x| api.constant(x as u32)).collect())
            .collect();
        
        // Compute matrix multiplication
        // let out:[[Variable; 256]; 1] = matrix_multplication_array(api, self.matrix_a,  weights_matrix_multiplication);
        // let out:[[Variable; N_COLS_B]; N_ROWS_A]  = matrix_multplication_naive2_array(api, self.matrix_a,  weights_matrix_multiplication);
        // let out:Vec<Vec<Variable>>  = matrix_multplication_naive3_array::<C,N_ROWS_A, N_COLS_A, N_COLS_B>(api, self.matrix_a,  weights_matrix_multiplication);


        // let out = matrix_multplication(api, two_d_array_to_vec(self.matrix_a),  weights_matrix_multiplication);
        let out = matrix_multplication_naive(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication);
        // let out = matrix_multplication_naive2(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication);
        // let out = matrix_multplication_naive3(api, two_d_array_to_vec(self.matrix_a), weights_matrix_multiplication);

        //Assert output of matrix multiplication
        for (j,row) in out.iter().enumerate() {
            for (k,&element) in row.iter().enumerate() {
                api.assert_is_equal(self.matrix_product_ab[j][k], element);
            }
        }

    }
}

mod io_reader {
    use ethnum::U256;
    use std::io::Read;
    use arith::FieldForECC;
    use serde::de::DeserializeOwned;


    use super::{Circuit, InputData, OutputData};

    use expander_compiler::frontend::*;
    /*
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
    */

    //This is the data structure for the input data to be read in from the json file

    // Read in input data from json file. Here, we focus on reading the inputs into the input layer of the circuit in a way that makes sense to us
    pub(crate) fn input_data_from_json<C: Config, GKRC>(file_path: &str, mut assignment: Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>, 
    {
        // Read the JSON file into a string
        let input_data: InputData = read_data_from_json(file_path);
        for (i, row) in input_data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from(element as u32) ;
            }
        }
        // Return the assignment
        assignment
    }

    // Read in output data from json file. Here, we focus on reading the outputs into the output layer of the circuit in a way that makes sense to us
    pub(crate) fn output_data_from_json<C: Config, GKRC>(file_path: &str, mut assignment: Circuit<<C as Config>::CircuitField>) -> Circuit<<C as expander_compiler::frontend::Config>::CircuitField>
    where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>, 
    {
        // Read the JSON file into a string
        let input_data: OutputData = read_data_from_json(file_path);
        for (i, row) in input_data.matrix_product_ab.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_product_ab[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }
        // Return the assignment
        assignment
    }
    /*
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
    */
    pub(crate) fn read_data_from_json<I>(file_path: &str) -> I
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
    println!("result compiled");

    println!(
        "Peak Memory used Overall : {:.2}", 
        GLOBAL.get_peak_memory() as f64 / (1024.0 * 1024.0)
    );
    let duration = start.elapsed();
    println!("Time elapsed: {}.{} seconds", duration.as_secs(), duration.subsec_millis());

    GLOBAL.reset_peak_memory(); // Note that other threads may impact the peak memory computation.
    let start = Instant::now(); 

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
    run_main::<BN254Config, BN254ConfigKeccak>();
}

fn main(){
    // run_gf2();
    // run_m31();
    run_bn254();
}