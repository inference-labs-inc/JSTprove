
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

#[global_allocator]
static GLOBAL: &PeakMemAlloc<System> = &INSTRUMENTED_SYSTEM;
/* 
Step 1: vanilla matrix multiplication of two matrices of compatible dimensions.
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix product ab has shape (m, k)
*/

const N_ROWS_A: usize = 3; // m
const N_COLS_A: usize = 4; // n
const N_ROWS_B: usize = 4; // n
const N_COLS_B: usize = 2; // k


declare_circuit!(Circuit {
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B], // shape (n, k)
    matrix_product_ab: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define(&self, api: &mut API<C>) {      
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_B {
                let mut row_col_product: Variable = api.constant(0);
                for k in 0..N_COLS_A {
                    let element_product = api.mul(self.matrix_a[i][k], self.matrix_b[k][j]);
                    row_col_product = api.add(row_col_product, element_product);
                }
                api.assert_is_equal(self.matrix_product_ab[i][j], row_col_product);               
            }
        }
    }
}

mod io_reader {
    use ethnum::U256;
    use std::io::Read;
    use arith::FieldForECC;
    use serde::Deserialize;

    use super::Circuit;

    use expander_compiler::frontend::*;

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct InputData {
        pub(crate) matrix_a: Vec<Vec<u64>>, // Shape (m, n) // Question: type Variable? // Alternative (if dimensions known in advance): [[Variable; N_COLS_A]; N_ROWS_A],
        pub(crate) matrix_b: Vec<Vec<u64>>, // Shape (n, k) // Question: type Variable? // Alternative (if dimensions known in advance): [[Variable; N_COLS_B]; N_ROWS_B],
    }

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) matrix_product_ab: Vec<Vec<u64>>, //  Shape (m, k) // Question: type Variable? // Alternative (if dimensions known in advance): [[Variable; N_COLS_B]; N_ROWS_A],
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

        let rows_a = data.matrix_a.len();  
        let cols_a = if rows_a > 0 { data.matrix_a[0].len() } else { 0 };  
        println!("matrix a shape: ({}, {})", rows_a, cols_a);  
        
        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }

        let rows_b = data.matrix_b.len();  
        let cols_b = if rows_b > 0 { data.matrix_b[0].len() } else { 0 };  
        println!("matrix b shape: ({}, {})", rows_b, cols_b); 

        for (i, row) in data.matrix_b.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_b[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
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
        let rows_ab = data.matrix_product_ab.len();  
        let cols_ab = if rows_ab > 0 { data.matrix_product_ab[0].len() } else { 0 };  
        println!("matrix product ab shape: ({}, {})", rows_ab, cols_ab); 

        for (i, row) in data.matrix_product_ab.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_product_ab[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
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
    println!("result compiled");
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
    println!("Size of proof: {} bytes", mem::size_of_val(&proof) + mem::size_of_val(&claimed_v));
    println!(
        "Peak Memory used Overall : {:.2}", 
        memory as f64 / (1024.0 * 1024.0)
    );
    
    println!("Time elapsed: {}.{} seconds", duration.as_secs(), duration.subsec_millis());
    let metrics = MetricsWriting::Metrics{
        experiment_name: experiment_name.to_string(),
        proof_size: size,
        max_mem: memory as f32 / (1024.0 * 1024.0),
        proof_time: duration.as_millis()
    };

    MetricsWriting::write_metrics(vec![metrics]);
}

mod MetricsWriting {
    use csv::WriterBuilder;
    use serde::Serialize;
    use std::fs::OpenOptions;

    #[derive(Serialize)]
    pub(crate) struct Metrics {
        pub(crate) experiment_name: String,
        pub(crate) proof_size: usize,
        pub(crate) max_mem: f32,
        pub(crate) proof_time: u128,
    }

    pub(crate) fn write_metrics(metrics: Vec<Metrics>){
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
    for i in 0..25{
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
}