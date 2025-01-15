
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};
use clap::{Command, Arg};


/* 
Step 3: scalar times matrix product of two matrices of compatible dimensions, plus a third matrix of campatible dimensions.
scaling factor alpha is an integer
matrix a has shape (m, n)
matrix b has shape (n, k)
matrix c has shape (m, k)
scaled matrix product plus matrix alpha ab + c has shape (m, k)
*/

const N_ROWS_A: usize = 3; // m
const N_COLS_A: usize = 4; // n
const N_ROWS_B: usize = 4; // n
const N_COLS_B: usize = 2; // k
const N_ROWS_C: usize = 3; // m
const N_COLS_C: usize = 2; // k

declare_circuit!(Circuit {
    alpha: Variable, // scaling factor
    matrix_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    matrix_b: [[Variable; N_COLS_B]; N_ROWS_B], // shape (n, k)
    matrix_c: [[Variable; N_COLS_C]; N_ROWS_C], // shape (m, k)
    scaled_matrix_product_sum_alpha_ab_plus_c: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define(&self, api: &mut API<C>) {      
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_B {
                let mut scaled_row_col_product_sum: Variable = api.constant(0);
                for k in 0..N_COLS_A {
                    let element_product = api.mul(self.matrix_a[i][k], self.matrix_b[k][j]);
                    scaled_row_col_product_sum = api.add(scaled_row_col_product_sum, element_product);                                      
                }
                scaled_row_col_product_sum = api.mul(scaled_row_col_product_sum, self.alpha);
                scaled_row_col_product_sum = api.add(scaled_row_col_product_sum, self.matrix_c[i][j]);
                api.assert_is_equal(self.scaled_matrix_product_sum_alpha_ab_plus_c[i][j], scaled_row_col_product_sum);               
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
        pub(crate) alpha: u64,
        pub(crate) matrix_a: Vec<Vec<u64>>, // Shape (m, n)  
        pub(crate) matrix_b: Vec<Vec<u64>>, // Shape (n, k) 
        pub(crate) matrix_c: Vec<Vec<u64>>, // Shape (m, k)
    }

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) scaled_matrix_product_sum_alpha_ab_plus_c: Vec<Vec<u64>>, // Shape (m, k)
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
        assignment.alpha = C::CircuitField::from_u256(U256::from(data.alpha));

        let rows_a = data.matrix_a.len();  
        let cols_a = if rows_a > 0 { data.matrix_a[0].len() } else { 0 };  
        println!("matrix a shape: ({}, {})", rows_a, cols_a);  
        
        for (i, row) in data.matrix_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_a[i][j] = C::CircuitField::from_u256(U256::from(element));
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

        let rows_c = data.matrix_c.len();  
        let cols_c = if rows_c > 0 { data.matrix_c[0].len() } else { 0 };  
        println!("matrix c shape: ({}, {})", rows_c, cols_c); 

        for (i, row) in data.matrix_c.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.matrix_c[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
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
        let rows_abc = data.scaled_matrix_product_sum_alpha_ab_plus_c.len();  
        let cols_abc = if rows_abc > 0 { data.scaled_matrix_product_sum_alpha_ab_plus_c[0].len() } else { 0 };  
        println!("scaled matrix product alpha ab plus matrix c shape: ({}, {})", rows_abc, cols_abc); 

        for (i, row) in data.scaled_matrix_product_sum_alpha_ab_plus_c.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.scaled_matrix_product_sum_alpha_ab_plus_c[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }
        assignment
    }
}

fn run_main<C: Config, GKRC>()
where
    GKRC: expander_config::GKRConfig<CircuitField = C::CircuitField>,
{

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

fn main(){
    run_gf2();
    run_m31();
    run_bn254();
}