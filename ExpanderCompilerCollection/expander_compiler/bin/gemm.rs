
use expander_compiler::frontend::*;
use expander_config::{
    BN254ConfigKeccak, BN254ConfigSha2, GF2ExtConfigKeccak, GF2ExtConfigSha2, M31ExtConfigKeccak,
    M31ExtConfigSha2,
};
use clap::{Command, Arg};


/* 
Matrix a has shape (m, n)
Matrix b has shape (n, k)
matmul(a,b) has shape (m, k)
*/

const N_ROWS_A: usize = 256; // m
const N_COLS_A: usize = 256; // n
const N_COLS_B: usize = 1; // k



declare_circuit!(Circuit {
    input_a: [[Variable; N_COLS_A]; N_ROWS_A], // shape (m, n)
    input_b: [[Variable; N_COLS_B]; N_COLS_A], // shape (n, k)
    output: [[Variable; N_COLS_B]; N_ROWS_A], // shape (m, k)
});

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define(&self, api: &mut API<C>) {
        for i in 0..N_ROWS_A {
            for j in 0..N_COLS_B {
                let mut row_col_product: Variable = api.constant(0);
                for k in 0..N_COLS_A {
                    let element_product = api.mul(self.input_a[i][k], self.input_b[k][j]);
                    row_col_product = api.add(row_col_product, element_product);
                }
                api.assert_is_equal(self.output[i][j], row_col_product);               
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
        pub(crate) input_a: Vec<Vec<u64>>, // Shape (m, n) // Question: type Variable? // Alternative (if dimensions known in advance): [[Variable; N_COLS_A]; N_ROWS_A],
        pub(crate) input_b: Vec<Vec<u64>>, // Shape (n, k) // Question: type Variable? // Alternative (if dimensions known in advance): [[Variable; N_COLS_B]; N_COLS_A],
    }

    #[derive(Deserialize)]
    #[derive(Clone)]
    pub(crate) struct OutputData {
        pub(crate) output: Vec<Vec<u64>>, //  Shape (m, k) // Question: type Variable? // Alternative (if dimensions known in advance): [[Variable; N_COLS_B]; N_ROWS_A],
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

        for (i, row) in data.input_a.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.input_a[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
            }
        }

        for (i, row) in data.input_b.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.input_b[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
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

        for (i, row) in data.output.iter().enumerate() {
            for (j, &element) in row.iter().enumerate() {
                assignment.output[i][j] = C::CircuitField::from_u256(U256::from(element)) ;
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