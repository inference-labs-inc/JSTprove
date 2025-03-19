use expander_compiler::frontend::*;
use gravy_circuits::runner::main_runner::{run_compile_and_serialize, run_main, run_verify_no_circuit, run_witness_and_proof};
use gravy_circuits::io::io_reader::{FileReader, IOReader};
use serde::Deserialize;
use clap::{Arg, Command};


declare_circuit!(Circuit {
    input_a: Variable,
    input_b: Variable,
    output: Variable, 
});

declare_circuit!(DummyCircuit {
    input_a: Variable,
    input_b: Variable,
    output: Variable, 
});

impl<C: Config> Define<C> for DummyCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        api.assert_is_equal(0, 10);
    }
}

//Still to factor this out

impl<C: Config> Define<C> for Circuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let out = api.add(self.input_a, self.input_b);

        api.assert_is_equal(out, self.output);
    }
}

#[derive(Deserialize, Clone)]
struct InputData {
    input_a: u32, // Shape (m, n)
    input_b: u32, // Shape (n, k)
}

//This is the data structure for the output data to be read in from the json file
#[derive(Deserialize, Clone)]
struct OutputData {
    output: u32,
}

impl<C: Config> IOReader<DummyCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: DummyCircuit<C::CircuitField>,
    ) -> DummyCircuit<C::CircuitField> {
        let data: InputData =
            <FileReader as IOReader<DummyCircuit<_>, C>>::read_data_from_json::<InputData>(file_path);

        // Assign inputs to assignment
       assignment.input_a = C::CircuitField::from(data.input_a);
       assignment.input_b = C::CircuitField::from(data.input_b);

        // Return the assignment
        assignment
    }
    fn read_outputs(
        &mut self,
        file_path: &str,
        mut assignment: DummyCircuit<C::CircuitField>,
    ) -> DummyCircuit<C::CircuitField> {
        let data: OutputData =
            <FileReader as IOReader<DummyCircuit<_>, C>>::read_data_from_json::<OutputData>(file_path);

        // Assign inputs to assignment
        assignment.output = C::CircuitField::from(data.output);
        
        assignment
    }
    fn get_path(&self) -> &str {
        &self.path
    }
}
fn handle_args(mut file_reader: FileReader) {

    let matches: clap::ArgMatches = Command::new("File Copier")
        .version("1.0")
        .about("Copies content from input file to output file")
        .arg(
            Arg::new("type")
                .help("The type of main runner we want to run")
                .required(true) // This argument is required
                .index(1), // Positional argument (first argument)
        )
        .arg(
            Arg::new("input")
                .help("The file to read circuit inputs from")
                .required(false) // This argument is required
                .long("input") // Use a long flag (e.g., --name)
                .short('i')  // Use a short flag (e.g., -n)
                // .index(2), // Positional argument (first argument)
        )
        .arg(
            Arg::new("output")
                .help("The outputs to the circuit")
                .required(false) // This argument is also required
                .long("output") // Use a long flag (e.g., --name)
                .short('o')  // Use a short flag (e.g., -n)
                // .index(3), // Positional argument (second argument)
        )
        .arg(
            Arg::new("name")
                .help("The name of the circuit for the file names to serialize/deserialize")
                .required(false) // This argument is also required
                .long("name") // Use a long flag (e.g., --name)
                .short('n')  // Use a short flag (e.g., -n)
        )
        .get_matches();

        // let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
        // let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"

    // The first argument is the command we need to identify
    // let command = &args[1];
    let command = matches.get_one::<String>("type").unwrap();
    

    match command.as_str() {
        "run_proof" => {
            let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
            run_main::<BN254Config, _,  Circuit<Variable>,
            DummyCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
                                    >(&mut file_reader, &input_path, &output_path);
        }
                                    
        "run_compile_circuit" => {
            // let circuit_name = &args[2];
            
            run_compile_and_serialize::<BN254Config,Circuit<Variable>>(&file_reader.path);
            // compile_circ(circuit_name, demo);
        }
        "run_gen_witness" => {
            let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
            let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
            run_witness_and_proof::<BN254Config, _, DummyCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>>(&mut file_reader, input_path, output_path, circuit_name);
        }
        // "run_prove_witness" => {
        //     let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
        //     let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
        //     let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"
        //     run_prove_witness(circuit_name, witness_name, proof_name, input_file, output_file, demo);
        // }
        "run_gen_verify"=> {

            // let input_path = matches.get_one::<String>("input").unwrap(); // "inputs/reward_input.json"
            // let output_path = matches.get_one::<String>("output").unwrap(); //"outputs/reward_output.json"
            let circuit_name = matches.get_one::<String>("name").unwrap(); //"outputs/reward_output.json"

            
            run_verify_no_circuit::<BN254Config, FileReader, DummyCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>>(&file_reader.path);
        }
        _ => {
            panic!("Unknown command or missing arguments.");
            // exit(1);
        }
    }
}

fn main() {
    let file_reader = FileReader {
        path: "simple_circuit".to_owned(),
    };
    
    // main_runner::run_bn254::<Circuit<Variable>,
    // Circuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
    //                         _>(&mut file_reader);

    // main_runner::run_bn254_seperate::<Circuit<Variable>,
    //                         DummyCircuit<<expander_compiler::frontend::BN254Config as expander_compiler::frontend::Config>::CircuitField>,
    //                                                 _>(&mut file_reader);
    handle_args(file_reader);
}
