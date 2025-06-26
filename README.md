`# Testing Environment

## Setup python environment

```
pip install -r requirements.txt
```

## Building Structure

The development/building process will involve working in two different areas of the codebase. We will begin with the python testing files. 

In `python_testing` directory, we will build the python representation of the code. With this, we can test the function we are trying to circuitize line by line for easier development. Additionally, through this code, we will write the inputs and outputs (and weights if applicable) of our function/circuit to file, so that the rust circuit can read this in. Next we will call our rust code to compile the circuit, run the witness and prove and verify the given inputs and outputs. For this process, use `python_testing/testing_circuits_base_functions.py` as an example/template.

In `gravy_circuits` directory, we will build the Expander circuits in rust for proving. We can follow the template in `gravy_circuits/bin/testing.rs`. The templates have been designed to make the development process as easy as possible. The helper functions are in `gravy_circuits/src/`. For creating circuits, we must define the inputs and outpsuts structure of the circuits. We must then specify how these get read into the circuit. Finally, we must design the circuit

## Python Testing

To run testing environment in python:

```
python -m python_testing.testing_circuit
```

## Circuit creation Instructions

The circuit creation process involves working with two (or three) files. 

1. Create the circuit file using `gravy_circuits/bin/testing.rs` as a template. The binary should belong in `gravy_circuits/bin/`. Add the name of the circuit and path to the binary in `gravy_circuits/Cargo.toml`

2. Work in `python_testing/testing_circuit`

    a. Change the self.name in the `__init__` to the name of the circuit [circuit_name], used in the file creation code specified in section 1.

    b. Generate inputs to the function in the `___init__` 

    c. Code the function to circuitize in `python_testing/testing_circuit`, 
  
    d. Define the inputs and outputs, to be sent to json for circuits

3. Create rust file in `gravy_circuits/bin` and add the relevant binary to the Cargo.toml in `gravy_circuits`

    a. Define inputs and outputs, to read into the circuit

    b. Define circuit

4. Run ``` python -m python_testing.circuit_tests``` to test the accuracy of the circuit

TODO: Incorporate proof time, proof size and max memory used results, into the circuit testing 

### Circuit Example

To run circuit example:
```
python -m python_testing.matrix_multiplication
```

Relevent files to explore -> `python_testing/matrix_multiplication.py` and `gravy_circuits/bin/matrix_multiplication.rs`


## Important note

We must run the cargo files with release for reasonable time process

## Common debugging issues

1. Outputs have not been quantized back down

2. Inputs must be converted to int, before process in python

3. Quantized parameter set to false when it should be true

4. Make sure weights being loaded into rust file are from correct circuit (the naming is correct)

# Command Line Interface

This CLI tool runs various circuit operations such as compilation, witness generation, proof generation, and verification. It dynamically loads circuit modules, resolves file paths using fuzzy matching, and can list all available circuit files that inherit from a `Circuit` or `ZKModel` class.

## Features

1. **Dynamic Module Loading**  
   - By default, the CLI looks for circuit modules in:
     - `python_testing.circuit_models`
     - `python_testing.circuit_components`
   - You can also specify a custom search path relative to `python_testing` with the `--circuit_search_path` flag.

2. **File Resolution**  
   - Automatically searches the project root for JSON input and output files using a simple or fuzzy match.
   - You can override the default filenames (e.g., `{circuit}_input.json` and `{circuit}_output.json`) with:
     - `--input` to point to an exact input file
     - `--output` to point to an exact output file
     - `--pattern` to use a custom format (e.g., `my_{circuit}_input.json`).

3. **Operation Flags**  
   - Run specific stages of your circuit workflow:
     - `--compile_circuit`: Compile the circuit.
     - `--gen_witness`: Generate a witness.
     - `--prove_witness`: Generate both the witness and proof.
     - `--gen_verify`: Run verification.   
     - `--end_to_end`: Run an all-in-one test if your circuit supports it.
   - `--all`: Run the main four stages in sequence:  
     1. Compile (`--compile_circuit`)  
     2. Generate Witness (`--gen_witness`)  
     3. Prove Witness (`--prove_witness`)  
     4. Verify (`--gen_verify`)

4. **List Available Circuits**  
   - Use the `--list_circuits` flag to recursively search for Python files containing a class that inherits from `Circuit` or `ZKModel`.  
   - By default, it searches in:
     - `python_testing/circuit_components`
     - `python_testing/circuit_models`
   - Override with `--circuit_search_path <some_relative_folder>` to search elsewhere.

## Basic Usage

1. **Install Dependencies**  
   Ensure you have **Python 3.12.x** installed and all required dependencies listed in `requirements.txt`:

2. **Run the CLI**  
   From the project root (`GravyTesting-Internal`):

   ```bash
   python -m cli --circuit simple_circuit --compile --circuit <circuit_file_path>
   python -m cli --circuit simple_circuit --gen_witness --witness <witness_file_path> --input <input_json_file_path> --output <output_json_file_path> --circuit_path <circuit_file_path>
   python -m cli --circuit simple_circuit --prove --witness <witness_file_path> --proof <proof_file_path> --circuit_path <circuit_file_path>
   python -m cli --circuit simple_circuit --verify --witness <witness_file_path> --proof <proof_file_path> --input <input_json_file_path> --output <output_json_file_path> --circuit_path <circuit_file_path>
   ```

   Dummy demo run through:
   ```bash
   Demo
   run through
   python -m cli --circuit simple_circuit --compile --circuit_path basic_circuit.txt
   python cli.py --circuit simple_circuit --gen_witness --witness witness.txt --input inputs/simple_circuit_input.json --output output/simple_circuit_output.json --circuit_path basic_circuit.txt
   python cli.py --circuit simple_circuit --prove --witness witness.txt --proof proof.bin --circuit_path basic_circuit.txt
   python cli.py --circuit simple_circuit --verify --witness witness.txt --proof proof.bin --input inputs/simple_circuit_input.json --output output/simple_circuit_output.json --circuit_path basic_circuit.txt

   Error about inputs not matching
   python cli.py --circuit simple_circuit --verify --witness witness.txt --proof proof.bin --circuit_path basic_circuit.txt
   Error around verification
   python cli.py --circuit simple_circuit --verify --witness witness.txt --input input.json --output output.json --circuit_path basic_circuit.txt
   run through 
   python cli.py --circuit simple_circuit --verify --witness witness.txt --proof proof.bin --input input.json --output output.json --circuit_path basic_circuit.txt


   python cli.py --circuit demo_cnn --class Demo --compile --circuit_path demo_circuit.txt
   python cli.py --circuit demo_cnn --class Demo --gen_witness --input inputs/demo_cnn_input.json --output output/demo_cnn_output.json  --circuit_path demo_circuit.txt

   ```


   To run Doom and the relevant slices

   python cli.py --circuit doom_model --class Doom --compile --circuit_path doom_circuit.txt
   python cli.py --circuit doom_model --class Doom --gen_witness --input inputs/doom_input.json --output output/doom_output.json  --circuit_path doom_circuit.txt


   python cli.py --circuit doom_slices --class DoomConv1 --compile --circuit_path doom_conv1_circuit.txt
   python cli.py --circuit doom_slices --class DoomConv1 --gen_witness --input inputs/doom_input.json --output output/doom_conv1_output.json  --circuit_path doom_conv1_circuit.txt

   python cli.py --circuit doom_slices --class DoomConv2 --compile --circuit_path doom_conv2_circuit.txt
   python cli.py --circuit doom_slices --class DoomConv2 --gen_witness --input output/doom_conv1_output.json --output output/doom_conv2_output.json  --circuit_path doom_conv2_circuit.txt


   python cli.py --circuit doom_slices --class DoomConv3 --compile --circuit_path doom_conv3_circuit.txt
   python cli.py --circuit doom_slices --class DoomConv3 --gen_witness --input output/doom_conv2_output.json --output output/doom_conv3_output.json  --circuit_path doom_conv3_circuit.txt

   python cli.py --circuit doom_slices --class DoomFC1 --compile --circuit_path doom_fc1_circuit.txt
   python cli.py --circuit doom_slices --class DoomFC1 --gen_witness --input output/doom_conv3_output.json --output output/doom_fc1_output.json  --circuit_path doom_fc1_circuit.txt

   python cli.py --circuit doom_slices --class DoomFC2 --compile --circuit_path doom_fc2_circuit.txt
   python cli.py --circuit doom_slices --class DoomFC2 --gen_witness --input output/doom_fc1_output.json --output output/doom_fc2_output.json  --circuit_path doom_fc2_circuit.txt
   python cli.py --circuit doom_slices --class DoomFC2 --gen_witness --input output/doom_fc1_output.json --output output/doom_output.json  --circuit_path doom_fc2_circuit.txt
   

   python cli.py --circuit maxpooling --class MaxPooling2D --compile --circuit_path maxpool_circuit.txt
   python cli.py --circuit maxpooling --class MaxPooling2D --gen_witness --input inputs/maxpooling_input.json --output output/maxpooling_output.json  --circuit_path maxpool_circuit.txt

   To run Net and the relevant slices

   python cli.py --circuit net_model --class NetModel --compile --circuit_path net_circuit.txt
   python cli.py --circuit net_model --class NetModel --gen_witness --input inputs/net_input.json --output output/net_output.json  --circuit_path net_circuit.txt


   python cli.py --circuit net_model --class NetConv1Model --compile --circuit_path net_conv1_circuit.txt
   python cli.py --circuit net_model --class NetConv1Model --gen_witness --input inputs/net_input.json --output output/net_conv1_output.json  --circuit_path net_conv1_circuit.txt

   python cli.py --circuit net_model --class NetConv2Model --compile --circuit_path net_conv2_circuit.txt
   python cli.py --circuit net_model --class NetConv2Model --gen_witness --input output/net_conv1_output.json --output output/net_conv2_output.json  --circuit_path net_conv2_circuit.txt


   python cli.py --circuit net_model --class NetFC1Model --compile --circuit_path net_fc1_circuit.txt
   python cli.py --circuit net_model --class NetFC1Model --gen_witness --input output/net_conv2_output.json --output output/net_fc1_output.json  --circuit_path net_fc1_circuit.txt

   python cli.py --circuit net_model --class NetFC2Model --compile --circuit_path net_fc2_circuit.txt
   python cli.py --circuit net_model --class NetFC2Model --gen_witness --input output/net_fc1_output.json --output output/net_fc2_output.json  --circuit_path net_fc2_circuit.txt

   python cli.py --circuit net_model --class NetFC3Model --compile --circuit_path net_fc3_circuit.txt
   python cli.py --circuit net_model --class NetFC3Model --gen_witness --input output/net_fc2_output.json --output output/net_fc3_output.json  --circuit_path net_fc3_circuit.txt
   python cli.py --circuit net_model --class NetFC3Model --gen_witness --input output/net_fc2_output.json --output output/net_output.json  --circuit_path net_fc3_circuit.txt
   

   python cli.py --circuit net_model --class NetFC3Model --compile --circuit_path slices/segment_4/segment_4_circuit.compiled 
   python cli.py --circuit net_model --class NetFC3Model --gen_witness --input slices/segment_4/segment_4_input.json --output slices/segment_4/segment_4_output.json  --circuit_path slices/segment_4/segment_4_circuit.compiled --witness slices/segment_4/segment_4_witness.compiled
   python cli.py --circuit net_model --class NetFC3Model --prove --witness slices/segment_4/segment_4_witness.compiled  --circuit_path slices/segment_4/segment_4_circuit.compiled  --proof slices/segment_4/segment_4_proof.bin

   python cli.py --circuit net_model --class NetFC3Model --verify --input slices/segment_4/segment_4_input.json --output slices/segment_4/segment_4_output.json  --circuit_path slices/segment_4/segment_4_circuit.compiled --witness slices/segment_4/segment_4_witness.compiled --proof slices/segment_4/segment_4_proof.bin


   RUSTFLAGS="-C target-cpu=native" cargo run \
  --manifest-path Expander/Cargo.toml \
  --bin expander-exec \
  --release -- prove \

  env RUSTFLAGS="-C target-cpu=native" mpiexec -n 1 cargo run --manifest-path Expander/Cargo.toml --bin expander-exec --release -- -p Raw prove -c slices/segment_4/segment_4_circuit.compiled -w slices/segment_4/segment_4_witness.compiled -o slices/segment_4/segment_4_proof.bin

  env RUSTFLAGS="-C target-cpu=native" mpiexec -n 1 cargo run --manifest-path Expander/Cargo.toml --bin expander-exec --release -- -p Raw verify -c slices/segment_4/segment_4_circuit.compiled -w slices/segment_4/segment_4_witness.compiled -i slices/segment_4/segment_4_proof.bin
  

   <!-- python cli.py --circuit maxpooling --class MaxPooling2D --compile --circuit_path maxpool_circuit.txt
   python cli.py --circuit maxpooling --class MaxPooling2D --gen_witness --input inputs/maxpooling_input.json --output output/maxpooling_output.json  --circuit_path maxpool_circuit.txt -->



   python cli.py --circuit eth_fraud --class Eth --compile --circuit_path eth_fraud_circuit.txt
   python cli.py --circuit eth_fraud --class Eth --gen_witness --input inputs/eth_fraud_input.json --output output/eth_fraud_output.json  --circuit_path eth_fraud_circuit.txt



   If you see this error, it means you need some private variables:
   called `Result::unwrap()` on an `Err` value: UserError("dest ir circuit invalid: circuit has no inputs")


   pipreqs ./ --ignore .venv,.venv_new

   ONNX 1.17 does not support python 3.13
   brew install libffi, open
   