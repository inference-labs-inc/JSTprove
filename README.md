# Testing Environment

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
   python cli.py --circuit simple_circuit --gen_witness --witness witness.txt --input input.json --output output.json --circuit_path basic_circuit.txt
   python cli.py --circuit simple_circuit --prove --witness witness.txt --proof proof.bin --circuit_path basic_circuit.txt
   python cli.py --circuit simple_circuit --verify --witness witness.txt --proof proof.bin --input input.json --output output.json --circuit_path basic_circuit.txt

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

   