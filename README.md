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

This CLI tool is designed to run various circuit operations, including circuit compilation, witness generation, proof generation, and verification. The CLI dynamically loads circuit modules and resolves file paths using fuzzy matching, making it easier to work with different circuit configurations and JSON file inputs/outputs.

## Features

- **Dynamic Module Loading:**  
  Load any circuit module and class (default is `SimpleCircuit`) from the `python_testing/circuit_models` directory.

- **File Resolution:**  
  Automatically searches for JSON files (with fuzzy matching) for input and output. You can override the default paths using command-line flags.

- **Operation Flags:**  
  Run specific stages of your circuit workflow:
  - `--compile_circuit`: Compile the circuit.
  - `--gen_witness`: Generate the witness.
  - `--prove_witness`: Generate both witness and proof.
  - `--gen_verify`: Run verification.
  - `--base_testing`: Run base testing (for quick testing).
  - `--end_to_end`: Run end-to-end testing.
  - `--all`: Run all stages in the following order: **compile_circuit**, **gen_witness**, **prove_witness**, then **gen_verify**.

- **List Available Circuits:**  
  Use the `--list_circuits` flag to search for available circuit files. By default, the search is limited to:
  - `python_testing/circuit_components`
  - `python_testing/circuit_models`  
  You can override this using the `--circuit_search_path` flag (path is relative to the project root).

## Usage

### Running Specific Operations
To run a specific operation, provide the corresponding flag along with the `--circuit` argument to specify which circuit module to load. For example:

```bash
python -m cli --circuit simple_circuit --compile_circuit
