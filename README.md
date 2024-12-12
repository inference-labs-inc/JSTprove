# Testing Environment

## Python Testing

To create a file in ECC to build circuits with, run:

```
python -m python_testing.file_creator [circuit_name]
```

This will create a rust file in `ExpanderCompilerCollection/expander_compiler/bin/[circuit_name].rs` and adjust the relevant cargo.toml file, in order to make this code callable

To run testing environment in python:

```
python -m python_testing.testing_circuit
```

## Circuit creation Instructions

The circuit creation process involves working with two (or three) files. 

1. Create the circuit file using 
```
python -m python_testing.file_creator [circuit_name] 
```

2. Work in `python_testing/testing_circuit`

    a. Change the self.name in the `__init__` to the name of the circuit [circuit_name], used in the file creation code specified in section 1.

    b. Generate inputs to the function in the `___init__` 

    c. Code the function to circuitize in `python_testing/testing_circuit`, 
  
    d. Define the inputs and outputs, to be sent to json for circuits

3. Go to newly generated circuit file in `ExpanderCompilerCollection/expander_compiler/bin`

    a. Define inputs and outputs, to read into the circuit

    b. Define circuit

4. Run ``` python -m python_testing.testing_circuit``` to test the accuracy of the circuit

TODO: Incorporate proof time, proof size and max memory used results, into the circuit testing 

### Circuit Example

To run circuit example:
```
python -m python_testing.testing_circuit_sn2_example
```

Relevent files to explore -> `python_testing/testing_circuit_sn2_example.py` and `ExpanderCompilerCollection/expander_compiler/bin/reward.rs`

## Expander Compiler Collection

Run the proof generation with:

```
cargo run --bin testing --manifest-path ExpanderCompilerCollection/Cargo.toml inputs/reward_input.json output/reward_output.json
```
