`# Testing Environment

## Setup python environment

```
pip install -r requirements.txt
```

## Building Structure

The development/building process will involve working in two different areas of the codebase. We will begin with the python testing files. 

In `python/testing/core` directory, we will build the python representation of the code. With this, we can test the function we are trying to circuitize line by line for easier development. Additionally, through this code, we will write the inputs and outputs (and weights if applicable) of our function/circuit to file, so that the rust circuit can read this in. Next we will call our rust code to compile the circuit, run the witness and prove and verify the given inputs and outputs. For this process, use `python/testing/core/testing_circuits_base_functions.py` as an example/template.

In `jstprove_circuits` directory, we will build the Expander circuits in rust for proving. We can follow the template in `jstprove_circuits/bin/testing.rs`. The templates have been designed to make the development process as easy as possible. The helper functions are in `jstprove_circuits/src/`. For creating circuits, we must define the inputs and outpsuts structure of the circuits. We must then specify how these get read into the circuit. Finally, we must design the circuit

## Python Testing

To run testing environment in python:

```
python -m python/testing/core.testing_circuit
```

## Circuit creation Instructions

The circuit creation process involves working with two (or three) files. 

1. Create the circuit file using `gravy_circuits/bin/testing.rs` as a template. The binary should belong in `gravy_circuits/bin/`. Add the name of the circuit and path to the binary in `gravy_circuits/Cargo.toml`

2. Work in `python/testing/core/testing_circuit`

    a. Change the self.name in the `__init__` to the name of the circuit [circuit_name], used in the file creation code specified in section 1.

    b. Generate inputs to the function in the `___init__` 

    c. Code the function to circuitize in `python/testing/core/testing_circuit`, 
  
    d. Define the inputs and outputs, to be sent to json for circuits

3. Create rust file in `jstprove_circuits/bin` and add the relevant binary to the Cargo.toml in `jstprove_circuits`

    a. Define inputs and outputs, to read into the circuit

    b. Define circuit

4. Run ``` python -m python/testing/core.circuit_tests``` to test the accuracy of the circuit

TODO: Incorporate proof time, proof size and max memory used results, into the circuit testing 

### Circuit Example

To run circuit example:
```
python -m python/testing/core.matrix_multiplication
```

Relevent files to explore -> `python/testing/core/matrix_multiplication.py` and `jstprove_circuits/bin/matrix_multiplication.rs`


## Important note

We must run the cargo files with release for reasonable time process

## Common debugging issues

1. Outputs have not been quantized back down

2. Inputs must be converted to int, before process in python

3. Quantized parameter set to false when it should be true

4. Make sure weights being loaded into rust file are from correct circuit (the naming is correct)

# Command-Line Interface

The JSTProve CLI runs four steps: **compile → witness → prove → verify**. It's intentionally barebones: no circuit class flags, no path inference. You must pass correct paths.

## Prereqs

* **Python 3.12** (with project deps installed).
 - Run commands **from the repo root** so `./target/release/onnx_generic_circuit` is found.
 - You do **not** have to build the Rust runner manually — the **compile** step
   (with `dev_mode=True`) will (re)build it as needed.

Tip: add `--no-banner` to hide the ASCII header.

## Help

```bash
python -m python.frontend.cli --help
python -m python.frontend.cli <subcommand> --help
# e.g.
python -m python.frontend.cli witness --help
```

## Commands (with Doom model example)

Paths used below:

* ONNX model: `python/models/models_onnx/doom.onnx`
* Example input JSON: `python_testing/models/inputs/doom_input.json`
* Artifacts: `artifacts/doom/*`

### 1) Compile

Generates a circuit file and a **quantized ONNX** model.

```bash
python -m python.frontend.cli compile \
  -m python/models/models_onnx/doom.onnx \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx
```

**Flags**

* `-m/--model-path` (required): original ONNX model
* `-c/--circuit-path` (required): output circuit path
* `-q/--quantized-path` (required): output quantized ONNX path

---

### 2) Witness

Reshapes/scales inputs, runs the (quantized) model to produce outputs, and writes the witness.

```bash
python -m python.frontend.cli witness \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin
```

**Flags**

* `-c/--circuit-path` (required): compiled circuit
* `-q/--quantized-path` (required): quantized ONNX
* `-i/--input-path` (required): input JSON
* `-o/--output-path` (required): output JSON (written)
* `-w/--witness-path` (required): witness file (written)

---

### 3) Prove

Creates a proof from the circuit + witness.

```bash
python -m python.frontend.cli prove \
  -c artifacts/doom/circuit.txt \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin
```

**Flags**

* `-c/--circuit-path` (required): compiled circuit
* `-w/--witness-path` (required): witness file
* `-p/--proof-path` (required): proof file (written)

---

### 4) Verify

Verifies the proof (requires the quantized model to hydrate input shapes).

```bash
python -m python.frontend.cli verify \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin
```

**Flags**

* `-c/--circuit-path` (required): compiled circuit
* `-q/--quantized-path` (required): quantized ONNX
* `-i/--input-path` (required): input JSON
* `-o/--output-path` (required): expected outputs JSON
* `-w/--witness-path` (required): witness file
* `-p/--proof-path` (required): proof file

---

## Notes & gotchas

* The default circuit is **GenericModelONNX**; you don’t pass a circuit class or name.
* All paths are **mandatory**; no automatic discovery or inference.
* Use a **`.onnx`** file for `--quantized-path`. If you see `ONNXRuntimeError … Protobuf parsing failed`, you likely pointed to a `.json` by mistake.
* If the runner isn't found, make sure you're launching from the repo root.
* The compile step will auto-build the runner if needed.
