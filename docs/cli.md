# CLI Reference

The JSTProve CLI runs four steps: **compile → witness → prove → verify**. It’s intentionally barebones: no circuit class flags, no path inference. You must pass correct paths.

---

## Synopsis

```bash
python -m python.frontend.cli [--no-banner] <command> [options]
````

* `--no-banner` — suppress the ASCII header.
* Abbreviations are **disabled**; use the full subcommand or an alias.

---

## Help

```bash
python -m python.frontend.cli --help
python -m python.frontend.cli <subcommand> --help
# e.g.
python -m python.frontend.cli witness --help
```

---

## Example paths used below

* ONNX model: `python/models/models_onnx/doom.onnx`
* Example input JSON: `python_testing/models/inputs/doom_input.json`
* Artifacts: `artifacts/doom/*`

---

## Commands

### compile (alias: `comp`)

Generate a circuit file and a **quantized ONNX** model.

**Options**

* `-m, --model-path <path>` (required) — original ONNX model
* `-c, --circuit-path <path>` (required) — output circuit path
* `-q, --quantized-path <path>` (required) — output quantized ONNX path

**Example**

```bash
python -m python.frontend.cli compile \
  -m python/models/models_onnx/doom.onnx \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx
```

---

### witness (alias: `wit`)

Reshapes/scales inputs, runs the quantized model to produce outputs, and writes the witness.

**Options**

* `-c, --circuit-path <path>` (required) — compiled circuit
* `-q, --quantized-path <path>` (required) — quantized ONNX
* `-i, --input-path <path>` (required) — input JSON
* `-o, --output-path <path>` (required) — output JSON (written)
* `-w, --witness-path <path>` (required) — witness file (written)

**Example**

```bash
python -m python.frontend.cli witness \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin
```

---

### prove (alias: `prov`)

Create a proof from the circuit + witness.

**Options**

* `-c, --circuit-path <path>` (required) — compiled circuit
* `-w, --witness-path <path>` (required) — witness file
* `-p, --proof-path <path>` (required) — proof file (written)

**Example**

```bash
python -m python.frontend.cli prove \
  -c artifacts/doom/circuit.txt \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin
```

---

### verify (alias: `ver`)

Verify the proof (requires the quantized model to hydrate input shapes).

**Options**

* `-c, --circuit-path <path>` (required) — compiled circuit
* `-q, --quantized-path <path>` (required) — quantized ONNX
* `-i, --input-path <path>` (required) — input JSON
* `-o, --output-path <path>` (required) — expected outputs JSON
* `-w, --witness-path <path>` (required) — witness file
* `-p, --proof-path <path>` (required) — proof file

**Example**

```bash
python -m python.frontend.cli verify \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin
```

---

## Short flags

* `-m, -c, -q, -i, -o, -w, -p`

## Command aliases

* `compile` → `comp`
* `witness` → `wit`
* `prove` → `prov`
* `verify` → `ver`

---

## Notes & gotchas

* The default circuit is **GenericModelONNX**; you don’t pass a circuit class or name.
* All paths are **mandatory**; no automatic discovery or inference.
* Use a **`.onnx`** file for `--quantized-path`. If you see `ONNXRuntimeError … Protobuf parsing failed`, you likely pointed to a `.json` by mistake.
* If the runner isn’t found, make sure you’re launching from the **repo root**.
* The **compile** step will auto-build the runner if needed.
