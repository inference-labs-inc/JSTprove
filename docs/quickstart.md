# Quickstart

## Prerequisites

See **[Installation](install.md)** first.

- **Python 3.12**
- **Rust toolchain** (stable)
- Run commands from the **repo root** (so the runner binary path resolves).
- The CLI's **compile** step will (re)build the Rust runner automatically when needed.

---

## Demo paths

- ONNX model: `python/models/models_onnx/doom.onnx`
- Example input JSON: `python_testing/models/inputs/doom_input.json`
- Artifacts dir: `artifacts/doom/*`

---

## 1) Compile

Generates a circuit and a **quantized ONNX**.

```bash
python -m python.frontend.cli compile \
  -m python/models/models_onnx/doom.onnx \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx
```

---

## 2) Witness

Reshapes/scales inputs, runs the quantized model, and writes witness + outputs.

```bash
python -m python.frontend.cli witness \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin
```

---

## 3) Prove

```bash
python -m python.frontend.cli prove \
  -c artifacts/doom/circuit.txt \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin
```

---

## 4) Verify

```bash
python -m python.frontend.cli verify \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin
```

If it prints **Verified**, you’re done 🎉
