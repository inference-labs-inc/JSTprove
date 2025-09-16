# Artifacts

This page describes the files JSTprove reads/writes during the pipeline.

---

## Files you'll typically see

- **Circuit** — `circuit.txt`
  Compiled Expander circuit description.
  _Produced by:_ `compile`

- **Quantized model** — `quantized.onnx`
  ONNX model with integerized ops (used by witness/verify to hydrate shapes).
  _Produced by:_ `compile`

- **Inputs** — your input JSON (you provide it)
  During witness/verify the CLI also creates a local `*_reshaped.json` (next to your CWD) after scaling/reshaping.
  _Consumed by:_ `witness`, `verify`

- **Outputs** — `output.json`
  Model outputs (integer domain) computed from the quantized model.
  _Produced by:_ `witness` (and used by `verify`)

- **Witness** — `witness.bin`
  Private inputs / auxiliary data for proving.
  _Produced by:_ `witness` (consumed by `prove`, `verify`)

- **Proof** — `proof.bin`
  Zero-knowledge proof blob.
  _Produced by:_ `prove` (checked by `verify`)

---

## Typical layout

You control all paths; the CLI **does not** infer directories.

```

artifacts/
lenet/
circuit.txt
quantized.onnx
output.json
witness.bin
proof.bin
models/
inputs/
lenet_input.json

```

> Note: `*_reshaped.json` is generated in your current working directory during witness/verify. It’s a convenience file reflecting the scaled/reshaped inputs actually fed into the circuit.

---

## Tips

- Keep artifacts from the **same compile** together (circuit + quantized ONNX) to avoid shape/version mismatches.
- If you change the ONNX model, **re-run compile** before witness/prove/verify.
- Store inputs/outputs under versioned folders if you need reproducibility.
