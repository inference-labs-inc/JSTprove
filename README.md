# JSTProve

[Zero-knowledge proofs](https://docs.inferencelabs.com/resources/glossary#zero-knowledge-proof) of ML [inference](https://docs.inferencelabs.com/resources/glossary#inference) on ONNX models — powered by [Polyhedra Network's **Expander**](https://github.com/PolyhedraZK/Expander) and [**Expander Compiler Collection (ECC)**](https://github.com/PolyhedraZK/ExpanderCompilerCollection).

- 🎯 **You bring ONNX** → we compile to a circuit, generate a witness, prove, and verify — via a simple CLI.
- ✅ Supported ops: **Conv2D**, **GEMM/MatMul**, **ReLU**, **MaxPool2D**.
- 🧰 Docs: see [docs/](docs/)  
  - [Overview](docs/overview.md) · [Quickstart](docs/quickstart.md) · [CLI Reference](docs/cli.md)  
  - [Models](docs/models.md) · [Artifacts](docs/artifacts.md) · [Troubleshooting](docs/troubleshooting.md) · [FAQ](docs/faq.md)

## Quickstart

```bash
# 1) Compile
python -m python.frontend.cli compile \
  -m python/models/models_onnx/doom.onnx \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx

# 2) Witness
python -m python.frontend.cli witness \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin

# 3) Prove
python -m python.frontend.cli prove \
  -c artifacts/doom/circuit.txt \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin

# 4) Verify
python -m python.frontend.cli verify \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx \
  -i python_testing/models/inputs/doom_input.json \
  -o artifacts/doom/output.json \
  -w artifacts/doom/witness.bin \
  -p artifacts/doom/proof.bin