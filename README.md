```
         _/    _/_/_/  _/_/_/_/_/  _/_/_/
        _/  _/            _/      _/    _/  _/  _/_/    _/_/    _/      _/    _/_/
       _/    _/_/        _/      _/_/_/    _/_/      _/    _/  _/      _/  _/_/_/_/
_/    _/        _/      _/      _/        _/        _/    _/    _/  _/    _/
 _/_/    _/_/_/        _/      _/        _/          _/_/        _/        _/_/_/
```

---

# JSTProve

Zero-knowledge proofs of ML inference on **ONNX** models — powered by [Polyhedra Network’s **Expander**](https://github.com/PolyhedraZK/Expander) (GKR/sum-check prover) and [**Expander Compiler Collection (ECC)**](https://github.com/PolyhedraZK/ExpanderCompilerCollection).

- 🎯 **You bring ONNX** → we quantize, compile to a circuit, generate a witness, prove, and verify — via a simple CLI.
- ✅ Supported ops (current): **Conv2D**, **GEMM/MatMul (FC)**, **ReLU**, **MaxPool2D**.
- 🧰 CLI details: see **[docs/cli.md](docs/cli.md)**

---

## What is JSTProve?

**JSTProve** is a [zkML](https://docs.inferencelabs.com/zk-ml) toolkit/CLI that produces [**zero-knowledge proofs**](https://docs.inferencelabs.com/resources/glossary#zero-knowledge-proof) **of AI** [**inference**](https://docs.inferencelabs.com/resources/glossary#inference).
You provide an **ONNX** model and inputs; JSTProve handles **quantization**, **circuit generation** (via ECC), **witness creation**, **proving** (via Expander), and **verification** — with explicit, user-controlled file paths.

### High-level architecture

- **Python pipeline:** Converts **ONNX → quantized ONNX**, prepares I/O, drives the Rust runner, exposes the **CLI**.
- **Rust crate:** `rust/jstprove_circuits` implements layer circuits (Conv2D, ReLU, MaxPool2D, GEMM/FC) and a runner.
- **Circuit frontend:** [ECC](https://github.com/PolyhedraZK/ExpanderCompilerCollection) Rust API for arithmetic circuits.
- **Prover backend:** [Expander](https://github.com/PolyhedraZK/Expander) (GKR/sum-check prover/verification).

```
ONNX model ─► Quantizer (Py) ─► Circuit via ECC (Rust) ─► Witness (Rust) ─► Proof (Rust) ─► Verify (Rust)
```

### Design principles

- **User-friendly frontend to Expander:** A thin, practical, circuit-based layer that makes Expander/ECC easy to use from a simple CLI — no circuit classes, no path inference, predictable artifacts.
- **Explicit & reproducible:** You pass exact paths; we emit concrete artifacts (circuit, quantized ONNX, witness, proof). No hidden discovery or heuristics.
- **Clear separation:** Python orchestrates the pipeline and I/O; Rust implements the circuits and invokes Expander/ECC.
- **Quantization that's simple & faithful:** We scale tensors, **round to integers**, run the model, and (where needed) **rescale** outputs back. Scaling keeps arithmetic cheap while remaining close to the original FP behavior.
- **Small, fast circuits when possible:** Where safe, we fuse common patterns (e.g., **Linear + ReLU**, **Conv + ReLU**) into streamlined circuit fragments to reduce constraints.
- **Deterministic debugging:** We prefer loud failures and inspectable intermediates (e.g., `*_reshaped.json`) over implicit magic.

---

## Installation

> Run commands from the **repo root** so the runner binary path (e.g., `./target/release/onnx_generic_circuit`) resolves.

### 0) System packages

#### Ubuntu/Debian

```bash
sudo apt-get update && sudo apt-get install -y \
  libopenmpi-dev openmpi-bin pkg-config libclang-dev clang
```

#### macOS

```bash
brew install open-mpi llvm
```

---

### 1) Rust toolchain

Install Rust via rustup (if you don't have it):

```bash
# macOS/Linux:
curl https://sh.rustup.rs -sSf | sh
# then restart your shell
```

Verify your install:

```bash
rustup --version
rustc --version
cargo --version
```

> This repo includes a `rust-toolchain.toml` that pins the required **nightly**.
> When you run `cargo` in this directory, rustup will automatically download/use
> the correct toolchain. You **do not** need to run `rustup override set nightly`.

(Optional) If you want to prefetch nightly ahead of time:

```bash
rustup toolchain install nightly
```

---

### 2) Install the app

#### Install from PyPI

```bash
pip install jstprove
```

#### Or install from source

Clone the repository:

```bash
git clone https://github.com/inference-labs-inc/JSTProve.git
cd JSTProve
```

Use `venv` to create a virtual environment (recommended):

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

### 3) Install & verify **Expander** (before building JSTProve)

JSTProve relies on Polyhedra Network’s **Expander** (prover) and **Expander Compiler Collection (ECC)** crates.
For a clean environment, install Expander and run its self-checks first.

```bash
# In a sibling folder (or anywhere you keep deps)
git clone https://github.com/PolyhedraZK/Expander.git
cd Expander

# Build (uses the toolchain you configured with rustup)
cargo build --release
```

**Verify Expander:** follow the “Correctness Test” (or equivalent) in the Expander README.
If you’re unsure, a quick smoke test is often:

```bash
cargo test --release
```

> Refer to the Expander README for the authoritative verification command(s), which may change over time.

_(You do **not** need to clone ECC separately unless you plan to override Cargo git sources; Cargo will fetch ECC automatically when building JSTProve.)_

---

### 4) Build the JSTProve runner (optional; the CLI can build on demand)

```bash
# From JSTProve repo root
cargo build --release
./target/release/onnx_generic_circuit --help
```

> The CLI `compile` step will **(re)build** the runner automatically when needed, so this step is just a sanity check.

---

### 5) Try the CLI

```bash
jstprove --help
```

You can now follow the **Quickstart** commands (compile → witness → prove → verify).

---

## Quickstart (LeNet demo)

Demo paths:

- ONNX: `python/models/models_onnx/lenet.onnx`
- Input JSON: `python/models/inputs/lenet_input.json`
- Artifacts: `artifacts/lenet/*`

1. **Compile** → circuit + **quantized ONNX**

```bash
jstprove compile \
  -m python/models/models_onnx/lenet.onnx \
  -c artifacts/lenet/circuit.txt \
  -q artifacts/lenet/quantized.onnx
```

2. **Witness** → reshape/scale inputs, run model, write witness + outputs

```bash
jstprove witness \
  -c artifacts/lenet/circuit.txt \
  -q artifacts/lenet/quantized.onnx \
  -i python/models/inputs/lenet_input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.bin
```

3. **Prove** → witness → proof

```bash
jstprove prove \
  -c artifacts/lenet/circuit.txt \
  -w artifacts/lenet/witness.bin \
  -p artifacts/lenet/proof.bin
```

4. **Verify** → check the proof (needs quantized ONNX for input shapes)

```bash
jstprove verify \
  -c artifacts/lenet/circuit.txt \
  -q artifacts/lenet/quantized.onnx \
  -i python/models/inputs/lenet_input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.bin \
  -p artifacts/lenet/proof.bin
```

If it prints **Verified**, you're done 🎉

---

## CLI reference

The CLI is intentionally minimal and **doesn't infer paths**.
See **[docs/cli.md](docs/cli.md)** for subcommands, flags, and examples.

---

## Troubleshooting

See **[docs/troubleshooting.md](docs/troubleshooting.md)**

---

## Contributing

See **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** for dev setup, pre-commit hooks, and PR guidelines.

---

## Legal

> **Placeholder (subject to change):**
>
> - **License:** See `LICENSE` (TBD). Third-party components (e.g., Expander, ECC) are licensed under their respective terms.
> - **No warranty:** Provided “as is” without warranties or conditions of any kind. Use at your own risk.
> - **Security & export:** Cryptography may be subject to local laws. Conduct your own security review before production use.
> - **Trademarks:** All product names, logos, and brands are property of their respective owners.

---

## Acknowledgments

We gratefully acknowledge [**Polyhedra Network**](https://polyhedra.network/) for:

- [**Expander**](https://github.com/PolyhedraZK/Expander) — the GKR/sumcheck proving system we build on.

- [**Expander Compiler Collection (ECC)**]() — the circuit frontend used to construct arithmetic circuits for ML layers.
