```
  888888  .d8888b. 88888888888
    "88b d88P  Y88b    888
     888 Y88b.         888
     888  "Y888b.      888  88888b.  888d888 .d88b.  888  888  .d88b.
     888     "Y88b.    888  888 "88b 888P"  d88""88b 888  888 d8P  Y8b
     888       "888    888  888  888 888    888  888 Y88  88P 88888888
     88P Y88b  d88P    888  888 d88P 888    Y88..88P  Y8bd8P  Y8b.
     888  "Y8888P"     888  88888P"  888     "Y88P"    Y88P    "Y8888
   .d88P                    888
 .d88P"                     888
888P"                       888
```
---

# JSTprove

Zero-knowledge proofs of ML inference on **ONNX** models — powered by [Polyhedra Network’s **Expander**](https://github.com/PolyhedraZK/Expander) (GKR/sum-check prover) and [**Expander Compiler Collection (ECC)**](https://github.com/PolyhedraZK/ExpanderCompilerCollection).

* 🎯 **You bring ONNX** → we quantize, compile to a circuit, generate a witness, prove, and verify — via a simple CLI.
* ✅ Supported ops (current): **Conv2D**, **GEMM/MatMul (FC)**, **ReLU**, **MaxPool2D**.
* 🧰 CLI details: see **[docs/cli.md](docs/cli.md)**

👉 Just want to see it in action? Jump to [Quickstart (LeNet demo)](#quickstart-lenet-demo).

---

## Table of Contents
<details>
<summary>Click to expand</summary>

- [What is JSTprove?](#what-is-jstprove)
  - [High-level architecture](#high-level-architecture)
  - [Design principles](#design-principles)
- [Installation](#installation)
  - [0) Requirements](#0-requirements)
  - [1) System packages](#1-system-packages)
  - [2) Rust toolchain](#2-rust-toolchain)
  - [3) Clone JSTprove & set up Python](#3-clone-jstprove--set-up-python)
  - [4) Install & verify Expander (before building JSTprove)](#4-install--verify-expander-before-building-jstprove)
  - [5) Build the JSTprove runner (optional; the CLI can build on demand)](#5-build-the-jstprove-runner-optional-the-cli-can-build-on-demand)
  - [6) Try the CLI](#6-try-the-cli)
- [Quickstart (LeNet demo)](#quickstart-lenet-demo)
- [CLI reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Legal](#legal)
- [Acknowledgments](#acknowledgments)

</details>

## What is JSTprove?

**JSTprove** is a [zkML](https://docs.inferencelabs.com/zk-ml) toolkit/CLI that produces [**zero-knowledge proofs**](https://docs.inferencelabs.com/resources/glossary#zero-knowledge-proof) **of AI** [**inference**](https://docs.inferencelabs.com/resources/glossary#inference).
You provide an **ONNX** model and inputs; JSTprove handles **quantization**, **circuit generation** (via ECC), **witness creation**, **proving** (via Expander), and **verification** — with explicit, user-controlled file paths.

### High-level architecture

* **Python pipeline:** Converts **ONNX → quantized ONNX**, prepares I/O, drives the Rust runner, exposes the **CLI**.
* **Rust crate:** `rust/jstprove_circuits` implements layer circuits (Conv2D, ReLU, MaxPool2D, GEMM/FC) and a runner.
* **Circuit frontend:** [ECC](https://github.com/PolyhedraZK/ExpanderCompilerCollection) Rust API for arithmetic circuits.
* **Prover backend:** [Expander](https://github.com/PolyhedraZK/Expander) (GKR/sum-check prover/verification).

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

### Installing from PyPI (Recommended)

#### Prerequisites
- **UV**: Fast Python package manager ([install UV](https://docs.astral.sh/uv/getting-started/installation/))
- **OpenMPI**: Required system dependency

**macOS:**
```bash
brew install open-mpi
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

**Other Linux:**
- RHEL/CentOS/Fedora: `sudo dnf install openmpi openmpi-devel`
- Arch: `sudo pacman -S openmpi`

#### Install JSTprove
```bash
uv tool install JSTprove
```

#### Verify installation
```bash
jst --help
```

> Note: The package includes all necessary binaries (onnx_generic_circuit and expander-exec) for the full workflow.

### Installing from GitHub Release

Download the appropriate wheel for your platform from the [latest release](https://github.com/Inference-Labs-Inc/jstprove/releases/latest):
- Linux: `JSTprove-*-manylinux_*.whl`
- macOS (Apple Silicon): `JSTprove-*-macosx_11_0_arm64.whl`

Then install:
```bash
uv tool install /path/to/JSTprove-*.whl
```

---

## Development Installation

<details>
<summary>Click to expand for development setup instructions</summary>

### 0) Requirements

- **Python**: 3.10–3.12 (⚠️ Not compatible with Python 3.13)
- **UV**: Fast Python package manager ([install UV](https://docs.astral.sh/uv/getting-started/installation/))

> Note: UV will automatically install and manage the correct Python version for you.

> **Heads-up:** If you just installed `uv` and the command isn't found, **close and reopen your terminal** (or re-source your shell init file) so the `uv` shim is picked up on`PATH`.

### 1) System packages

> Run commands from the **repo root** so the runner binary path (e.g., `./target/release/onnx_generic_circuit`) resolves.

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

### 2) Rust toolchain

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

### 3) Clone JSTprove & set up Python

```bash
git clone https://github.com/inference-labs-inc/JSTprove.git
cd JSTprove

# Install dependencies with UV (automatically creates and manages virtual environment)
uv sync
```

> If `uv` was just installed, you may need to **restart your terminal** before running `uv sync`.

---

### 4) Install & verify **Expander** (before building JSTprove)

JSTprove relies on Polyhedra Network’s **Expander** (prover) and **Expander Compiler Collection (ECC)** crates.
For a clean environment, install Expander and run its self-checks first.
To keep paths simple (and to match our scripts), **clone Expander as a subfolder of this repo**:

```bash
# From the JSTprove repo root
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

> **Why inside the repo?** Our example commands and helper scripts assume `./Expander` as the manifest path. Keeping Expander elsewhere can lead to `manifest path 'Expander/Cargo.toml' does not exist` errors unless you always pass absolute paths.

*(You do **not** need to clone ECC separately unless you plan to override Cargo git sources; Cargo will fetch ECC automatically when building JSTprove.)*

---

### 5) Build the JSTprove runner (optional; the CLI can build on demand)

```bash
# Make sure you're back in the JSTprove repo root (not in Expander).
# If you just followed Step 3, run:
cd ../JSTprove

# Then build:
cargo build --release
```

> The CLI `compile` step will **(re)build** the runner automatically when needed, so this step is just a sanity check.

---

### 6) Install and try the CLI

**Option A: Install in virtual environment (for development)**
```bash
# Install as editable package in venv
uv pip install -e .

# Try the CLI (with venv activated)
jst --help
```

**Option B: Install globally (for regular use)**
```bash
# Install as global tool
uv tool install .

# Try the CLI (available globally)
jst --help
```

> ⏳ Note: The first time you run this command it may take a little while due to Python/Rust imports and initialization. This is normal—subsequent runs will be faster.

You can now follow the **Quickstart** commands (compile → witness → prove → verify).

</details>

---

## Quickstart (LeNet demo)

Demo paths:

* ONNX: `python/models/models_onnx/lenet.onnx`
* Input JSON: `python/models/inputs/lenet_input.json`
* Artifacts: `artifacts/lenet/*`

> ⏳ Note: The commands below may take a little longer _the first time_ they are run, as dependencies and binaries are initialized. After that, runtime reflects the actual computation (e.g., compiling circuits, generating witnesses, or proving), which can still be intensive depending on the model.

1. **Compile** → circuit + **quantized ONNX**

```bash
jst compile \
  -m python/models/models_onnx/lenet.onnx \
  -c artifacts/lenet/circuit.txt
```

2. **Witness** → reshape/scale inputs, run model, write witness + outputs

```bash
jst witness \
  -c artifacts/lenet/circuit.txt \
  -i python/models/inputs/lenet_input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.bin
```

3. **Prove** → witness → proof

```bash
jst prove \
  -c artifacts/lenet/circuit.txt \
  -w artifacts/lenet/witness.bin \
  -p artifacts/lenet/proof.bin
```

4. **Verify** → check the proof (needs quantized ONNX for input shapes)

```bash
jst verify \
  -c artifacts/lenet/circuit.txt \
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
> * **License:** See `LICENSE` (TBD). Third-party components (e.g., Expander, ECC) are licensed under their respective terms.
> * **No warranty:** Provided “as is” without warranties or conditions of any kind. Use at your own risk.
> * **Security & export:** Cryptography may be subject to local laws. Conduct your own security review before production use.
> * **Trademarks:** All product names, logos, and brands are property of their respective owners.

---

## Acknowledgments

We gratefully acknowledge [**Polyhedra Network**](https://polyhedra.network/) for:

* [**Expander**](https://github.com/PolyhedraZK/Expander) — the GKR/sumcheck proving system we build on.

* [**Expander Compiler Collection (ECC)**]() — the circuit frontend used to construct arithmetic circuits for ML layers.
