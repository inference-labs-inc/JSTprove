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

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/inference-labs-inc/JSTprove)
[![Telegram](https://img.shields.io/badge/Telegram-Join%20Channel-0088cc?style=flat-square&logo=telegram)](https://t.me/inference_labs)
[![Twitter](https://img.shields.io/badge/Twitter-Follow%20Us-1DA1F2?style=flat-square&logo=twitter)](https://x.com/inference_labs)
[![Website](https://img.shields.io/badge/Website-Visit%20Us-ff7139?style=flat-square&logo=firefox-browser)](https://inferencelabs.com)
[![White paper](https://img.shields.io/badge/Whitepaper-Read-lightgrey?style=flat-square&logo=read-the-docs)](https://doi.org/10.48550/arXiv.2510.21024)

Zero-knowledge proofs of ML inference on **ONNX** models — powered by [Polyhedra Network’s **Expander**](https://github.com/PolyhedraZK/Expander) (GKR/sum-check prover) and [**Expander Compiler Collection (ECC)**](https://github.com/inference-labs-inc/ecc).

Supported ops (current): **Add**, **BatchNormalization**, **Clip**, **Constant**, **Conv**, **Div**, **Flatten**, **Gemm**, **Max**, **MaxPool**, **Min**, **Mul**, **ReLU**, **Reshape**, **Squeeze**, **Sub**, **Unsqueeze**. CLI details: see **[docs/cli.md](docs/cli.md)**.

Just want to see it in action? Jump to [Quickstart (LeNet demo)](#quickstart-lenet-demo).
Curious about how it works under the hood? Check out the [white paper](https://doi.org/10.48550/arXiv.2510.21024).

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
  - [3) Clone JSTprove](#3-clone-jstprove)
  - [4) Build the JSTprove binaries](#4-build-the-jstprove-binaries)
  - [5) Verify the build](#5-verify-the-build)
- [Quickstart (LeNet demo)](#quickstart-lenet-demo)
- [CLI reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [Acknowledgments](#acknowledgments)

</details>

## What is JSTprove?

**JSTprove** is a [zkML](https://docs.inferencelabs.com/zk-ml) toolkit/CLI that produces [**zero-knowledge proofs**](https://docs.inferencelabs.com/resources/glossary#zero-knowledge-proof) **of AI** [**inference**](https://docs.inferencelabs.com/resources/glossary#inference).
You provide an **ONNX** model and inputs; JSTprove handles **quantization**, **circuit generation** (via ECC), **witness creation**, **proving** (via Expander), and **verification** — with explicit, user-controlled file paths.

### High-level architecture

* **Python package:** Thin PyO3 bindings exposing `Circuit`, `WitnessResult`, and `BatchResult` from Rust.
* **Rust workspace:** Four crates (`jstprove_circuits`, `jstprove_io`, `jstprove_onnx`, `jstprove_remainder`) plus `jstprove_pyo3` (excluded from workspace). Two CLI binaries: `jstprove` (Expander backend) and `jstprove-remainder` (Remainder backend).
* **Circuit frontend:** [ECC](https://github.com/inference-labs-inc/ecc) Rust API for arithmetic circuits.
* **Prover backend:** [Expander](https://github.com/PolyhedraZK/Expander) (GKR/sum-check prover/verification).

```text
ONNX model ─► Circuit via ECC (Rust) ─► Witness (Rust) ─► Proof (Rust) ─► Verify (Rust)
```

### Design principles

- **User-friendly frontend to Expander:** A thin, practical, circuit-based layer that makes Expander/ECC easy to use from a simple CLI — no circuit classes, no path inference, predictable artifacts.
- **Explicit & reproducible:** You pass exact paths; we emit concrete artifacts (compiled circuit, witness, proof). No hidden discovery or heuristics.
- **Quantization that's simple & faithful:** We scale tensors, **round to integers**, run the model, and (where needed) **rescale** outputs back. Scaling keeps arithmetic cheap while remaining close to the original FP behavior.
- **Small, fast circuits when possible:** Where safe, we fuse common patterns (e.g., **Linear + ReLU**, **Conv + ReLU**) into streamlined circuit fragments to reduce constraints.

---

## Installation

### Installing from PyPI (Recommended)

#### Prerequisites
- **UV**: Fast Python package manager ([install UV](https://docs.astral.sh/uv/getting-started/installation/))

#### Install JSTprove
```bash
uv tool install JSTprove
```

#### Verify installation
```bash
jstprove --help
```

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

- **Python**: >=3.9
- **UV**: Fast Python package manager ([install UV](https://docs.astral.sh/uv/getting-started/installation/))

> Note: UV will automatically install and manage the correct Python version for you.

> **Heads-up:** If you just installed `uv` and the command isn't found, **close and reopen your terminal** (or re-source your shell init file) so the `uv` shim is picked up on`PATH`.

### 1) System packages

> Run commands from the **repo root** so the runner binary path (e.g., `./target/release/jstprove`) resolves.

#### Ubuntu/Debian
```bash
sudo apt-get update && sudo apt-get install -y \
  pkg-config libclang-dev clang
```

#### macOS

```bash
brew install llvm
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

### 3) Clone JSTprove

```bash
git clone https://github.com/inference-labs-inc/JSTprove.git
cd JSTprove
```

> The Python package has no pip-installable dependencies. It is built with [maturin](https://www.maturin.rs/) from the PyO3 crate (`maturin develop --release`).

---

### 4) Build the JSTprove binaries

```bash
cargo build --release
```

> Cargo will automatically fetch the ECC dependency from `https://github.com/inference-labs-inc/ecc`. No local clone of Expander or ECC is needed.

---

### 5) Verify the build

```bash
./target/release/jstprove --help
./target/release/jstprove-remainder --help
```

You can now follow the **Quickstart** commands (compile -> witness -> prove -> verify).

</details>

---

## Quickstart (LeNet demo)

Demo paths:

* ONNX model: `rust/jstprove_remainder/models/lenet.onnx`
* Artifacts: `artifacts/lenet/*`

1. **Compile** (using the Expander backend) -- produces a `CompiledCircuit` msgpack bundle

```bash
jstprove msgpack_compile \
  --onnx rust/jstprove_remainder/models/lenet.onnx \
  -c artifacts/lenet/circuit.msgpack
```

2. **Witness** -- generate witness from compiled circuit + inputs/outputs

```bash
jstprove run_gen_witness \
  -c artifacts/lenet/circuit.msgpack \
  -i artifacts/lenet/input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.msgpack
```

3. **Prove** -- witness -> proof

```bash
jstprove run_prove_witness \
  -c artifacts/lenet/circuit.msgpack \
  -w artifacts/lenet/witness.msgpack \
  -p artifacts/lenet/proof.msgpack
```

4. **Verify** -- check the proof

```bash
jstprove run_gen_verify \
  -c artifacts/lenet/circuit.msgpack \
  -i artifacts/lenet/input.json \
  -o artifacts/lenet/output.json \
  -w artifacts/lenet/witness.msgpack \
  -p artifacts/lenet/proof.msgpack
```

If it prints **Verified**, you're done.

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

## Disclaimer

**JSTProve** is **experimental and unaudited**. It is provided on an **open-source, “as-is” basis**, without any warranties or guarantees of fitness for a particular purpose.

Use of JSTProve in **production environments is strongly discouraged**. The codebase may contain bugs, vulnerabilities, or incomplete features that could lead to unexpected results, failures, or security risks.

By using, modifying, or distributing this software, you acknowledge that:

 - It has not undergone a formal security review or audit.
 - It may change substantially over time, including breaking changes.
 - You assume full responsibility for any outcomes resulting from its use.

JSTProve is made available in the spirit of **research, experimentation, and community collaboration**. Contributions are welcome, but please proceed with caution and do not rely on this software for systems where correctness, reliability, or security are critical.

---

## Acknowledgments

We gratefully acknowledge [**Polyhedra Network**](https://polyhedra.network/) for:

* [**Expander**](https://github.com/PolyhedraZK/Expander) — the GKR/sumcheck proving system we build on.

* [**Expander Compiler Collection (ECC)**](https://github.com/inference-labs-inc/ecc) — the circuit frontend used to construct arithmetic circuits for ML layers.
