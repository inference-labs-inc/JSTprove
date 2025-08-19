# Overview

## What is JSTProve?

**JSTProve** is a [zkML](https://docs.inferencelabs.com/zk-ml) toolkit that produces [**zero-knowledge proofs**](https://docs.inferencelabs.com/resources/glossary#zero-knowledge-proof) of [neural network](https://docs.inferencelabs.com/resources/glossary#neural-networks) [inference](https://docs.inferencelabs.com/resources/glossary#inference).  
You bring an **ONNX** model; JSTProve handles **quantization**, **circuit generation**, **witness creation**, **proving**, and **verification** — via a simple CLI.

- Get started: see the [Quickstart](quickstart.md)
- Full command details: see the [CLI Reference](cli.md)

---

## Architecture (high-level)

- **Prover backend:** [Polyhedra Network's **Expander**](https://github.com/PolyhedraZK/Expander), a GKR / sumcheck–based proving system.
- **Circuit frontend:** [**Expander Compiler Collection (ECC)**](https://github.com/PolyhedraZK/ExpanderCompilerCollection) Rust API to build arithmetic circuits for ML layers.
- **Rust crate:** `rust/jstprove_circuits` implements layer circuits (Conv2D, ReLU, MaxPool2D, GEMM/FC) and a runner that talks to Expander.
- **Python pipeline:** Converts **ONNX → quantized ONNX**, sets up I/O, drives the Rust runner, and exposes the **CLI**.

```text
ONNX model  ──► Quantizer ──► Circuit (ECC/Expander) ──► Witness ──► Proof ──► Verify
                  (Py)            (Rust)                 (Rust)       (Rust)    (Rust)
```
---

## Supported operators (current)

- **Linear:** Fully Connected / GEMM, Matrix Multiply
- **Convolution:** Conv2D
- **Activation:** ReLU
- **Pooling:** MaxPool2D

---

## What you do vs. what JSTProve does

- **You (the user):**
  - Provide an **ONNX** model and an **input JSON**.
  - Run the CLI steps: **compile → witness → prove → verify**.

- **JSTProve:**
  - **Quantizes** the model to integers (power-of-two scaling).
  - **Builds** an arithmetic circuit via ECC.
  - **Runs** the quantized model to produce expected outputs.
  - **Generates** the witness and **produces** a proof with Expander.
  - **Verifies** the proof.

---

## Design principles

- **Explicit & reproducible:** You pass exact paths; we emit concrete artifacts (circuit, quantized ONNX, witness, proof). No hidden discovery or heuristics.
- **Clear separation:** Python orchestrates the pipeline and I/O; Rust implements the circuits and invokes Expander/ECC.
- **Quantization that’s simple & faithful:** We scale tensors by a **power of two**, **round to integers**, run the model, and (where needed) **rescale** outputs back. Power-of-two scaling keeps arithmetic cheap while remaining close to the original FP behavior.
- **Focused operator set (for now):** We currently support **Conv2D**, **GEMM/MatMul (FC)**, **ReLU**, **MaxPool2D**. This reflects what we've circuitized so far; we plan to add more ops incrementally.
- **Small, fast circuits when possible:** Where safe, we fuse common patterns (e.g., **Linear + ReLU**, **Conv + ReLU**) into streamlined circuit fragments to reduce constraints.
- **Deterministic debugging:** We prefer loud failures and inspectable intermediates (e.g., `*_reshaped.json`) over implicit magic.
- **Cost-aware testing:** Unit/CLI tests mock heavy code paths for speed; end-to-end runs are reserved for targeted scenarios.

---

See the [Quickstart](quickstart.md) for a 10-minute demo and the [CLI Reference](cli.md) for all flags.