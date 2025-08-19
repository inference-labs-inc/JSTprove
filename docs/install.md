# Installation

This page covers setting up JSTProve for local development and CLI use.

---

## System requirements

- **Python 3.12**
- **Rust toolchain (stable)** — install via [rustup](https://rustup.rs)
- A Unix-like shell (macOS/Linux)

> We recommend running JSTProve commands from the **repo root** so the runner binary path resolves cleanly.

---

## 1) Clone and create a virtualenv

```bash
git clone <YOUR_JSTPROVE_REPO_URL>
cd jstprove

# create & activate a virtualenv
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
````

---

## 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

> Optional (developer convenience):
> If you want to import the project as a package elsewhere or plan to add console scripts later:
>
> ```bash
> pip install -e .
> ```

---

## 3) Install Rust toolchain

If you don't have it yet:

```bash
# one-liner installer (macOS/Linux/WSL):
curl https://sh.rustup.rs -sSf | sh

# verify
rustc --version
cargo --version
```

---

## 4) (Optional) Pre-build the runner

Not required — the CLI's **compile** step will (re)build it automatically when needed.
If you want to verify your Rust setup:

```bash
# from repo root
cargo build --release

# sanity check
./target/release/onnx_generic_circuit --help
```

---

## 5) External dependency: Expander / ECC

JSTProve integrates **Expander** (GKR/sumcheck prover) and **ECC** (circuit compiler) as **Rust dependencies**.

* **Default:** Cargo fetches these crates automatically; you do **not** need to clone Expander manually.
* **Advanced (pinned/local checkout):**
  If your organization wants to build against a **local Expander/ECC repo**, either:

  * Point dependencies to a local `path` in `Cargo.toml`, or
  * Use a `[patch]` section to override the git source with your local path.

> If you go the local-repo route, follow the Expander/ECC README for their prerequisites. JSTProve does not bundle or vend those repos.

---

## 6) Verify the installation

From the repo root, try printing the CLI help:

```bash
python -m python.frontend.cli --help
```

You can also run a quick compile (will build the runner if needed):

```bash
python -m python.frontend.cli compile \
  -m python/models/models_onnx/doom.onnx \
  -c artifacts/doom/circuit.txt \
  -q artifacts/doom/quantized.onnx
```

If that succeeds, you’re ready to run **witness → prove → verify** (see [**Quickstart**](quickstart.md)).
