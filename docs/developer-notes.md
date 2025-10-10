# Developer Notes

Internal notes for contributors working on JSTprove (Python + Rust).

> For environment setup, pre-commit, formatting policy, and PR workflow, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Repo layout

```

.
├─ python/                    # CLI and pipeline
│  └─ frontend/cli.py         # JSTprove CLI entrypoint
├─ python/testing/            # unit/integration tests
│  └─ core/tests/             # CLI tests, etc.
└─ rust/
   └─ jstprove_circuits/     # Rust crate: circuits + runner

````

---

## Rust binaries

- Main runner: `onnx_generic_circuit`
- Simple demo: `simple_circuit`

You can build them manually if needed:

```bash
# from repo root
cargo build --release
# or explicitly (if not using a workspace root):
cargo build --release --manifest-path rust/jstprove_circuits/Cargo.toml
````

These must be built manually if you are making changes to the rust side of the codebase, without the entire codebase package updating.

Artifacts typically appear under `./target/release/`.

> The CLI **compile** step will (re)build the runner automatically when needed.

---

## Python tests

Before running tests, make sure to install test dependencies:

```bash
uv sync --group test
```

* **Unit** CLI tests **mock** `base_testing` (fast; no heavy Rust).
* Integration/E2E for heavy models live elsewhere; use `simple_circuit` or small ONNX models for smoke tests.

Examples:

```bash
# run unit + integration markers from repo root
uv run pytest --unit --integration

# run e2e tests.
Place model to be run in python/models/models_onnx/<model_name>.onnx
uv run pytest --e2e --<model_name>
```

---

## Useful environment variables

Suppress the ASCII banner in non-interactive runs:

```bash
export JSTPROVE_NO_BANNER=1
```

---

## Notes & conventions

* The CLI uses **GenericModelONNX** by default (no circuit class/name flags).
* Paths are **explicit** (no inference).
* Keep artifacts from the **same compile** together: `circuit.txt` + `quantized.onnx`.
  If the ONNX changes, **re-run compile**.
* Run commands from the **repo root** so `./target/release/*` is resolvable.

```
