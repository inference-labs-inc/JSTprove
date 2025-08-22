# Developer Notes

Internal notes for contributors working on JSTProve (Python + Rust).

> For environment setup, pre-commit, formatting policy, and PR workflow, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Repo layout

```

.
├─ python/                    # CLI and pipeline
│  └─ frontend/cli.py         # jstprove CLI entrypoint
├─ python/testing/            # unit/integration tests
│  └─ core/tests/             # CLI tests, etc.
└─ rust/
   └─ jstprove\_circuits/     # Rust crate: circuits + runner

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

Artifacts typically appear under `./target/release/`.

> The CLI **compile** step will (re)build the runner automatically when needed.

---

## Python tests

* **Unit** CLI tests **mock** `base_testing` (fast; no heavy Rust).
* Integration/E2E for heavy models live elsewhere; use `simple_circuit` or small ONNX models for smoke tests.

Examples:

```bash
# run unit + integration markers from repo root
pytest --unit --integration

# run only CLI tests
pytest python/testing/core/tests/test_cli.py -q
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
