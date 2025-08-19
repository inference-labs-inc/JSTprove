# Developer Notes

These notes are for contributors working on JSTProve internals (Python + Rust).

---

## Repo layout

```

.
├─ python/                   # CLI and pipeline
│  └─ frontend/cli.py        # jstprove CLI entrypoint
├─ python/testing/           # unit/integration tests
│  └─ core/tests/            # CLI tests, etc.
└─ rust/
    └─ jstprove_circuits/    # Rust crate: circuits + runner

````

---

## Rust binaries

- Main runner: `onnx_generic_circuit`
- Simple demo: `simple_circuit`

You can build them manually if needed:

```bash
# from repo root
cargo build --release
# or explicitly:
cargo build --release --manifest-path rust/jstprove_circuits/Cargo.toml
````

* Artifacts typically appear under `./target/release/` (or the crate’s `target/release/` depending on your workspace setup).
* The CLI **compile** step will (re)build the runner automatically when needed (dev build path).

---

## Python tests

* **Unit** CLI tests **mock** `base_testing` (fast; no heavy Rust runs).
* Integration/E2E that compile heavy models live elsewhere; use the Rust `simple_circuit` binary or small ONNX models for quick checks.

Typical invocations:

```bash
# run unit + integration markers from repo root
pytest --unit --integration

# run only CLI tests
pytest python/testing/core/tests/test_cli.py -q
```

---

## Useful environment variables

* Suppress banner in non-interactive runs:

  ```bash
  export JSTPROVE_NO_BANNER=1
  ```

---

## Releases / tags

Tag a release:

```bash
git tag -a vX.Y.Z -m "JSTProve vX.Y.Z"
git push origin vX.Y.Z
```

---

## Notes & conventions

* The CLI uses the **GenericModelONNX** circuit by default (no class/name flags).
* Paths are **explicit** (no inference).
* Keep artifacts from the **same compile** together: `circuit.txt` + `quantized.onnx`; if the ONNX changes, **re-run compile**.
