# Developer Notes

Internal notes for contributors working on JSTprove.

> For environment setup, pre-commit, formatting policy, and PR workflow, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Repo layout

```
.
├─ pysrc/
│  └─ jstprove/
│     └─ __init__.py            # PyO3 bindings: Circuit, WitnessResult, BatchResult
├─ rust/
│  ├─ jstprove_circuits/        # Layer circuits, runner, CLI (jstprove binary)
│  ├─ jstprove_io/              # Serialization: msgpack envelope, zstd compression
│  ├─ jstprove_onnx/            # ONNX parsing utilities
│  ├─ jstprove_remainder/       # Remainder backend circuits + CLI (jstprove-remainder binary)
│  └─ jstprove_pyo3/            # PyO3 bridge (excluded from workspace)
├─ docs/
├─ hooks/
├─ Cargo.toml                   # Workspace root
└─ pyproject.toml                # Maturin-based Python build
```

---

## Rust binaries

- `jstprove` -- Expander backend CLI (from `jstprove_circuits`)
- `jstprove-remainder` -- Remainder backend CLI (from `jstprove_remainder`)
- `simple_circuit` -- minimal demo circuit

Build from repo root:

```bash
cargo build --release
```

Artifacts appear under `./target/release/`.

---

## Python package

The Python package is a thin PyO3 binding layer with no pip dependencies. It is built with [maturin](https://www.maturin.rs/):

```bash
maturin develop --release
```

The only Python source file is `pysrc/jstprove/__init__.py`, which re-exports `Circuit`, `WitnessResult`, and `BatchResult` from the native extension.

---

## Notes & conventions

- The circuit type used by the Expander backend is `Circuit`.
- Paths are **explicit** (no inference).
- Keep artifacts from the **same compile** together. If the ONNX model changes, **re-run compile**.
- Run commands from the **repo root** so `./target/release/*` is resolvable.
