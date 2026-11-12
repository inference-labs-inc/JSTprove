# Developer Notes

Internal notes for contributors working on JSTprove.

> For environment setup, pre-commit, formatting policy, and PR workflow, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Repo layout

```
.
├─ pysrc/
│  └─ jstprove/
│     └─ __init__.py            # PyO3 bindings: holographic_setup_vk
├─ rust/
│  ├─ jstprove_circuits/        # Layer circuits, runner, CLI (jstprove binary)
│  ├─ jstprove_io/              # Serialization: msgpack envelope, zstd compression
│  ├─ jstprove_onnx/            # ONNX parsing utilities
│  └─ jstprove_pyo3/            # PyO3 bridge (excluded from workspace)
├─ docs/
├─ hooks/
├─ Cargo.toml                   # Workspace root
└─ pyproject.toml                # Maturin-based Python build
```

---

## Rust binaries

- `jstprove` -- Expander backend CLI (from `jstprove_circuits`)
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

The only Python source file is `pysrc/jstprove/__init__.py`, which re-exports `holographic_setup_vk` from the native extension.

---

## Notes & conventions

- The circuit type used by the Expander backend is `Circuit`.
- Paths are **explicit** (no inference).
- Keep artifacts from the **same compile** together. If the ONNX model changes, **re-run compile**.
- Run commands from the **repo root** so `./target/release/*` is resolvable.
