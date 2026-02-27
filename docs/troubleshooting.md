# Troubleshooting

Common issues and quick fixes.

---

## Runner not found

- Run commands from the **repo root** so `./target/release/jstprove` and `./target/release/jstprove-remainder` are visible.
- Ensure you have run `cargo build --release`.

---

## Shape or "out of bounds" errors during witness

- Ensure your `--input` matches the model's input **shape**.
- Re-run **compile** after changing the model (to refresh the compiled circuit).
- Make sure **witness** and **verify** both use the **same compiled circuit** produced by the last compile.

---

## Verification complains about shapes

- If your model has multiple inputs, ensure the input JSON includes **all input keys** with correct shapes.

---

## Slow runs

- Large CNNs are heavy. For smoke tests:
  - Use a **smaller model** or **reduced input size**.
  - Or use the `simple_circuit` Rust binary to validate the toolchain quickly.

---

## General tips

- Keep artifacts from the **same compile** together (compiled circuit + witness + proof).
- If anything looks mismatched, re-run **compile -> witness -> prove -> verify** end-to-end.
