# Troubleshooting

Common issues and quick fixes.

---

## Runner not found

- Run commands from the **repo root** so `./target/release/*` is visible.
- Re-run **compile** (it will build the runner automatically if needed).


---

## Shape or "out of bounds" errors during witness

- Ensure your `--input-path` matches the model's input **shape**.
- Re-run **compile** after changing the model (to refresh circuit + quantization).
- Make sure **witness** and **verify** both use the **same `quantized.onnx`** produced by the last compile.

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

- Keep artifacts from the **same compile** together: `circuit.txt` + `quantized.onnx`.
- If anything looks mismatched, re-run **compile → witness → prove → verify** end-to-end.
- Set `JSTPROVE_NO_BANNER=1` or use `--no-banner` for quiet logs in CI.
