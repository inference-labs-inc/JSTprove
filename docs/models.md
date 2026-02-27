# Models

This page explains what kinds of models JSTprove supports and how they're handled internally.

---

## Supported operators (current)

- **Arithmetic:** **Add**, **Div**, **Mul**, **Sub**
- **Comparison:** **Max**, **Min**
- **Convolution:** **Conv**
- **Activation:** **Clip**, **ReLU**
- **Pooling:** **MaxPool**
- **Linear:** **Gemm**
- **Shaping / graph ops:** **Constant**, **Flatten**, **Reshape**, **Squeeze**, **Unsqueeze**
- **Normalization:** **BatchNormalization**

---

## ONNX expectations

- Export models with ops limited to the supported operators listed above.

---

## Quantization

- Quantization is **automatic** in the pipeline during **compile**.
- Internally, inputs and weights are scaled to integers, and tensors are reshaped to the expected shapes before witness generation.
- The CLI's **witness** and **verify** stages take care of **rescale + reshape** via circuit helpers.

---

## Input / Output JSON

- **Input JSON** should contain your model inputs as numeric arrays.
  - If values are floats, they'll be **scaled and rounded** automatically during witness/verify.
  - If your key is named exactly `"input"` (single-input models), it will be reshaped to the model's input shape.
- Multi-input models are now supported.
  - Make sure to match the name of the inputs to the model, to the inputs that the model expects to receive.

**Single-input example (flattened vector):**

```json
{
  "input": [0, 1, 2, 3, 4, 5]
}
```

**Single-input example (already shaped, e.g., 1×1×28×28):**

```json
{
  "input": [[[[0, 1, 2, "... 28 values ..."], "... 28 rows ..."]]]
}
```

- **Output JSON** produced by the pipeline is written under the key `"output"`, e.g.:

```json
{
  "output": [0, 0, 1, 0, 0, 0, 0]
}
```

---

## Best practices

- Use **one** ONNX model per compile. If you change the model, **re-run compile** to refresh the circuit and quantization.
- Keep a consistent set of artifacts from the same compile run (compiled circuit, witness, proof).
- For large CNNs, start with a small batch size and small inputs to validate the pipeline before scaling up.
