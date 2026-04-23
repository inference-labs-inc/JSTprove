# Conformance tolerance table

All values are α-scaled fixed-point integers (α = 2^18 = 262144).
Tolerances are expressed in **quantised units** (abs) and fractional error (rel).

| Op group | Ops covered | abs tolerance | rel tolerance | Reason |
|---|---|---|---|---|
| Rescaling arithmetic | Gemm, MatMul, Conv 1×1, Conv 3×3, ConvTranspose, BatchNorm | 3 | 0 | Two-pass fixed-point multiply/divide introduces ≤1 ULP rounding per multiply and the bias adds at most one more. |
| Rescaling arithmetic | Div (INT64 divisor) | 0 | 0 | Exact integer division; no floating-point rounding. |
| Rescaling arithmetic | Pow (integer exponent 2 or 3) | 0 | 0 | Pure integer polynomial; no rounding. |
| Transcendental | Exp, Sigmoid, Softmax, Log, Tanh | 5 | 0 | Hint computes in f64 with a single `round()`; tract and JSTProve use identical rounding, yielding ≤1 ULP. Tolerance of 5 absorbs the f32→f64 cast rounding present in the tract reference. |
| Transcendental | LayerNorm | 5 | 0 | Two-level rounding (mean, variance, normalise). |
| Transcendental | ReduceMean | 5 | 0 | Single division rounded to i64. |
| Transcendental | Sqrt | 5 | 0 | f64 sqrt + round; agrees with tract to within 1 ULP but the tract reference operates in f32 so 5-unit headroom is kept. |
| Pooling | MaxPool | 0 | 0 | Exact max-selection; no arithmetic rounding. |
| Pooling | GlobalAveragePool | 5 | 0 | Sum divided by pool area; one rounding step. |
| Pooling | AveragePool | n/a | n/a | reference_only (JSTProve does not implement AveragePool). |
| Normalization | InstanceNorm, GroupNorm | n/a | n/a | reference_only (not yet implemented). |
| Spatial | Resize nearest | 0 | 0 | Nearest-neighbour copy; no interpolation. |
| Spatial | Resize linear (uniform input) | 5 | 0 | Bilinear interpolation rounded to i64; tested with uniform values to avoid boundary-extrapolation sign disagreement between tract and JSTProve. |
| TopK | TopK | 0 | 0 | Sorting / selection; no arithmetic rounding. |

## Notes

- **α = 262144 = 2^18** is JSTProve's fixed-point scale factor.  One "unit" in abs tolerance corresponds to `1/α ≈ 3.8 × 10⁻⁶` in real-valued terms.
- **reference_only** cases run only the tract reference backend; the JSTProve circuit backend is not invoked.  They serve as placeholder tests for ops that are not yet circuit-implemented.
- **Resize linear boundary behaviour**: JSTProve's `resize_hint` clamps negative interpolation results to zero.  The test uses a spatially uniform input to avoid boundary-extrapolation differences between tract and the JSTProve hint.
- **Erf, Cos, Sin**: Excluded from the conformance suite because their `FunctionLookupTable` / `DecomposedExpLookup` gadgets generate ~4 M circuit variables, causing OOM in the debug test binary.  These ops are covered by the full end-to-end pipeline test.
- **GridSample**: Excluded because tract does not implement it.  Covered by the full end-to-end pipeline test.
- **Gelu**: Not in ONNX opset 17 standard form supported by tract.  Covered by the full end-to-end pipeline test.
