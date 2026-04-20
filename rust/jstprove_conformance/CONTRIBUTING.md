# Contributing to the JSTProve Conformance Harness

This document explains how to add a new operator to the conformance harness, how CI works,
and how to diagnose failures.

---

## Overview

The conformance harness (`jstprove-conformance`) is a differential testing framework.
For each operator it:

1. Builds a minimal single-op ONNX model.
2. Runs the model through **tract** (reference backend) to get expected outputs.
3. Runs the same model through **JSTProve** (witness generation only — no proving).
4. Compares outputs element-wise within a per-operator tolerance.

All test cases are deterministic: fixed seeds, fixed input values, fixed shapes.

---

## Running the harness locally

```bash
# Full run — all operators, all cases (may take several minutes)
cargo test -p jstprove-conformance --test conformance -- --nocapture

# Full run with all failures reported (not just the first)
CONFORMANCE_FAIL_FAST=0 cargo test -p jstprove-conformance --test conformance -- --nocapture

# Fast CI-size run — 5 cases per operator group
cargo test -p jstprove-conformance --test conformance --features ci -- --nocapture

# Single operator group
cargo test -p jstprove-conformance --test conformance -- structural_ops --nocapture
```

---

## Adding a new operator `FooOp`

### Step 1 — Implement the op in `jstprove_circuits`

Follow the "Adding a new op" checklist:

- `jstprove_onnx/src/graph.rs`: add variant to `OpType`
- `jstprove_onnx/src/compat.rs`: add to `SUPPORTED_OPS`
- `jstprove_onnx/src/shape_inference.rs`: add to `infer_layer_output_shape`
- `jstprove_onnx/src/quantizer.rs`: add to `compute_layer_bound` and `is_range_check_op`
- `jstprove_circuits/src/circuit_functions/layers/`: create layer file, register in `mod.rs`
- `jstprove_circuits/src/circuit_functions/layers/layer_kinds.rs`: add to `define_layers!`
- `jstprove_circuits/src/expander_metadata.rs`: add arm to `op_type_to_string`

If the op needs a hint (transcendental, non-polynomial):
- `jstprove_circuits/src/circuit_functions/hints/`: create hint file, register in `mod.rs`

### Step 2 — Choose which file to add the case to

| Group | File | Ops |
|-------|------|-----|
| A–D (structural, arithmetic, boolean, reduction) | `src/generator/cases_m3.rs` | INT64-typed |
| E–J (rescaling, transcendental, pooling, etc.)   | `src/generator/cases_m4.rs` | FLOAT-typed |

### Step 3 — Add hardcoded test cases

Open the appropriate `cases_m*.rs` file and add cases inside the relevant `*_cases()` function.

**For INT64 ops** (no α-scaling):
```rust
// ---- FooOp ----
cases.push(exact(
    "FooOp",
    0,                              // seed (unique within this function)
    &[("x", &[4], INT64)],          // ONNX input shapes
    &[("y", &[4], INT64)],          // ONNX output shapes
    &[NodeAttr { name: "mode", value: AttrValue::Int(0) }],
    &[],                            // initializers
    vec![vec![1_i64, 2, 3, 4]],    // flat input values
));
```

**For FLOAT ops** (α-scaled inputs, real-valued weights):
```rust
// ---- FooOp ----
{
    let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let onnx_bytes = build_single_op_model_ordered(
        "FooOp",
        &[("X", &[1, 4], FLOAT)],
        &[("Y", &[1, 4], FLOAT)],
        &[NodeAttr { name: "epsilon", value: AttrValue::Float(1e-5) }],
        &[],
        &["X", "weight"],
        &[FloatInit { name: "weight", dims: vec![4], data: weight_data }],
    ).expect("build FooOp model failed");

    // Inputs are α-scaled: 1.0 real → 1 * ALPHA
    let x_vals: Vec<i64> = vec![ALPHA, 2 * ALPHA, 3 * ALPHA, 4 * ALPHA];

    cases.push(TestCase {
        op_name: "FooOp",
        seed: 0xF100,       // unique seed for FooOp cases
        onnx_bytes,
        inputs: vec![x_vals],
        tolerance: tol(2),  // ±2 α-scaled ULPs; adjust based on op precision
    });
}
```

Add at least **3 diverse cases** per operator:
- A baseline case with typical values
- An edge case (zeros, large values, or adversarial shapes)
- A case that exercises the specific code path you implemented

### Step 4 — Register the group test (if adding a new group)

If you're adding a brand-new operator group (not an existing milestone group), add a public
function in `cases_m*.rs` and wire it up in:

- `src/generator/mod.rs`: add `pub use cases_m*::{...}`
- `tests/conformance.rs`: add a `#[test]` function calling `run_group`

For groups within existing milestones, the new cases are automatically picked up by the
existing `*_cases()` call.

### Step 5 — Choose the right tolerance

| Tolerance | When to use |
|-----------|-------------|
| `Tolerance::EXACT` | INT64 ops, structural ops — no rounding ever |
| `tol(1)` | Single rescale pass (e.g., `x / alpha` with one division) |
| `tol(2)` | Two rescale passes, or ops with f32→i64 rounding (Gemm, Conv) |
| `tol(5)` | Complex hint ops with accumulated rounding (LayerNorm, Gelu) |
| `Tolerance { abs, rel, reason }` | Custom: document your reasoning in `tolerance_table.md` |

### Step 6 — If tract does not support the op

If tract cannot run the op, mark the case as `reference_only` by wrapping it in a
`TestCase` struct with no explicit flag — the runner detects this automatically when
the full JSTProve run errors but the reference run passes.

Alternatively, provide a hardcoded expected output and compare against it directly in
a custom `#[test]` function rather than using `run_group`.

### Step 7 — Add a regression fixture for historical failures

If your implementation has a known difficult edge case, or you fixed a bug, add a
regression fixture in `src/fixtures.rs`:

```rust
fn f_my_op_edge_case() -> RegressionFixture {
    let onnx_bytes = build_single_op_model(
        "FooOp",
        &[("x", &[4], INT64)],
        &[("y", &[4], INT64)],
        &[],
        &[],
    ).expect("build FooOp fixture failed");

    RegressionFixture {
        id: "fooOp_edge_case",
        fixed_in: "PR #NNN",
        failure_description: "FooOp with inputs [...] produced wrong output due to ...",
        case: TestCase {
            op_name: "FooOp",
            seed: 0xFXXX,
            onnx_bytes,
            inputs: vec![vec![...]],
            tolerance: Tolerance::EXACT,
        },
        allow_jstprove_error: false,
    }
}
```

Then register it in `all_regression_fixtures()`.

### Step 8 — Run and verify

```bash
# Check only your new op
cargo test -p jstprove-conformance --test conformance -- <your_test_fn> --nocapture

# Check the full suite still passes
cargo test -p jstprove-conformance --test conformance -- --nocapture

# Check clippy is clean
cargo clippy -p jstprove-conformance --all-targets -- -D warnings
```

---

## CI workflow

The conformance harness runs on every PR that touches:
- `rust/jstprove_circuits/src/circuit_functions/` — operator circuit implementations
- `rust/jstprove_onnx/src/` — ONNX parsing and quantization
- `rust/jstprove_conformance/` — the harness itself

The CI job uses `--features ci` which limits each operator group to 5 cases
(instead of the full set) to stay under the 5-minute budget.

**If CI fails:**
1. Look at the step log for the failing op name, seed, and element delta.
2. Reproduce locally:
   ```bash
   CONFORMANCE_FAIL_FAST=0 cargo test -p jstprove-conformance --test conformance -- --nocapture 2>&1 | grep -A5 "FAIL\|ERROR"
   ```
3. Use the seed to reproduce the exact inputs deterministically.

---

## Tolerance table

See `src/tolerance_table.md` for the per-operator tolerance justifications.

---

## Architecture of the harness

```
src/
  lib.rs                  — public re-exports
  tolerance.rs            — Tolerance struct and check logic
  onnx_builder.rs         — builds single-op ONNX models (protobuf)
  runner.rs               — ConformanceRunner: tract + JSTProve paths
  fixtures.rs             — RegressionFixture and all_regression_fixtures()
  generator/
    mod.rs                — re-exports, default_case_count()
    builder.rs            — TestCaseBuilder, shrink(), DEFAULT_SEEDS
    cases_m3.rs           — Group A–D: structural/arithmetic/boolean/reduction
    cases_m4.rs           — Group E–J: rescaling/transcendental/pooling/spatial/topk
    op_specs.rs           — OpInputSpec, TensorSpec (for property-based generation)
    shapes.rs             — ShapeSpec (vec, matrix, tensor, broadcast pair, …)
    values.rs             — ValueSpec (mixed, near-zero, boundary, …), ALPHA, SAFE_RANGE
tests/
  conformance.rs          — #[test] functions, run_group helper
```

Key constants:
- `ALPHA = 2^18 = 262144` — the fixed-point scale used throughout JSTProve.
- `SAFE_RANGE = (-ALPHA * 8192, ALPHA * 8192)` — ≈ ±2^31, the safe quantized value range.
- `DEFAULT_SEEDS` — 10 fixed seeds covering baseline, adversarial, boundary, extreme cases.
- `default_case_count()` — returns 5 under `--features ci`, `usize::MAX` otherwise.
