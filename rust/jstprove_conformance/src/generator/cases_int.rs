//! Fixed test cases for Group A–D operators (structural, arithmetic, boolean, reduction).
//!
//! Design notes:
//! - All test cases use **INT64** typed tensors (elem_type = 7).  JSTProve
//!   detects INT64 and bypasses alpha-quantisation, so the circuit sees the
//!   raw integer values — and tract also computes directly in integer
//!   arithmetic.  This keeps the conformance check simple and exact.
//!
//! - Operators that multiply two alpha-scaled values (Mul, LeakyRelu,
//!   HardSwish) require FLOAT typed inputs to agree between JSTProve and
//!   tract. Those ops are tested in `reference_only` mode (tract path only).
//!
//! - Every test case calls `build_single_op_model` directly so that shapes,
//!   initialisers, and attributes are fully explicit and reproducible.

use super::default_case_count;
use crate::onnx_builder::{
    build_single_op_model, build_single_op_model_ordered, AttrValue, Initializer, NodeAttr,
};
use crate::runner::TestCase;
use crate::tolerance::Tolerance;

const INT64: i32 = 7;
const BOOL: i32 = 9;

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn make_case(
    op_name: &'static str,
    seed: u64,
    input_shapes: &[(&str, &[i64], i32)],
    output_shapes: &[(&str, &[i64], i32)],
    attrs: &[NodeAttr],
    initializers: &[Initializer],
    inputs: Vec<Vec<i64>>,
    tolerance: Tolerance,
) -> TestCase {
    let onnx_bytes =
        build_single_op_model(op_name, input_shapes, output_shapes, attrs, initializers)
            .unwrap_or_else(|e| panic!("build_single_op_model({op_name}) failed: {e:#}"));
    TestCase {
        op_name,
        seed,
        onnx_bytes,
        inputs,
        tolerance,
        ignore_extra_reference_outputs: false,
        allow_jstprove_error: false,
    }
}

fn exact(
    op_name: &'static str,
    seed: u64,
    input_shapes: &[(&str, &[i64], i32)],
    output_shapes: &[(&str, &[i64], i32)],
    attrs: &[NodeAttr],
    initializers: &[Initializer],
    inputs: Vec<Vec<i64>>,
) -> TestCase {
    make_case(
        op_name,
        seed,
        input_shapes,
        output_shapes,
        attrs,
        initializers,
        inputs,
        Tolerance::EXACT,
    )
}

// ---------------------------------------------------------------------------
// Group A — Structural ops
// ---------------------------------------------------------------------------

/// Returns test cases for all structural operators.
/// All use INT64 tensors and EXACT tolerance (no arithmetic, no rounding).
pub fn structural_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- Reshape ----
    // [12] → [3, 4]
    cases.push(exact(
        "Reshape",
        0,
        &[("data", &[12], INT64)],
        &[("output", &[3, 4], INT64)],
        &[],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![3, 4],
        }],
        vec![(1_i64..=12).collect()],
    ));
    // [6] → [2, 3]
    cases.push(exact(
        "Reshape",
        1,
        &[("data", &[6], INT64)],
        &[("output", &[2, 3], INT64)],
        &[],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![2, 3],
        }],
        vec![vec![10, 20, 30, 40, 50, 60]],
    ));

    // ---- Flatten ----
    // [2, 3, 4] axis=1 → [2, 12]
    cases.push(exact(
        "Flatten",
        0,
        &[("data", &[2, 3, 4], INT64)],
        &[("output", &[2, 12], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(1),
        }],
        &[],
        vec![(1_i64..=24).collect()],
    ));
    // [2, 3, 4] axis=0 → [1, 24]
    cases.push(exact(
        "Flatten",
        1,
        &[("data", &[2, 3, 4], INT64)],
        &[("output", &[1, 24], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[],
        vec![(1_i64..=24).collect()],
    ));

    // ---- Squeeze ----
    // [1, 4, 1] axes=[0, 2] → [4]
    cases.push(exact(
        "Squeeze",
        0,
        &[("data", &[1, 4, 1], INT64)],
        &[("output", &[4], INT64)],
        &[],
        &[Initializer {
            name: "axes",
            dims: vec![2],
            data: vec![0, 2],
        }],
        vec![vec![10, 20, 30, 40]],
    ));
    // [1, 1, 3] axes=[0, 1] → [3]
    cases.push(exact(
        "Squeeze",
        1,
        &[("data", &[1, 1, 3], INT64)],
        &[("output", &[3], INT64)],
        &[],
        &[Initializer {
            name: "axes",
            dims: vec![2],
            data: vec![0, 1],
        }],
        vec![vec![5, 6, 7]],
    ));

    // ---- Unsqueeze ----
    // [4] axes=[0, 2] → [1, 4, 1]
    cases.push(exact(
        "Unsqueeze",
        0,
        &[("data", &[4], INT64)],
        &[("output", &[1, 4, 1], INT64)],
        &[],
        &[Initializer {
            name: "axes",
            dims: vec![2],
            data: vec![0, 2],
        }],
        vec![vec![10, 20, 30, 40]],
    ));
    // [3] axes=[1] → [3, 1]
    cases.push(exact(
        "Unsqueeze",
        1,
        &[("data", &[3], INT64)],
        &[("output", &[3, 1], INT64)],
        &[],
        &[Initializer {
            name: "axes",
            dims: vec![1],
            data: vec![1],
        }],
        vec![vec![7, 8, 9]],
    ));

    // Cast: INT64→INT64 triggers tract optimization that returns TDim, which our
    // run_tract i64 extractor can't handle non-INT64 Cast outputs currently.

    // ---- ConstantOfShape ----
    // shape=[3, 4] fill=0 (INT64) → [3, 4] all zeros
    // Explicitly specify INT64 fill value — without it, ONNX defaults to float32
    cases.push(exact(
        "ConstantOfShape",
        0,
        &[], // no dynamic inputs
        &[("output", &[3, 4], INT64)],
        &[NodeAttr {
            name: "value",
            value: AttrValue::Tensor {
                dims: vec![],
                data_type: INT64,
                int64_data: vec![0],
            },
        }],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![3, 4],
        }],
        vec![], // no dynamic inputs
    ));
    // shape=[2, 2] fill=7 (via value attribute)
    cases.push(exact(
        "ConstantOfShape",
        1,
        &[],
        &[("output", &[2, 2], INT64)],
        &[NodeAttr {
            name: "value",
            value: AttrValue::Tensor {
                dims: vec![],
                data_type: INT64,
                int64_data: vec![7],
            },
        }],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![2, 2],
        }],
        vec![],
    ));

    // ---- Tile ----
    // [2, 3] repeats=[2, 1] → [4, 3]
    cases.push(exact(
        "Tile",
        0,
        &[("data", &[2, 3], INT64)],
        &[("output", &[4, 3], INT64)],
        &[],
        &[Initializer {
            name: "repeats",
            dims: vec![2],
            data: vec![2, 1],
        }],
        vec![vec![1, 2, 3, 4, 5, 6]],
    ));
    // [1, 4] repeats=[3, 1] → [3, 4]  (regression shape)
    cases.push(exact(
        "Tile",
        1,
        &[("data", &[1, 4], INT64)],
        &[("output", &[3, 4], INT64)],
        &[],
        &[Initializer {
            name: "repeats",
            dims: vec![2],
            data: vec![3, 1],
        }],
        vec![vec![10, 20, 30, 40]],
    ));

    // ---- Gather ----
    // data[4, 3] indices=[1, 2] axis=0 → output[2, 3]
    cases.push(exact(
        "Gather",
        0,
        &[("data", &[4, 3], INT64)],
        &[("output", &[2, 3], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[Initializer {
            name: "indices",
            dims: vec![2],
            data: vec![1, 2],
        }],
        vec![(1_i64..=12).collect()],
    ));
    // data[5] indices=[0, 4] axis=0 → output[2] (boundary indices)
    cases.push(exact(
        "Gather",
        1,
        &[("data", &[5], INT64)],
        &[("output", &[2], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[Initializer {
            name: "indices",
            dims: vec![2],
            data: vec![0, 4],
        }],
        vec![vec![10, 20, 30, 40, 50]],
    ));

    // ---- GatherElements ----
    // data[2, 4] axis=1 indices[[0,3,1,2],[3,2,0,1]] → output[2, 4]
    cases.push(exact(
        "GatherElements",
        0,
        &[("data", &[2, 4], INT64)],
        &[("output", &[2, 4], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(1),
        }],
        &[Initializer {
            name: "indices",
            dims: vec![2, 4],
            data: vec![0, 3, 1, 2, 3, 2, 0, 1],
        }],
        vec![vec![10, 20, 30, 40, 50, 60, 70, 80]],
    ));

    // ---- Transpose ----
    // [2, 3] perm=[1, 0] → [3, 2]
    cases.push(exact(
        "Transpose",
        0,
        &[("data", &[2, 3], INT64)],
        &[("output", &[3, 2], INT64)],
        &[NodeAttr {
            name: "perm",
            value: AttrValue::Ints(vec![1, 0]),
        }],
        &[],
        vec![vec![1, 2, 3, 4, 5, 6]],
    ));
    // [2, 3, 4] perm=[0, 2, 1] → [2, 4, 3]
    cases.push(exact(
        "Transpose",
        1,
        &[("data", &[2, 3, 4], INT64)],
        &[("output", &[2, 4, 3], INT64)],
        &[NodeAttr {
            name: "perm",
            value: AttrValue::Ints(vec![0, 2, 1]),
        }],
        &[],
        vec![(1_i64..=24).collect()],
    ));
    // rank-1 identity transposition [5] → [5]
    cases.push(exact(
        "Transpose",
        2,
        &[("data", &[5], INT64)],
        &[("output", &[5], INT64)],
        &[NodeAttr {
            name: "perm",
            value: AttrValue::Ints(vec![0]),
        }],
        &[],
        vec![vec![1, -2, 3, -4, 5]],
    ));

    // ---- Concat ----
    // axis=0: [3, 4] + [3, 4] → [6, 4]
    cases.push(exact(
        "Concat",
        0,
        &[("a", &[3, 4], INT64), ("b", &[3, 4], INT64)],
        &[("output", &[6, 4], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[],
        vec![(1_i64..=12).collect(), (13_i64..=24).collect()],
    ));
    // axis=1: [3, 4] + [3, 4] → [3, 8]
    cases.push(exact(
        "Concat",
        1,
        &[("a", &[3, 4], INT64), ("b", &[3, 4], INT64)],
        &[("output", &[3, 8], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(1),
        }],
        &[],
        vec![(1_i64..=12).collect(), (13_i64..=24).collect()],
    ));
    // singleton dim concat: [1, 4] + [1, 4] → [2, 4]
    cases.push(exact(
        "Concat",
        2,
        &[("a", &[1, 4], INT64), ("b", &[1, 4], INT64)],
        &[("output", &[2, 4], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[],
        vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
    ));

    // ---- Slice ----
    // data[4, 6] starts=[1, 0] ends=[3, 4] steps=[1, 1] → [2, 4]
    cases.push(exact(
        "Slice",
        0,
        &[("data", &[4, 6], INT64)],
        &[("output", &[2, 4], INT64)],
        &[],
        &[
            Initializer {
                name: "starts",
                dims: vec![2],
                data: vec![1, 0],
            },
            Initializer {
                name: "ends",
                dims: vec![2],
                data: vec![3, 4],
            },
            Initializer {
                name: "axes",
                dims: vec![2],
                data: vec![0, 1],
            },
            Initializer {
                name: "steps",
                dims: vec![2],
                data: vec![1, 1],
            },
        ],
        vec![(1_i64..=24).collect()],
    ));
    // data[8] starts=[1] ends=[7] steps=[2] → [3]
    cases.push(exact(
        "Slice",
        1,
        &[("data", &[8], INT64)],
        &[("output", &[3], INT64)],
        &[],
        &[
            Initializer {
                name: "starts",
                dims: vec![1],
                data: vec![1],
            },
            Initializer {
                name: "ends",
                dims: vec![1],
                data: vec![7],
            },
            Initializer {
                name: "axes",
                dims: vec![1],
                data: vec![0],
            },
            Initializer {
                name: "steps",
                dims: vec![1],
                data: vec![2],
            },
        ],
        vec![vec![10, 20, 30, 40, 50, 60, 70, 80]],
    ));

    // ---- Shape ----
    // data[2, 3, 4] → output[3] = [2, 3, 4]
    cases.push(exact(
        "Shape",
        0,
        &[("data", &[2, 3, 4], INT64)],
        &[("output", &[3], INT64)],
        &[],
        &[],
        vec![(1_i64..=24).collect()],
    ));
    // data[5] → output[1] = [5]
    cases.push(exact(
        "Shape",
        1,
        &[("data", &[5], INT64)],
        &[("output", &[1], INT64)],
        &[],
        &[],
        vec![vec![0, 1, 2, 3, 4]],
    ));

    // ---- Expand ----
    // [1, 4] → [3, 4]
    cases.push(exact(
        "Expand",
        0,
        &[("data", &[1, 4], INT64)],
        &[("output", &[3, 4], INT64)],
        &[],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![3, 4],
        }],
        vec![vec![1, 2, 3, 4]],
    ));
    // [1, 1] → [3, 4] (scalar-like broadcast)
    cases.push(exact(
        "Expand",
        1,
        &[("data", &[1, 1], INT64)],
        &[("output", &[3, 4], INT64)],
        &[],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![3, 4],
        }],
        vec![vec![42]],
    ));

    // ---- Pad ----
    // data[2, 3] pads=[1, 0, 0, 1] → [3, 4]  (pad row before axis-0, pad col after axis-1)
    cases.push(exact(
        "Pad",
        0,
        &[("data", &[2, 3], INT64)],
        &[("output", &[3, 4], INT64)],
        &[NodeAttr {
            name: "mode",
            value: AttrValue::String(b"constant".to_vec()),
        }],
        &[Initializer {
            name: "pads",
            dims: vec![4],
            data: vec![1, 0, 0, 1],
        }],
        vec![vec![1, 2, 3, 4, 5, 6]],
    ));

    // ---- Split ----
    // [6, 4] axis=0 split=[3, 3] → out0[3, 4], out1[3, 4]
    cases.push(exact(
        "Split",
        0,
        &[("data", &[6, 4], INT64)],
        &[("out0", &[3, 4], INT64), ("out1", &[3, 4], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[Initializer {
            name: "split",
            dims: vec![2],
            data: vec![3, 3],
        }],
        vec![(1_i64..=24).collect()],
    ));

    // ---- Identity ----
    cases.push(exact(
        "Identity",
        0,
        &[("data", &[5], INT64)],
        &[("output", &[5], INT64)],
        &[],
        &[],
        vec![vec![-2, -1, 0, 1, 2]],
    ));

    // ---- Neg ----
    cases.push(exact(
        "Neg",
        0,
        &[("data", &[5], INT64)],
        &[("output", &[5], INT64)],
        &[],
        &[],
        vec![vec![-10, -1, 0, 1, 10]],
    ));

    // ---- Range ----
    // start=0, limit=6, delta=2 → output[3] = [0, 2, 4]
    cases.push(exact(
        "Range",
        0,
        &[], // all inputs are initialisers
        &[("output", &[3], INT64)],
        &[],
        &[
            Initializer {
                name: "start",
                dims: vec![],
                data: vec![0],
            },
            Initializer {
                name: "limit",
                dims: vec![],
                data: vec![6],
            },
            Initializer {
                name: "delta",
                dims: vec![],
                data: vec![2],
            },
        ],
        vec![],
    ));
    // start=1, limit=5, delta=1 → output[4] = [1, 2, 3, 4]
    cases.push(exact(
        "Range",
        1,
        &[],
        &[("output", &[4], INT64)],
        &[],
        &[
            Initializer {
                name: "start",
                dims: vec![],
                data: vec![1],
            },
            Initializer {
                name: "limit",
                dims: vec![],
                data: vec![5],
            },
            Initializer {
                name: "delta",
                dims: vec![],
                data: vec![1],
            },
        ],
        vec![],
    ));

    // ---- ScatterND ----
    // data[4] indices=[[1],[3]] updates=[10,20] → [4] with data[1]=10, data[3]=20
    // ONNX node input order: [data, indices, updates]
    // updates is a dynamic graph input (JSTProve circuit variable).
    // indices is a compile-time initializer (JSTProve reads via w_and_b_map).
    cases.push({
        let onnx_bytes = build_single_op_model_ordered(
            "ScatterND",
            &[("data", &[4], INT64), ("updates", &[2], INT64)],
            &[("output", &[4], INT64)],
            &[],
            &[Initializer {
                name: "indices",
                dims: vec![2, 1],
                data: vec![1, 3],
            }],
            &["data", "indices", "updates"],
            &[],
        )
        .expect("build_single_op_model_ordered(ScatterND) failed");
        TestCase {
            op_name: "ScatterND",
            seed: 0,
            onnx_bytes,
            inputs: vec![vec![1, 2, 3, 4], vec![10, 20]],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
            allow_jstprove_error: false,
        }
    });

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group B — Simple element-wise arithmetic
// ---------------------------------------------------------------------------

/// Returns test cases for arithmetic operators.
/// Mul, LeakyRelu, HardSwish are tested in `reference_only` mode because
/// they require FLOAT (alpha-scaled) input semantics; full JSTProve
/// comparison is deferred until FLOAT input handling is wired up.
#[allow(clippy::vec_init_then_push)]
pub fn arithmetic_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- Add ----
    cases.push(exact(
        "Add",
        0,
        &[("a", &[8], INT64), ("b", &[8], INT64)],
        &[("output", &[8], INT64)],
        &[],
        &[],
        vec![
            vec![-4, -3, -2, -1, 0, 1, 2, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        ],
    ));
    cases.push(exact(
        "Add",
        1,
        &[("a", &[4, 2], INT64), ("b", &[4, 2], INT64)],
        &[("output", &[4, 2], INT64)],
        &[],
        &[],
        vec![(1_i64..=8).collect(), vec![0; 8]],
    ));

    // ---- Sub ----
    cases.push(exact(
        "Sub",
        0,
        &[("a", &[6], INT64), ("b", &[6], INT64)],
        &[("output", &[6], INT64)],
        &[],
        &[],
        vec![vec![10, 20, 30, 40, 50, 60], vec![1, 5, 10, 15, 25, 35]],
    ));

    // ---- Relu ----
    cases.push(exact(
        "Relu",
        0,
        &[("x", &[8], INT64)],
        &[("y", &[8], INT64)],
        &[],
        &[],
        vec![vec![-5, -3, -1, 0, 1, 3, 5, 100]],
    ));
    cases.push(exact(
        "Relu",
        1,
        &[("x", &[4, 2], INT64)],
        &[("y", &[4, 2], INT64)],
        &[],
        &[],
        vec![vec![-10, -1, 0, 0, 1, 5, 100, 200]],
    ));

    // ---- Clip ----
    // Clip [-5, 5] on range [-10, 10]
    cases.push(exact(
        "Clip",
        0,
        &[("x", &[8], INT64)],
        &[("y", &[8], INT64)],
        &[],
        &[
            Initializer {
                name: "min",
                dims: vec![],
                data: vec![-5],
            },
            Initializer {
                name: "max",
                dims: vec![],
                data: vec![5],
            },
        ],
        vec![vec![-10, -6, -5, -1, 0, 1, 5, 10]],
    ));

    // ---- Max (element-wise) ----
    cases.push(exact(
        "Max",
        0,
        &[("a", &[6], INT64), ("b", &[6], INT64)],
        &[("output", &[6], INT64)],
        &[],
        &[],
        vec![vec![-3, 0, 5, 10, -1, 8], vec![2, -1, 3, 10, 0, 9]],
    ));

    // ---- Min (element-wise) ----
    cases.push(exact(
        "Min",
        0,
        &[("a", &[6], INT64), ("b", &[6], INT64)],
        &[("output", &[6], INT64)],
        &[],
        &[],
        vec![vec![-3, 0, 5, 10, -1, 8], vec![2, -1, 3, 10, 0, 9]],
    ));

    // Mul, LeakyRelu, HardSwish require FLOAT-typed inputs for correct semantics:
    // - JSTProve assumes alpha-scaled inputs and applies rescale (÷α) for these ops.
    // - tract does not support INT64 for LeakyRelu/HardSwish.
    // Full conformance tests for these ops require FLOAT input support in the runner.

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group C — Boolean / comparison ops
// ---------------------------------------------------------------------------

#[allow(clippy::vec_init_then_push)]
pub fn boolean_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- Equal ----
    // Note: Equal/Greater/Less output BOOL (type 9) per ONNX spec.
    cases.push(exact(
        "Equal",
        0,
        &[("a", &[8], INT64), ("b", &[8], INT64)],
        &[("output", &[8], BOOL)],
        &[],
        &[],
        vec![vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1, 0, 3, 0, 5, 0, 7, 0]],
    ));

    // ---- Greater ----
    // JSTProve comparison ops use unsigned (non-negative) field elements.
    // Negative INT64 values are stored as p - |n|, making them appear as very large
    // positive numbers — so test cases here use only non-negative integers.
    cases.push(exact(
        "Greater",
        0,
        &[("a", &[6], INT64), ("b", &[6], INT64)],
        &[("output", &[6], BOOL)],
        &[],
        &[],
        vec![vec![0, 0, 1, 5, 0, 200], vec![0, 1, 0, 3, 3, 100]],
    ));

    // ---- Less ----
    cases.push(exact(
        "Less",
        0,
        &[("a", &[6], INT64), ("b", &[6], INT64)],
        &[("output", &[6], BOOL)],
        &[],
        &[],
        vec![vec![0, 0, 1, 5, 0, 200], vec![0, 1, 0, 3, 3, 100]],
    ));

    // ---- And ----
    // And inputs must be BOOL per ONNX spec.
    cases.push(exact(
        "And",
        0,
        &[("a", &[6], BOOL), ("b", &[6], BOOL)],
        &[("output", &[6], BOOL)],
        &[],
        &[],
        vec![vec![0, 1, 0, 1, 0, 1], vec![0, 0, 1, 1, 0, 1]],
    ));

    // ---- Not ----
    // Not input and output must be BOOL per ONNX spec.
    cases.push(exact(
        "Not",
        0,
        &[("x", &[6], BOOL)],
        &[("output", &[6], BOOL)],
        &[],
        &[],
        vec![vec![0, 1, 0, 1, 1, 0]],
    ));

    // ---- Where ----
    // condition must be BOOL; x/y and output can be INT64.
    cases.push(exact(
        "Where",
        0,
        &[
            ("condition", &[6], BOOL),
            ("x", &[6], INT64),
            ("y", &[6], INT64),
        ],
        &[("output", &[6], INT64)],
        &[],
        &[],
        vec![
            vec![1, 0, 1, 0, 1, 0],       // mask
            vec![10, 20, 30, 40, 50, 60], // on-true
            vec![-1, -2, -3, -4, -5, -6], // on-false
        ],
    ));

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group D — Reduction ops
// ---------------------------------------------------------------------------

#[allow(clippy::vec_init_then_push)]
pub fn reduction_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- ReduceSum ----
    // [4, 4] axes=[1] keepdims=1 → [4, 1]
    cases.push(exact(
        "ReduceSum",
        0,
        &[("data", &[4, 4], INT64)],
        &[("output", &[4, 1], INT64)],
        &[NodeAttr {
            name: "keepdims",
            value: AttrValue::Int(1),
        }],
        &[Initializer {
            name: "axes",
            dims: vec![1],
            data: vec![1],
        }],
        vec![(1_i64..=16).collect()],
    ));
    // [4, 4] axes=[0] keepdims=0 → [4]
    cases.push(exact(
        "ReduceSum",
        1,
        &[("data", &[4, 4], INT64)],
        &[("output", &[4], INT64)],
        &[NodeAttr {
            name: "keepdims",
            value: AttrValue::Int(0),
        }],
        &[Initializer {
            name: "axes",
            dims: vec![1],
            data: vec![0],
        }],
        vec![(1_i64..=16).collect()],
    ));
    // [8] axes=[0] keepdims=0 → scalar [1]
    cases.push(exact(
        "ReduceSum",
        2,
        &[("data", &[8], INT64)],
        &[("output", &[1], INT64)],
        &[NodeAttr {
            name: "keepdims",
            value: AttrValue::Int(0),
        }],
        &[Initializer {
            name: "axes",
            dims: vec![1],
            data: vec![0],
        }],
        vec![vec![1, 2, 3, 4, 5, 6, 7, 8]],
    ));

    // ---- ReduceMax ----
    // In opset 17, ReduceMax takes `axes` as an attribute (not an input).
    // [4, 4] axes=[1] keepdims=1 → [4, 1]
    cases.push(exact(
        "ReduceMax",
        0,
        &[("data", &[4, 4], INT64)],
        &[("output", &[4, 1], INT64)],
        &[
            NodeAttr {
                name: "keepdims",
                value: AttrValue::Int(1),
            },
            NodeAttr {
                name: "axes",
                value: AttrValue::Ints(vec![1]),
            },
        ],
        &[],
        vec![vec![5, 3, 8, 1, 2, 9, 4, 7, 6, 10, 11, 12, 0, 1, 2, 15]],
    ));
    // [6] axes=[0] keepdims=0 → scalar [1]
    cases.push(exact(
        "ReduceMax",
        1,
        &[("data", &[6], INT64)],
        &[("output", &[1], INT64)],
        &[
            NodeAttr {
                name: "keepdims",
                value: AttrValue::Int(0),
            },
            NodeAttr {
                name: "axes",
                value: AttrValue::Ints(vec![0]),
            },
        ],
        &[],
        vec![vec![5, 10, 3, 1, 7, 2]],
    ));

    cases.truncate(default_case_count());
    cases
}
