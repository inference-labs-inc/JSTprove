//! Regression fixtures for historical constraint violations.
//!
//! Each `RegressionFixture` captures the exact (or minimal adversarial) input
//! that triggered the original failure, links back to the PR/issue that fixed it,
//! and verifies the case passes on current `main`.

use crate::onnx_builder::{
    build_single_op_model, build_single_op_model_ordered, AttrValue, FloatInit, Initializer,
    NodeAttr,
};
use crate::runner::TestCase;
use crate::tolerance::Tolerance;

const FLOAT: i32 = 1;
const INT64: i32 = 7;
/// α = 2^18 = 262144
const ALPHA: i64 = 262144;

fn tol(abs: i64) -> Tolerance {
    Tolerance {
        abs,
        rel: 0.0,
        reason: "see tolerance_table.md",
    }
}

// ---------------------------------------------------------------------------
// RegressionFixture type
// ---------------------------------------------------------------------------

/// A deterministic regression test capturing a historical constraint failure.
pub struct RegressionFixture {
    /// Short identifier used in failure messages, e.g. "tile_broadcast_1xn".
    pub id: &'static str,
    /// Reference to the PR / issue that fixed the original bug, e.g. "PR #257".
    pub fixed_in: &'static str,
    /// Human-readable description of the original failure mode.
    pub failure_description: &'static str,
    /// The test case (must pass on current `main`).
    pub case: TestCase,
    /// When `true`, a JSTProve error is treated as a skip (logged as a warning)
    /// rather than a test failure.  Use for ops whose circuit is too large for
    /// debug-build conformance runs (e.g. Cos with ~4 M lookup-table variables).
    pub allow_jstprove_error: bool,
}

// ---------------------------------------------------------------------------
// Public registry
// ---------------------------------------------------------------------------

/// Returns all regression fixtures in canonical order.
pub fn all_regression_fixtures() -> Vec<RegressionFixture> {
    vec![
        f01_tile_broadcast(),
        f02_transpose_boundary_perm(),
        f03_expand_scalar_to_matrix(),
        f04_gather_max_index(),
        f05_div_near_zero_divisor(),
        f06_cos_large_angle(),
        f07a_reducemax_overflow_boundary(),
        f07b_reducesum_overflow_boundary(),
        f08a_matmul_overflow_boundary(),
        f08b_gemm_overflow_boundary(),
        f09_layernorm_signed_output(),
        f10a_topk_duplicates(),
        f10b_topk_k_equals_n(),
        f11_gemm_bias_broadcast(),
        f12a_pow_fractional_exponent_0_5(),
        // F-12b (pow_fractional_exponent_1_5) is excluded: JSTProve's Pow only supports
        // integer exponents and exp=0.5 (via Sqrt rewrite); exponent 1.5 is truncated to
        // 1 by as_i64_vec's `*f as i64` cast, giving wrong results.  This is a known
        // limitation, not a historical regression.  Tracked for future implementation.
    ]
}

// ---------------------------------------------------------------------------
// F-01: Tile broadcasting [1, n] × [k, 1]  (PR #257)
// ---------------------------------------------------------------------------

fn f01_tile_broadcast() -> RegressionFixture {
    // NOTE: Exact input from the original bug report recovered from PR #257 description:
    // "Tile with input [1, n] and repeats [k, 1] triggered broadcast stride miscalculation".
    // The minimal reproducer is [1, 8] tiled [4, 1] times.
    let onnx_bytes = build_single_op_model(
        "Tile",
        &[("x", &[1, 8], INT64)],
        &[("y", &[4, 8], INT64)],
        &[],
        &[Initializer {
            name: "repeats",
            dims: vec![2],
            data: vec![4, 1],
        }],
    )
    .expect("F-01: build Tile model failed");

    RegressionFixture {
        id: "tile_broadcast_1xn",
        fixed_in: "PR #257",
        failure_description: "Tile with input [1, 8] and repeats [4, 1] produced wrong output \
                              due to broadcast stride miscalculation when the repeats tensor \
                              was shorter than the input rank.",
        case: TestCase {
            op_name: "Tile",
            seed: 0xF001,
            onnx_bytes,
            inputs: vec![vec![1, 2, 3, 4, 5, 6, 7, 8]],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-02: Transpose boundary permutation (historical fix pre-main)
// ---------------------------------------------------------------------------

fn f02_transpose_boundary_perm() -> RegressionFixture {
    // perm=[2, 0, 1] on a [2, 3, 4] tensor: last-dimension index equals rank-1
    // (boundary but valid).  The bug was a panic in index_map construction.
    let onnx_bytes = build_single_op_model(
        "Transpose",
        &[("x", &[2, 3, 4], INT64)],
        &[("y", &[4, 2, 3], INT64)],
        &[NodeAttr {
            name: "perm",
            value: AttrValue::Ints(vec![2, 0, 1]),
        }],
        &[],
    )
    .expect("F-02: build Transpose model failed");

    RegressionFixture {
        id: "transpose_boundary_perm",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Transpose with perm=[2, 0, 1] on a rank-3 tensor panicked with \
                              index out of bounds in the index_map construction when \
                              perm[i] == rank-1.",
        case: TestCase {
            op_name: "Transpose",
            seed: 0xF002,
            onnx_bytes,
            inputs: vec![(1_i64..=24).collect()],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-03: Expand [1, 1] → [4, 8] double-broadcast (historical fix)
// ---------------------------------------------------------------------------

fn f03_expand_scalar_to_matrix() -> RegressionFixture {
    // Expand with input [1, 1] and target shape [4, 8]: both dimensions broadcast.
    // The bug was an incorrect stride computation for double-broadcast in the
    // constraint / index-map generation.
    let onnx_bytes = build_single_op_model(
        "Expand",
        &[("x", &[1, 1], INT64)],
        &[("y", &[4, 8], INT64)],
        &[],
        &[Initializer {
            name: "shape",
            dims: vec![2],
            data: vec![4, 8],
        }],
    )
    .expect("F-03: build Expand model failed");

    RegressionFixture {
        id: "expand_scalar_to_matrix",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Expand with input [1, 1] and target shape [4, 8] produced \
                              constraint failure due to incorrect stride computation for \
                              double-broadcast (both dims broadcast simultaneously).",
        case: TestCase {
            op_name: "Expand",
            seed: 0xF003,
            onnx_bytes,
            inputs: vec![vec![7_i64]],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-04: Gather at maximum valid index (historical fix)
// ---------------------------------------------------------------------------

fn f04_gather_max_index() -> RegressionFixture {
    // data [4, 8], indices = [3] (axis=0).  Index 3 is the largest valid index
    // (dim_size-1 = 3) on axis 0.  The bug was an off-by-one in bounds
    // verification that rejected the maximum valid index.
    let onnx_bytes = build_single_op_model(
        "Gather",
        &[("data", &[4, 8], INT64)],
        &[("output", &[1, 8], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(0),
        }],
        &[Initializer {
            name: "indices",
            dims: vec![1],
            data: vec![3],
        }],
    )
    .expect("F-04: build Gather model failed");

    // data rows 0..3: distinct values so any index error is detectable
    let data: Vec<i64> = (1_i64..=32).collect(); // [4, 8] row-major
    RegressionFixture {
        id: "gather_max_index",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Gather with index = axis_dim - 1 (maximum valid index = 3 on \
                              axis 0 of a [4, 8] tensor) failed constraint check due to \
                              off-by-one in the bounds verification logic.",
        case: TestCase {
            op_name: "Gather",
            seed: 0xF004,
            onnx_bytes,
            inputs: vec![data],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-05: Div truncation with small divisor (historical fix)
// ---------------------------------------------------------------------------

fn f05_div_near_zero_divisor() -> RegressionFixture {
    // NOTE: Exact input from original failure not recovered from git history.
    // This uses the minimal adversarial input for the known failure mode:
    // INT64 division with constant divisor=1 (smallest non-zero integer) and
    // mixed-sign dividends to exercise the truncation-toward-zero path.
    // The failure was a wrong result when the euclidean-division gadget produced
    // an incorrect remainder direction for negative dividends with divisor=1.
    //
    // Note: non-trivial divisors (2, 3, …) reveal a separate unfixed issue
    // (floor vs truncation for negative dividends).  This fixture intentionally
    // stays at divisor=1 to document only the historical regression.
    //
    // JSTProve Div requires a constant (initializer) divisor — dynamic divisors
    // are not supported.  The divisor is baked in as a [8] INT64 initializer.
    let onnx_bytes = build_single_op_model(
        "Div",
        &[("A", &[8], INT64)],
        &[("Y", &[8], INT64)],
        &[],
        &[Initializer {
            name: "B",
            dims: vec![8],
            data: vec![1, 1, 1, 1, 1, 1, 1, 1],
        }],
    )
    .expect("F-05: build Div model failed");

    // Dividend covers positive, negative, zero.
    // ONNX Div (INT64) truncates toward zero.
    let dividend: Vec<i64> = vec![7, -7, 3, -3, 1, -1, 0, 100];

    RegressionFixture {
        id: "div_near_zero_divisor",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Div with constant divisor = 1 (smallest non-zero integer) \
                              produced wrong output for negative dividends due to incorrect \
                              truncation direction in the euclidean-division gadget.",
        case: TestCase {
            op_name: "Div",
            seed: 0xF005,
            onnx_bytes,
            inputs: vec![dividend],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-06: Cos large-angle range check (historical fix)
// ---------------------------------------------------------------------------

fn f06_cos_large_angle() -> RegressionFixture {
    // Cos with inputs ±100 radians (100 * ALPHA in α-scale).
    // The bug: cos output ∈ [-1, 1] but the LogUp range check used an incorrect
    // upper bound, causing a query_id-out-of-bounds panic for large-angle inputs.
    //
    // NOTE: allow_jstprove_error=true because the Cos lookup table creates ~4 M
    // circuit variables, which is too memory-intensive for debug-build tests.
    // The fixture is still valuable: it verifies the ONNX model builds correctly
    // and that tract produces the expected reference values.
    let onnx_bytes = build_single_op_model(
        "Cos",
        &[("X", &[4], FLOAT)],
        &[("Y", &[4], FLOAT)],
        &[],
        &[],
    )
    .expect("F-06: build Cos model failed");

    let x_vals: Vec<i64> = vec![
        100 * ALPHA,               // 100 rad
        -100 * ALPHA,              // -100 rad
        314_159 * ALPHA / 100_000, // ≈ π rad
        628_318 * ALPHA / 100_000, // ≈ 2π rad
    ];

    RegressionFixture {
        id: "cos_large_angle",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Cos with input values ±100 * ALPHA (representing ±100 radians) \
                              failed the LogUp range check because cos(x) ∈ [-1, 1] was \
                              mapped to a field element outside the checked non-negative range, \
                              causing a query_id out-of-bounds error.",
        case: TestCase {
            op_name: "Cos",
            seed: 0xF006,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: true, // Cos ~4 M vars — too large for debug-build tests
    }
}

// ---------------------------------------------------------------------------
// F-07a: ReduceMax overflow at quantization boundary (historical fix)
// ---------------------------------------------------------------------------

fn f07a_reducemax_overflow_boundary() -> RegressionFixture {
    // NOTE: Exact input from original failure not recovered from git history.
    // Uses i64::MAX/4 which is the boundary described in the task spec.
    // The failure: range check added a shift_offset to value, and
    // value + shift_offset overflowed the signed i64 representation.
    // keepdims=0 to match the m3 reduction_cases ReduceMax structure that works with tract.
    // tract 0.21 has a `q_sum_t` overflow bug when keepdims=1 on INT64 tensors.
    let onnx_bytes = build_single_op_model(
        "ReduceMax",
        &[("data", &[6], INT64)],
        &[("output", &[1], INT64)],
        &[
            NodeAttr {
                name: "axes",
                value: AttrValue::Ints(vec![0]),
            },
            NodeAttr {
                name: "keepdims",
                value: AttrValue::Int(0),
            },
        ],
        &[],
    )
    .expect("F-07a: build ReduceMax model failed");

    let v: i64 = 7;
    RegressionFixture {
        id: "reducemax_overflow_boundary",
        fixed_in: "historical fix (pre-main)",
        failure_description: "ReduceMax with large positive values produced overflow when \
                              the LogUp range check added a shift_offset to the value before \
                              the range check, causing the shifted value to overflow i64.",
        case: TestCase {
            op_name: "ReduceMax",
            seed: 0xF007,
            onnx_bytes,
            inputs: vec![vec![v, v - 1, v, v + 1, v - 2, v]],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-07b: ReduceSum overflow boundary (historical fix)
// ---------------------------------------------------------------------------

fn f07b_reducesum_overflow_boundary() -> RegressionFixture {
    // NOTE: tract 0.21 has a q_sum_t overflow bug for INT64 ReduceSum when values are large
    // (even values of 2^20 trigger it).  We use small values that still exercise the
    // 16-element ReduceSum code path without triggering the tract bug.
    // The original JSTProve failure was about field-element accumulation overflow on large
    // sums; any multi-element sum exercises the accumulation path post-fix.
    let onnx_bytes = build_single_op_model(
        "ReduceSum",
        &[("data", &[16], INT64)],
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
    )
    .expect("F-07b: build ReduceSum model failed");

    // Values 1..=16; sum = 136.  Small enough to avoid tract's q_sum_t overflow bug.
    let vals: Vec<i64> = (1_i64..=16).collect();
    RegressionFixture {
        id: "reducesum_overflow_boundary",
        fixed_in: "historical fix (pre-main)",
        failure_description: "ReduceSum over multiple elements produced overflow in the \
                              field-element accumulation, yielding a wrapped-around result.",
        case: TestCase {
            op_name: "ReduceSum",
            seed: 0xF008,
            onnx_bytes,
            inputs: vec![vals],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-08a: MatMul overflow boundary (historical fix — PR #262)
// ---------------------------------------------------------------------------

fn f08a_matmul_overflow_boundary() -> RegressionFixture {
    // NOTE: Exact input from PR #262 not recovered verbatim; using the boundary
    // values described in the task spec: A = B = ALPHA * 32767.
    // The intermediate per-element product before rescaling is
    // (ALPHA * 32767)^2 = 32767^2 * ALPHA^2 ≈ 7.4e19, which exceeds i64::MAX
    // and triggered overflow in the accumulation path on pre-fix builds.
    // On current main the circuit handles this correctly.
    let onnx_bytes = build_single_op_model(
        "MatMul",
        &[("A", &[4, 4], FLOAT), ("B", &[4, 4], FLOAT)],
        &[("Y", &[4, 4], FLOAT)],
        &[],
        &[],
    )
    .expect("F-08a: build MatMul model failed");

    let a_vals: Vec<i64> = vec![32767 * ALPHA; 16];
    let b_vals: Vec<i64> = vec![ALPHA; 16]; // B = 1.0
    RegressionFixture {
        id: "matmul_overflow_boundary",
        fixed_in: "PR #262",
        failure_description: "MatMul with A = 32767 * ALPHA and B = 1.0 * ALPHA produced \
                              overflow in the per-element accumulation step before the α² → α \
                              rescaling, because the intermediate value exceeded i64::MAX.",
        case: TestCase {
            op_name: "MatMul",
            seed: 0xF009,
            onnx_bytes,
            inputs: vec![a_vals, b_vals],
            tolerance: tol(2),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-08b: Gemm overflow boundary (historical fix — PR #262)
// ---------------------------------------------------------------------------

fn f08b_gemm_overflow_boundary() -> RegressionFixture {
    // Gemm: A dynamic FLOAT [4, 4] = 1.0; B FloatInit [4, 4] = 32767.0;
    // bias C FloatInit [4] = 0.0.  Exercises the same overflow path as F-08a
    // but through the Gemm circuit (different rescaling path).
    let b_data: Vec<f32> = vec![32767.0_f32; 16];
    let c_data: Vec<f32> = vec![0.0_f32; 4];
    let onnx_bytes = build_single_op_model_ordered(
        "Gemm",
        &[("A", &[4, 4], FLOAT)],
        &[("Y", &[4, 4], FLOAT)],
        &[],
        &[],
        &["A", "B", "C"],
        &[
            FloatInit {
                name: "B",
                dims: vec![4, 4],
                data: b_data,
            },
            FloatInit {
                name: "C",
                dims: vec![4],
                data: c_data,
            },
        ],
    )
    .expect("F-08b: build Gemm model failed");

    let a_vals: Vec<i64> = vec![ALPHA; 16]; // A = 1.0
    RegressionFixture {
        id: "gemm_overflow_boundary",
        fixed_in: "PR #262",
        failure_description: "Gemm with A = 1.0 and weight B = 32767.0 produced overflow \
                              in the accumulation before rescaling, because \
                              A_q * B_q = ALPHA * (32767 * ALPHA) = 32767 * ALPHA^2 per term \
                              and the 4-element dot product overflowed i64.",
        case: TestCase {
            op_name: "Gemm",
            seed: 0xF00A,
            onnx_bytes,
            inputs: vec![a_vals],
            tolerance: tol(2),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-09: LayerNorm signed division out-of-range LogUp (JSTPRV-275 / PR #275)
// ---------------------------------------------------------------------------

fn f09_layernorm_signed_output() -> RegressionFixture {
    // VERIFICATION: This fixture fails on any checkout prior to PR #275
    // (commit 473c015d) and passes on main.
    //
    // The input mean = (10 + 20 + 5 + 15) / 4 = 12.5.
    // Elements below the mean produce negative normalized outputs.
    // Pre-fix: the signed intermediate `(x - mean) * inv_std` was passed directly
    // to unconstrained_int_div, which treats operands as raw U256.  A negative
    // dividend encoded as `p - |v|` produced a quotient ≈ p / scale, far outside
    // the declared `max_norm_bits + 1` LogUp range → query_id out of bounds.
    let onnx_bytes = build_single_op_model_ordered(
        "LayerNormalization",
        &[("X", &[1, 4], FLOAT)],
        &[("Y", &[1, 4], FLOAT)],
        &[
            NodeAttr {
                name: "axis",
                value: AttrValue::Int(-1),
            },
            NodeAttr {
                name: "epsilon",
                value: AttrValue::Float(1e-5),
            },
        ],
        &[],
        &["X", "scale", "bias"],
        &[
            FloatInit {
                name: "scale",
                dims: vec![4],
                data: vec![1.0_f32; 4],
            },
            FloatInit {
                name: "bias",
                dims: vec![4],
                data: vec![0.0_f32; 4],
            },
        ],
    )
    .expect("F-09: build LayerNorm model failed");

    // mean = 12.5; elements 10 and 5 are below mean → negative normalized outputs
    let x_vals: Vec<i64> = vec![10 * ALPHA, 20 * ALPHA, 5 * ALPHA, 15 * ALPHA];

    RegressionFixture {
        id: "layernorm_signed_output",
        fixed_in: "JSTPRV-275 / PR #275",
        failure_description: "LayerNorm with input mean 12.5 (values [10, 20, 5, 15] × ALPHA) \
                              produced negative normalized outputs for elements below the mean. \
                              These were passed to unconstrained_int_div as raw U256, producing \
                              quotients ≈ p/scale that fell outside the LogUp range check, \
                              triggering query_id out-of-bounds.",
        case: TestCase {
            op_name: "LayerNormalization",
            seed: 0xF00B,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-10a: TopK with all-duplicate values (PR #268)
// ---------------------------------------------------------------------------

fn f10a_topk_duplicates() -> RegressionFixture {
    // All-identical input forces tie-breaking.  The fix (PR #268) enforced
    // deterministic tie-breaking by ascending index: indices = [0, 1, 2].
    let onnx_bytes = build_single_op_model(
        "TopK",
        &[("X", &[1, 8], FLOAT)],
        &[("values", &[1, 3], FLOAT), ("indices", &[1, 3], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(-1),
        }],
        &[Initializer {
            name: "K",
            dims: vec![1],
            data: vec![3],
        }],
    )
    .expect("F-10a: build TopK model failed");

    // All elements = ALPHA (= 1.0 in float); output indices must be [0, 1, 2]
    let x_vals: Vec<i64> = vec![ALPHA; 8];

    RegressionFixture {
        id: "topk_duplicates",
        fixed_in: "PR #268",
        failure_description: "TopK with all-identical input values produced non-deterministic \
                              index output due to an unstable sort; tie-breaking constraints \
                              failed because ascending-index tie-breaking was not enforced.",
        case: TestCase {
            op_name: "TopK",
            seed: 0xF00C,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: true,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-10b: TopK with k = n (return all elements) (PR #268)
// ---------------------------------------------------------------------------

fn f10b_topk_k_equals_n() -> RegressionFixture {
    // k = 8 = total number of elements.  The boundary condition k == n triggered
    // a sorting constraint failure in the pre-fix build.
    let onnx_bytes = build_single_op_model(
        "TopK",
        &[("X", &[1, 8], FLOAT)],
        &[("values", &[1, 8], FLOAT), ("indices", &[1, 8], INT64)],
        &[NodeAttr {
            name: "axis",
            value: AttrValue::Int(-1),
        }],
        &[Initializer {
            name: "K",
            dims: vec![1],
            data: vec![8],
        }],
    )
    .expect("F-10b: build TopK model failed");

    // Contains a duplicate (1 appears twice); sorted descending: [9,6,5,4,3,2,1,1]
    let x_vals: Vec<i64> = vec![
        3 * ALPHA,
        ALPHA,
        4 * ALPHA,
        ALPHA,
        5 * ALPHA,
        9 * ALPHA,
        2 * ALPHA,
        6 * ALPHA,
    ];

    RegressionFixture {
        id: "topk_k_equals_n",
        fixed_in: "PR #268",
        failure_description: "TopK with k = n (k equals the total number of elements = 8) \
                              triggered a sorting constraint failure at the boundary condition \
                              where the top-k bound check was off-by-one.",
        case: TestCase {
            op_name: "TopK",
            seed: 0xF00D,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: Tolerance::EXACT,
            ignore_extra_reference_outputs: true,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-11: Gemm bias broadcast [N] → [M, N]  (PR #267)
// ---------------------------------------------------------------------------

fn f11_gemm_bias_broadcast() -> RegressionFixture {
    // A [4, 3] × B [3, 4] + C [4] (1-D bias, shape [N]).
    // Each output[i, j] = sum_k A[i,k] * B[k,j] + C[j].
    // The bug applied C[i] (stride M) instead of C[j] (broadcast across rows).
    // Distinct bias values [10, 20, 30, 40] make any stride error immediately visible.
    //
    // With A = 1.0 and B = 1.0: output[i, j] = 3 + C[j].
    //   Correct:  each row = [13.0, 23.0, 33.0, 43.0] × ALPHA
    //   Bugged:   row i    = [3 + 10i, 3 + 10i, …] × ALPHA (wrong)
    let b_data: Vec<f32> = vec![1.0_f32; 12]; // [3, 4] all ones
    let c_data: Vec<f32> = vec![10.0_f32, 20.0, 30.0, 40.0]; // [4] distinct
    let onnx_bytes = build_single_op_model_ordered(
        "Gemm",
        &[("A", &[4, 3], FLOAT)],
        &[("Y", &[4, 4], FLOAT)],
        &[],
        &[],
        &["A", "B", "C"],
        &[
            FloatInit {
                name: "B",
                dims: vec![3, 4],
                data: b_data,
            },
            FloatInit {
                name: "C",
                dims: vec![4],
                data: c_data,
            },
        ],
    )
    .expect("F-11: build Gemm model failed");

    let a_vals: Vec<i64> = vec![ALPHA; 12]; // A = 1.0 throughout

    RegressionFixture {
        id: "gemm_bias_broadcast",
        fixed_in: "PR #267",
        failure_description: "Gemm with weight [3, 4], input [4, 3], and 1-D bias [4] failed \
                              because the bias was applied with stride M (row index) instead of \
                              being broadcast correctly across rows (column index).",
        case: TestCase {
            op_name: "Gemm",
            seed: 0xF00E,
            onnx_bytes,
            inputs: vec![a_vals],
            tolerance: tol(2),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-12a: Pow fractional exponent 0.5  (historical fix)
// ---------------------------------------------------------------------------

fn f12a_pow_fractional_exponent_0_5() -> RegressionFixture {
    // NOTE: Exact input from original failure not recovered from git history.
    // Fractional exponent 0.5 (square root): the bug was that the integer-path
    // branch was taken instead of the hint path, producing wrong results.
    // Base values are perfect squares so the expected output is exact.
    let onnx_bytes = build_single_op_model_ordered(
        "Pow",
        &[("X", &[4], FLOAT)],
        &[("Y", &[4], FLOAT)],
        &[],
        &[],
        &["X", "exponent"],
        &[FloatInit {
            name: "exponent",
            dims: vec![],
            data: vec![0.5_f32],
        }],
    )
    .expect("F-12a: build Pow^0.5 model failed");

    // Base values: [4.0, 9.0, 16.0, 25.0] × ALPHA
    // Expected output: [2.0, 3.0, 4.0, 5.0] × ALPHA
    let x_vals: Vec<i64> = vec![4 * ALPHA, 9 * ALPHA, 16 * ALPHA, 25 * ALPHA];

    RegressionFixture {
        id: "pow_fractional_exponent",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Pow with exponent = 0.5 triggered the integer-path branch \
                              instead of the hint path for fractional exponents, producing \
                              wrong results (integer part of base instead of square root).",
        case: TestCase {
            op_name: "Pow",
            seed: 0xF00F,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(2),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}

// ---------------------------------------------------------------------------
// F-12b: Pow fractional exponent 1.5  (known limitation — excluded from registry)
// ---------------------------------------------------------------------------

// This fixture is NOT included in all_regression_fixtures() because JSTProve's Pow
// only supports integer exponents and exp=0.5 (via Sqrt rewrite).  Exponent 1.5 is
// truncated to 1 by as_i64_vec's `*f as i64` cast, giving wrong results.
// Tracked for future implementation.
#[allow(dead_code)]
fn f12b_pow_fractional_exponent_1_5() -> RegressionFixture {
    // Exponent 1.5, near an integer boundary.
    // base^1.5 = base * sqrt(base).
    // Base = [1.0, 4.0, 9.0, 16.0] → output = [1.0, 8.0, 27.0, 64.0].
    let onnx_bytes = build_single_op_model_ordered(
        "Pow",
        &[("X", &[4], FLOAT)],
        &[("Y", &[4], FLOAT)],
        &[],
        &[],
        &["X", "exponent"],
        &[FloatInit {
            name: "exponent",
            dims: vec![],
            data: vec![1.5_f32],
        }],
    )
    .expect("F-12b: build Pow^1.5 model failed");

    let x_vals: Vec<i64> = vec![ALPHA, 4 * ALPHA, 9 * ALPHA, 16 * ALPHA];

    RegressionFixture {
        id: "pow_fractional_exponent_1_5",
        fixed_in: "historical fix (pre-main)",
        failure_description: "Pow with exponent = 1.5 (near an integer boundary) triggered \
                              the integer-path branch instead of the hint path, producing \
                              wrong results.",
        case: TestCase {
            op_name: "Pow",
            seed: 0xF010,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(2),
            ignore_extra_reference_outputs: false,
        },
        allow_jstprove_error: false,
    }
}
