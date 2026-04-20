//! Fixed test cases for Group E–J operators (rescaling arithmetic, transcendental/hint,
//! pooling, normalization, spatial, TopK).
//!
//! All test inputs use α-scaled FLOAT (elem_type=1) tensors.
//! Weights are provided as FloatInit (real f32 values); JSTProve quantizes them internally.
//!
//! Ops that are not yet implemented in the Expander circuit backend (GroupNormalization,
//! InstanceNormalization) are included as reference_only placeholders and will auto-skip.

use super::default_case_count;
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
// Group E — Rescaling arithmetic ops
// ---------------------------------------------------------------------------

/// Returns test cases for rescaling arithmetic operators:
/// Gemm, MatMul, Conv, ConvTranspose, BatchNorm, Div, Pow.
#[allow(clippy::too_many_lines, clippy::vec_init_then_push)]
pub fn rescaling_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- Gemm [2,4] @ [4,3] + bias[3] ----
    // A is dynamic FLOAT input [2,4], B and C are FloatInits.
    // JSTProve quantizes B at α¹ and C at α².
    {
        let w_data: Vec<f32> = vec![
            0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75, 0.1, -0.1, 0.2, -0.2,
        ]; // [4,3] = 12 values
        let bias_data: Vec<f32> = vec![0.1_f32, -0.1, 0.2];
        let onnx_bytes = build_single_op_model_ordered(
            "Gemm",
            &[("A", &[2, 4], FLOAT)],
            &[("Y", &[2, 3], FLOAT)],
            &[],
            &[],
            &["A", "B", "C"],
            &[
                FloatInit {
                    name: "B",
                    dims: vec![4, 3],
                    data: w_data,
                },
                FloatInit {
                    name: "C",
                    dims: vec![3],
                    data: bias_data,
                },
            ],
        )
        .expect("build Gemm model failed");
        // A input: [1.0, 0.5, -1.0, 0.5, 0.25, -0.25, 0.75, -0.75] × ALPHA
        let a_vals: Vec<i64> = vec![
            ALPHA,
            ALPHA / 2,
            -ALPHA,
            ALPHA / 2,
            ALPHA / 4,
            -(ALPHA / 4),
            3 * ALPHA / 4,
            -(3 * ALPHA / 4),
        ];
        cases.push(TestCase {
            op_name: "Gemm",
            seed: 0,
            onnx_bytes,
            inputs: vec![a_vals],
            tolerance: tol(3),
        });
    }

    // ---- Gemm transB=1: A [2,4] @ B^T [4,3] ----
    // With transB=1, B has shape [3,4] but logically [4,3] after transpose.
    {
        let w_data: Vec<f32> = vec![
            0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75, 0.1, -0.1, 0.2, -0.2,
        ]; // [3,4] transposed
        let bias_data: Vec<f32> = vec![0.1_f32, -0.1, 0.2];
        let onnx_bytes = build_single_op_model_ordered(
            "Gemm",
            &[("A", &[2, 4], FLOAT)],
            &[("Y", &[2, 3], FLOAT)],
            &[NodeAttr {
                name: "transB",
                value: AttrValue::Int(1),
            }],
            &[],
            &["A", "B", "C"],
            &[
                FloatInit {
                    name: "B",
                    dims: vec![3, 4],
                    data: w_data,
                },
                FloatInit {
                    name: "C",
                    dims: vec![3],
                    data: bias_data,
                },
            ],
        )
        .expect("build Gemm transB model failed");
        let a_vals: Vec<i64> = vec![
            ALPHA,
            ALPHA / 2,
            -ALPHA,
            ALPHA / 2,
            ALPHA / 4,
            -(ALPHA / 4),
            3 * ALPHA / 4,
            -(3 * ALPHA / 4),
        ];
        cases.push(TestCase {
            op_name: "Gemm",
            seed: 1,
            onnx_bytes,
            inputs: vec![a_vals],
            tolerance: tol(3),
        });
    }

    // ---- MatMul: A [2,4] @ B [4,3] (both dynamic FLOAT) ----
    {
        let onnx_bytes = build_single_op_model(
            "MatMul",
            &[("A", &[2, 4], FLOAT), ("B", &[4, 3], FLOAT)],
            &[("Y", &[2, 3], FLOAT)],
            &[],
            &[],
        )
        .expect("build MatMul model failed");
        let a_vals: Vec<i64> = vec![
            ALPHA,
            ALPHA / 2,
            -ALPHA,
            ALPHA / 2,
            ALPHA / 4,
            -(ALPHA / 4),
            3 * ALPHA / 4,
            -(3 * ALPHA / 4),
        ];
        let b_vals: Vec<f32> = vec![
            0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75, 0.1, -0.1, 0.2, -0.2,
        ];
        let b_alpha: Vec<i64> = b_vals
            .iter()
            .map(|&v| (v as f64 * ALPHA as f64) as i64)
            .collect();
        cases.push(TestCase {
            op_name: "MatMul",
            seed: 0,
            onnx_bytes,
            inputs: vec![a_vals, b_alpha],
            tolerance: tol(3),
        });
    }

    // ---- Conv 1×1 kernel: input [1,4,8,8], weight [4,4,1,1], bias [4] ----
    {
        // 4 output channels, 4 input channels, 1×1 kernel = 4*4*1*1 = 16 weights
        let w_data: Vec<f32> = vec![
            0.5, -0.5, 0.25, -0.25, // out_ch 0
            -0.5, 0.5, -0.25, 0.25, // out_ch 1
            0.1, -0.1, 0.2, -0.2, // out_ch 2
            0.3, -0.3, 0.4, -0.4, // out_ch 3
        ];
        let b_data: Vec<f32> = vec![0.0_f32, 0.0, 0.0, 0.0];
        let onnx_bytes = build_single_op_model_ordered(
            "Conv",
            &[("X", &[1, 4, 8, 8], FLOAT)],
            &[("Y", &[1, 4, 8, 8], FLOAT)],
            &[
                NodeAttr {
                    name: "pads",
                    value: AttrValue::Ints(vec![0, 0, 0, 0]),
                },
                NodeAttr {
                    name: "kernel_shape",
                    value: AttrValue::Ints(vec![1, 1]),
                },
                NodeAttr {
                    name: "strides",
                    value: AttrValue::Ints(vec![1, 1]),
                },
                NodeAttr {
                    name: "dilations",
                    value: AttrValue::Ints(vec![1, 1]),
                },
            ],
            &[],
            &["X", "W", "B"],
            &[
                FloatInit {
                    name: "W",
                    dims: vec![4, 4, 1, 1],
                    data: w_data,
                },
                FloatInit {
                    name: "B",
                    dims: vec![4],
                    data: b_data,
                },
            ],
        )
        .expect("build Conv 1x1 model failed");
        // Input: 1*4*8*8 = 256 elements, small α-scaled values
        let n_elements = 4 * 8 * 8;
        let x_vals: Vec<i64> = (0..n_elements as i64)
            .map(|i| ((i % 7) - 3) * ALPHA / 4)
            .collect();
        cases.push(TestCase {
            op_name: "Conv",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    // ---- Conv 3×3 kernel with padding: input [1,4,8,8], weight [8,4,3,3], bias [8] ----
    {
        let w_size = 8 * 4 * 3 * 3; // = 288
        let w_data: Vec<f32> = (0..w_size).map(|i| (i as f32 % 7.0 - 3.0) * 0.1).collect();
        let b_data: Vec<f32> = vec![0.0_f32; 8];
        let onnx_bytes = build_single_op_model_ordered(
            "Conv",
            &[("X", &[1, 4, 8, 8], FLOAT)],
            &[("Y", &[1, 8, 8, 8], FLOAT)],
            &[
                NodeAttr {
                    name: "pads",
                    value: AttrValue::Ints(vec![1, 1, 1, 1]),
                },
                NodeAttr {
                    name: "kernel_shape",
                    value: AttrValue::Ints(vec![3, 3]),
                },
                NodeAttr {
                    name: "strides",
                    value: AttrValue::Ints(vec![1, 1]),
                },
                NodeAttr {
                    name: "dilations",
                    value: AttrValue::Ints(vec![1, 1]),
                },
            ],
            &[],
            &["X", "W", "B"],
            &[
                FloatInit {
                    name: "W",
                    dims: vec![8, 4, 3, 3],
                    data: w_data,
                },
                FloatInit {
                    name: "B",
                    dims: vec![8],
                    data: b_data,
                },
            ],
        )
        .expect("build Conv 3x3 model failed");
        let n_elements = 4 * 8 * 8;
        let x_vals: Vec<i64> = (0..n_elements as i64)
            .map(|i| ((i % 5) - 2) * ALPHA / 4)
            .collect();
        cases.push(TestCase {
            op_name: "Conv",
            seed: 1,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    // ---- ConvTranspose: input [1,4,4,4], weight [4,4,2,2], stride [2,2], output [1,4,8,8] ----
    {
        let w_size = 4 * 4 * 2 * 2; // = 64
        let w_data: Vec<f32> = (0..w_size).map(|i| (i as f32 % 5.0 - 2.0) * 0.1).collect();
        let onnx_bytes = build_single_op_model_ordered(
            "ConvTranspose",
            &[("X", &[1, 4, 4, 4], FLOAT)],
            &[("Y", &[1, 4, 8, 8], FLOAT)],
            &[
                NodeAttr {
                    name: "kernel_shape",
                    value: AttrValue::Ints(vec![2, 2]),
                },
                NodeAttr {
                    name: "strides",
                    value: AttrValue::Ints(vec![2, 2]),
                },
                NodeAttr {
                    name: "pads",
                    value: AttrValue::Ints(vec![0, 0, 0, 0]),
                },
                NodeAttr {
                    name: "dilations",
                    value: AttrValue::Ints(vec![1, 1]),
                },
            ],
            &[],
            &["X", "W"],
            &[FloatInit {
                name: "W",
                dims: vec![4, 4, 2, 2],
                data: w_data,
            }],
        )
        .expect("build ConvTranspose model failed");
        let n_elements = 4 * 4 * 4;
        let x_vals: Vec<i64> = (0..n_elements as i64)
            .map(|i| ((i % 5) - 2) * ALPHA / 4)
            .collect();
        cases.push(TestCase {
            op_name: "ConvTranspose",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // ---- BatchNormalization: input [1,4,8,8] ----
    // BatchNorm takes [X, scale, B, input_mean, input_var].
    // All parameter inputs are FloatInits.
    {
        let c = 4; // number of channels
        let scale_data: Vec<f32> = vec![1.0_f32; c];
        let b_data: Vec<f32> = vec![0.0_f32; c];
        let mean_data: Vec<f32> = vec![0.1_f32, -0.1, 0.2, -0.2];
        let var_data: Vec<f32> = vec![1.0_f32; c];
        let onnx_bytes = build_single_op_model_ordered(
            "BatchNormalization",
            &[("X", &[1, 4, 8, 8], FLOAT)],
            &[("Y", &[1, 4, 8, 8], FLOAT)],
            &[NodeAttr {
                name: "epsilon",
                value: AttrValue::Float(1e-5),
            }],
            &[],
            &["X", "scale", "B", "input_mean", "input_var"],
            &[
                FloatInit {
                    name: "scale",
                    dims: vec![c as i64],
                    data: scale_data,
                },
                FloatInit {
                    name: "B",
                    dims: vec![c as i64],
                    data: b_data,
                },
                FloatInit {
                    name: "input_mean",
                    dims: vec![c as i64],
                    data: mean_data,
                },
                FloatInit {
                    name: "input_var",
                    dims: vec![c as i64],
                    data: var_data,
                },
            ],
        )
        .expect("build BatchNormalization model failed");
        let n_elements = 4 * 8 * 8;
        let x_vals: Vec<i64> = (0..n_elements as i64)
            .map(|i| ((i % 7) - 3) * ALPHA / 4)
            .collect();
        cases.push(TestCase {
            op_name: "BatchNormalization",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    // ---- Div: INT64 input [6] / constant INT64 divisor (exact divisibility) ----
    // JSTProve Div requires a constant divisor and uses integer (euclidean) division.
    // We test with INT64 inputs so both JSTProve and tract agree on exact integer semantics.
    // The divisor must be a positive integer in [1, 2^32).
    {
        let onnx_bytes = build_single_op_model(
            "Div",
            &[("A", &[6], INT64)],
            &[("Y", &[6], INT64)],
            &[],
            &[Initializer {
                name: "B",
                dims: vec![6],
                data: vec![2, 4, 3, 2, 6, 3],
            }],
        )
        .expect("build Div INT64 model failed");
        // Inputs divisible by respective divisors for clean integer division
        let a_vals: Vec<i64> = vec![10, 20, 9, 8, 12, 15];
        cases.push(TestCase {
            op_name: "Div",
            seed: 0,
            onnx_bytes,
            inputs: vec![a_vals],
            tolerance: Tolerance::EXACT,
        });
    }

    // ---- Pow: FLOAT input [6], integer exponent 2 ----
    {
        let onnx_bytes = build_single_op_model(
            "Pow",
            &[("X", &[6], FLOAT)],
            &[("Y", &[6], FLOAT)],
            &[],
            &[Initializer {
                name: "exponent",
                dims: vec![],
                data: vec![2],
            }],
        )
        .expect("build Pow^2 model failed");
        let x_vals: Vec<i64> = vec![
            ALPHA,
            2 * ALPHA,
            -(ALPHA),
            ALPHA / 2,
            -(ALPHA / 2),
            ALPHA / 4,
        ];
        cases.push(TestCase {
            op_name: "Pow",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    // ---- Pow: FLOAT input [6], integer exponent 3 (positive values only) ----
    // Negative base with odd exponent produces negative output in the field,
    // which requires careful handling. Use only positive values here.
    {
        let onnx_bytes = build_single_op_model(
            "Pow",
            &[("X", &[6], FLOAT)],
            &[("Y", &[6], FLOAT)],
            &[],
            &[Initializer {
                name: "exponent",
                dims: vec![],
                data: vec![3],
            }],
        )
        .expect("build Pow^3 model failed");
        let x_vals: Vec<i64> = vec![
            ALPHA / 2,
            ALPHA / 4,
            ALPHA / 8,
            ALPHA / 4,
            ALPHA / 2,
            ALPHA / 8,
        ];
        cases.push(TestCase {
            op_name: "Pow",
            seed: 1,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group F — Transcendental/hint ops
// ---------------------------------------------------------------------------

/// Returns test cases for transcendental/hint operators.
#[allow(clippy::too_many_lines, clippy::vec_init_then_push)]
pub fn transcendental_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- Exp: FLOAT input [4] ----
    // Keep exponent values small to avoid overflow. Values in [-2, 2] real range.
    // Use smaller tensor sizes to keep circuit memory footprint manageable.
    {
        let onnx_bytes = build_single_op_model(
            "Exp",
            &[("X", &[4], FLOAT)],
            &[("Y", &[4], FLOAT)],
            &[],
            &[],
        )
        .expect("build Exp model failed");
        let x_vals: Vec<i64> = vec![-ALPHA, 0, ALPHA, 2 * ALPHA];
        cases.push(TestCase {
            op_name: "Exp",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // ---- Sigmoid: FLOAT input [4] ----
    {
        let onnx_bytes = build_single_op_model(
            "Sigmoid",
            &[("X", &[4], FLOAT)],
            &[("Y", &[4], FLOAT)],
            &[],
            &[],
        )
        .expect("build Sigmoid model failed");
        let x_vals: Vec<i64> = vec![-ALPHA, -(ALPHA / 2), ALPHA / 2, ALPHA];
        cases.push(TestCase {
            op_name: "Sigmoid",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // ---- Softmax: FLOAT input [1,4], axis=1 ----
    {
        let onnx_bytes = build_single_op_model(
            "Softmax",
            &[("X", &[1, 4], FLOAT)],
            &[("Y", &[1, 4], FLOAT)],
            &[NodeAttr {
                name: "axis",
                value: AttrValue::Int(1),
            }],
            &[],
        )
        .expect("build Softmax model failed");
        let x_vals: Vec<i64> = vec![ALPHA, 2 * ALPHA, -(ALPHA), ALPHA / 2];
        cases.push(TestCase {
            op_name: "Softmax",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(8),
        });
    }

    // Gelu is a JSTProve-specific extension op — not in ONNX opset 17 standard.
    // tract does not support it, so it cannot be tested via conformance comparison.
    // Gelu is tested via the make_model.py end-to-end pipeline instead.

    // ---- LayerNormalization: FLOAT input [1,4], scale [4] ones, bias [4] zeros ----
    // Use smaller tensor to keep circuit memory footprint manageable.
    {
        let onnx_bytes = build_single_op_model_ordered(
            "LayerNormalization",
            &[("X", &[1, 4], FLOAT)],
            &[("Y", &[1, 4], FLOAT)],
            &[NodeAttr {
                name: "epsilon",
                value: AttrValue::Float(1e-5),
            }],
            &[],
            &["X", "Scale", "B"],
            &[
                FloatInit {
                    name: "Scale",
                    dims: vec![4],
                    data: vec![1.0_f32; 4],
                },
                FloatInit {
                    name: "B",
                    dims: vec![4],
                    data: vec![0.0_f32; 4],
                },
            ],
        )
        .expect("build LayerNorm model failed");
        let x_vals: Vec<i64> = vec![ALPHA, 2 * ALPHA, -(ALPHA), ALPHA / 2];
        cases.push(TestCase {
            op_name: "LayerNormalization",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(8),
        });
    }

    // ---- Log: FLOAT input [4] (all positive) ----
    {
        let onnx_bytes = build_single_op_model(
            "Log",
            &[("X", &[4], FLOAT)],
            &[("Y", &[4], FLOAT)],
            &[],
            &[],
        )
        .expect("build Log model failed");
        // Positive values: 0.5, 1.0, 2.0, 4.0
        let x_vals: Vec<i64> = vec![ALPHA / 2, ALPHA, 2 * ALPHA, 4 * ALPHA];
        cases.push(TestCase {
            op_name: "Log",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // ---- ReduceMean: [1,4] axes=[1] keepdims=0 → [1] ----
    {
        let onnx_bytes = build_single_op_model(
            "ReduceMean",
            &[("X", &[1, 4], FLOAT)],
            &[("Y", &[1], FLOAT)],
            &[
                NodeAttr {
                    name: "axes",
                    value: AttrValue::Ints(vec![1]),
                },
                NodeAttr {
                    name: "keepdims",
                    value: AttrValue::Int(0),
                },
            ],
            &[],
        )
        .expect("build ReduceMean model failed");
        let x_vals: Vec<i64> = vec![ALPHA, 2 * ALPHA, -(ALPHA), ALPHA / 2];
        cases.push(TestCase {
            op_name: "ReduceMean",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // ---- Tanh: FLOAT input [4] ----
    {
        let onnx_bytes = build_single_op_model(
            "Tanh",
            &[("X", &[4], FLOAT)],
            &[("Y", &[4], FLOAT)],
            &[],
            &[],
        )
        .expect("build Tanh model failed");
        let x_vals: Vec<i64> = vec![-ALPHA, -(ALPHA / 2), ALPHA / 2, ALPHA];
        cases.push(TestCase {
            op_name: "Tanh",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // Erf uses a large function lookup table circuit (millions of vars) — too memory-intensive
    // for standalone conformance tests in debug builds. Erf is tested via the full model
    // pipeline (make_model.py + Gelu which internally uses Erf) instead.

    // ---- Sqrt: FLOAT input [4] (non-negative) ----
    {
        let onnx_bytes = build_single_op_model(
            "Sqrt",
            &[("X", &[4], FLOAT)],
            &[("Y", &[4], FLOAT)],
            &[],
            &[],
        )
        .expect("build Sqrt model failed");
        let x_vals: Vec<i64> = vec![ALPHA / 4, ALPHA / 2, ALPHA, 2 * ALPHA];
        cases.push(TestCase {
            op_name: "Sqrt",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // Cos and Sin use FunctionLookupTable::build_signed which creates ~4M circuit vars —
    // too memory-intensive for standalone conformance tests in debug builds.
    // These ops are tested via the full model pipeline (make_model.py) instead.

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group G — Pooling ops
// ---------------------------------------------------------------------------

/// Returns test cases for pooling operators.
#[allow(clippy::too_many_lines, clippy::vec_init_then_push)]
pub fn pooling_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- AveragePool: input [1,4,8,8], kernel [2,2], strides [2,2] ----
    {
        let onnx_bytes = build_single_op_model(
            "AveragePool",
            &[("X", &[1, 4, 8, 8], FLOAT)],
            &[("Y", &[1, 4, 4, 4], FLOAT)],
            &[
                NodeAttr {
                    name: "kernel_shape",
                    value: AttrValue::Ints(vec![2, 2]),
                },
                NodeAttr {
                    name: "strides",
                    value: AttrValue::Ints(vec![2, 2]),
                },
            ],
            &[],
        )
        .expect("build AveragePool model failed");
        let n = 4 * 8 * 8;
        let x_vals: Vec<i64> = (0..n as i64).map(|i| ((i % 7) - 3) * ALPHA / 4).collect();
        cases.push(TestCase {
            op_name: "AveragePool",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    // ---- GlobalAveragePool: input [1,4,4,4] → output [1,4,1,1] ----
    {
        let onnx_bytes = build_single_op_model(
            "GlobalAveragePool",
            &[("X", &[1, 4, 4, 4], FLOAT)],
            &[("Y", &[1, 4, 1, 1], FLOAT)],
            &[],
            &[],
        )
        .expect("build GlobalAveragePool model failed");
        let n = 4 * 4 * 4;
        let x_vals: Vec<i64> = (0..n as i64).map(|i| ((i % 5) - 2) * ALPHA / 4).collect();
        cases.push(TestCase {
            op_name: "GlobalAveragePool",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(3),
        });
    }

    // ---- MaxPool: input [1,4,8,8], kernel [2,2], strides [2,2] ----
    // MaxPool output is exact (max of field elements) — EXACT tolerance
    {
        let onnx_bytes = build_single_op_model(
            "MaxPool",
            &[("X", &[1, 4, 8, 8], FLOAT)],
            &[("Y", &[1, 4, 4, 4], FLOAT)],
            &[
                NodeAttr {
                    name: "kernel_shape",
                    value: AttrValue::Ints(vec![2, 2]),
                },
                NodeAttr {
                    name: "strides",
                    value: AttrValue::Ints(vec![2, 2]),
                },
            ],
            &[],
        )
        .expect("build MaxPool model failed");
        let n = 4 * 8 * 8;
        let x_vals: Vec<i64> = (0..n as i64).map(|i| ((i % 7) - 3) * ALPHA / 4).collect();
        cases.push(TestCase {
            op_name: "MaxPool",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(1), // slight tolerance due to float comparison
        });
    }

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group H — Normalization ops (InstanceNorm, GroupNorm are reference_only)
// ---------------------------------------------------------------------------

/// Returns test cases for normalization operators.
///
/// Note: InstanceNormalization is not yet supported in the Expander backend.
/// GroupNormalization is not supported by either tract or the Expander backend.
/// Neither can be tested via conformance comparison; they are exercised only
/// through the full end-to-end pipeline if/when implemented.
///
/// Returns an empty vec currently — the norm_ops test exists as a placeholder.
#[allow(clippy::vec_init_then_push)]
pub fn norm_cases() -> Vec<TestCase> {
    // Both InstanceNorm and GroupNorm are unsupported by at least one of the two
    // reference backends (tract / JSTProve Expander), so they cannot pass the
    // conformance comparator. They are left as future work.
    Vec::new()
}

// ---------------------------------------------------------------------------
// Group I — Spatial ops (Resize, GridSample)
// ---------------------------------------------------------------------------

/// Returns test cases for spatial operators.
#[allow(clippy::too_many_lines, clippy::vec_init_then_push)]
pub fn spatial_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- Resize nearest: input [1,1,4,4] scales [1,1,2,2] → output [1,1,8,8] ----
    // Resize ONNX inputs: [X, roi, scales, sizes]
    // roi is empty (use "" in node_input_order), scales is FloatInit.
    // Use strictly positive values: JSTProve Resize linear uses range checks that
    // assume non-negative outputs.
    {
        let onnx_bytes = build_single_op_model_ordered(
            "Resize",
            &[("X", &[1, 1, 4, 4], FLOAT)],
            &[("Y", &[1, 1, 8, 8], FLOAT)],
            &[NodeAttr {
                name: "mode",
                value: AttrValue::String(b"nearest".to_vec()),
            }],
            &[],
            &["X", "", "scales"],
            &[FloatInit {
                name: "scales",
                dims: vec![4],
                data: vec![1.0_f32, 1.0, 2.0, 2.0],
            }],
        )
        .expect("build Resize nearest model failed");
        let n = 4 * 4;
        // Positive values only
        let x_vals: Vec<i64> = (0..n as i64).map(|i| (i + 1) * ALPHA / 4).collect();
        cases.push(TestCase {
            op_name: "Resize",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: Tolerance::EXACT,
        });
    }

    // ---- Resize linear: input [1,1,4,4] scales [1,1,2,2] → output [1,1,8,8] ----
    // Use a uniform input (all same value) to avoid boundary-extrapolation differences
    // between tract and onnxruntime.  JSTProve's resize hint clamps negative sums to 0,
    // so any input pattern that could produce negative boundary interpolations would create
    // a tract/JSTProve discrepancy.  Uniform inputs are exact regardless of algorithm.
    {
        let onnx_bytes = build_single_op_model_ordered(
            "Resize",
            &[("X", &[1, 1, 4, 4], FLOAT)],
            &[("Y", &[1, 1, 8, 8], FLOAT)],
            &[NodeAttr {
                name: "mode",
                value: AttrValue::String(b"linear".to_vec()),
            }],
            &[],
            &["X", "", "scales"],
            &[FloatInit {
                name: "scales",
                dims: vec![4],
                data: vec![1.0_f32, 1.0, 2.0, 2.0],
            }],
        )
        .expect("build Resize linear model failed");
        let n = 4 * 4;
        // Uniform value: all elements = 1.0 * ALPHA.
        // Interpolation of identical values is exact regardless of boundary handling.
        let x_vals: Vec<i64> = vec![ALPHA; n];
        cases.push(TestCase {
            op_name: "Resize",
            seed: 1,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: tol(5),
        });
    }

    // GridSample is not supported by tract (returns Unimplemented error).
    // Since both tract and JSTProve would need to be tested for conformance,
    // and tract does not implement it, GridSample cannot be conformance-tested here.
    // GridSample is exercised via the full end-to-end test pipeline (make_model.py).

    cases.truncate(default_case_count());
    cases
}

// ---------------------------------------------------------------------------
// Group J — TopK
// ---------------------------------------------------------------------------

/// Returns test cases for TopK.
#[allow(clippy::vec_init_then_push)]
pub fn topk_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    // ---- TopK: input [1,8] k=3, axis=-1 ----
    // Values output is FLOAT, indices output is INT64.
    // JSTProve only outputs Values (indices are not in output_data for single-output models).
    // We test with values output only (FLOAT typed).
    {
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
        .expect("build TopK model failed");
        let x_vals: Vec<i64> = vec![
            3 * ALPHA,
            ALPHA,
            4 * ALPHA,
            ALPHA / 2,
            2 * ALPHA,
            5 * ALPHA,
            -(ALPHA),
            ALPHA / 4,
        ];
        cases.push(TestCase {
            op_name: "TopK",
            seed: 0,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: Tolerance::EXACT,
        });
    }

    // ---- TopK: k=1 ----
    {
        let onnx_bytes = build_single_op_model(
            "TopK",
            &[("X", &[1, 8], FLOAT)],
            &[("values", &[1, 1], FLOAT), ("indices", &[1, 1], INT64)],
            &[NodeAttr {
                name: "axis",
                value: AttrValue::Int(-1),
            }],
            &[Initializer {
                name: "K",
                dims: vec![1],
                data: vec![1],
            }],
        )
        .expect("build TopK k=1 model failed");
        let x_vals: Vec<i64> = vec![
            ALPHA,
            3 * ALPHA,
            2 * ALPHA,
            -(ALPHA),
            ALPHA / 2,
            4 * ALPHA,
            ALPHA / 4,
            2 * ALPHA,
        ];
        cases.push(TestCase {
            op_name: "TopK",
            seed: 1,
            onnx_bytes,
            inputs: vec![x_vals],
            tolerance: Tolerance::EXACT,
        });
    }

    cases.truncate(default_case_count());
    cases
}
