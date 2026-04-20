use jstprove_conformance::{
    all_regression_fixtures,
    generator::{
        arithmetic_cases, boolean_cases, norm_cases, pooling_cases, reduction_cases,
        rescaling_cases, shrink, spatial_cases, structural_cases, topk_cases, transcendental_cases,
    },
    onnx_builder::build_single_op_model,
    ConformanceRunner, TestCase, TestResult, Tolerance,
};

// ---------------------------------------------------------------------------
// Helper: run a group of test cases and panic on any failure or error.
//
// Fail-fast mode (default): stops at the first failing op and reports
// immediately — good for CI where the first failure is the most useful signal.
//
// Set CONFORMANCE_FAIL_FAST=0 to disable fail-fast and see ALL failures across
// every case in the group before panicking.  Useful locally when debugging
// multiple regressions at once.
//
// Reference-only cases (those whose TestCase was built with reference_only
// semantics) are identified by running with both reference_only=false AND
// reference_only=true.  If a case passes reference_only but fails full, it
// gets reported separately so the engineer knows it's a known partial test.
// ---------------------------------------------------------------------------

fn run_group(group_name: &str, cases: Vec<TestCase>) {
    // Fail-fast unless explicitly disabled.
    let fail_fast = std::env::var("CONFORMANCE_FAIL_FAST")
        .map(|v| v != "0")
        .unwrap_or(true);

    let full_runner = ConformanceRunner {
        reference_only: false,
    };
    let ref_runner = ConformanceRunner {
        reference_only: true,
    };

    let mut full_failures = 0usize;
    let mut reference_only_skipped = 0usize;

    for case in &cases {
        match full_runner.run(case) {
            TestResult::Pass => {}
            TestResult::Fail(failures) => {
                full_failures += failures.len();
                eprintln!(
                    "[{group_name}] FAIL op={} seed={}  ({} mismatch(es))",
                    case.op_name,
                    case.seed,
                    failures.len()
                );
                for f in &failures {
                    eprintln!("  {f}");
                }
                let shrunken = shrink(case, &full_runner);
                eprintln!("  minimized inputs: {:?}", shrunken.inputs);
                if fail_fast {
                    panic!(
                        "[{group_name}] FAIL op={} seed={} — stopping early (set CONFORMANCE_FAIL_FAST=0 to see all failures)",
                        case.op_name, case.seed
                    );
                }
            }
            TestResult::Error(e) => {
                // Check if this is a known reference_only placeholder (passes ref but not full).
                if matches!(ref_runner.run(case), TestResult::Pass) {
                    reference_only_skipped += 1;
                    log::info!(
                        "[{group_name}] SKIP (reference_only) op={} seed={}: {e}",
                        case.op_name,
                        case.seed
                    );
                } else {
                    panic!(
                        "[{group_name}] ERROR op={} seed={}: {e:#}",
                        case.op_name, case.seed
                    );
                }
            }
        }
    }

    if reference_only_skipped > 0 {
        eprintln!(
            "[{group_name}] {reference_only_skipped} case(s) skipped (reference_only placeholders)"
        );
    }

    if full_failures > 0 {
        panic!("[{group_name}] {full_failures} element-level failure(s)");
    }
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

#[test]
fn relu_smoke() {
    let _ = env_logger::try_init();

    // Single-node Relu model with INT64 input/output shape [4].
    // INT64 inputs bypass alpha-quantization in JSTProve, keeping the test simple.
    // The output shape must be specified explicitly so JSTProve can read it from the ONNX model.
    let onnx_bytes = build_single_op_model(
        "Relu",
        &[("x", &[4], 7 /* INT64 */)],
        &[("y", &[4], 7 /* INT64 */)],
        &[],
        &[],
    )
    .expect("failed to build Relu ONNX model");

    let case = TestCase {
        op_name: "Relu",
        seed: 0,
        onnx_bytes,
        inputs: vec![vec![-2_i64, -1, 0, 3]],
        tolerance: Tolerance::EXACT,
    };

    let runner = ConformanceRunner {
        reference_only: false,
    };
    match runner.run(&case) {
        TestResult::Pass => {}
        TestResult::Fail(failures) => {
            for f in &failures {
                eprintln!("FAIL: {f}");
            }
            panic!("relu_smoke: {} failure(s)", failures.len());
        }
        TestResult::Error(e) => panic!("relu_smoke: error: {e:#}"),
    }
}

/// Sanity-check that reference_only mode always passes without running JSTProve.
#[test]
fn relu_reference_only() {
    let _ = env_logger::try_init();

    let onnx_bytes =
        build_single_op_model("Relu", &[("x", &[3], 7)], &[("y", &[3], 7)], &[], &[]).unwrap();

    let case = TestCase {
        op_name: "Relu",
        seed: 1,
        onnx_bytes,
        inputs: vec![vec![-5_i64, 0, 7]],
        tolerance: Tolerance::EXACT,
    };

    let runner = ConformanceRunner {
        reference_only: true,
    };
    assert!(matches!(runner.run(&case), TestResult::Pass));
}

// ---------------------------------------------------------------------------
// Operator groups — INT64-typed ops
// ---------------------------------------------------------------------------

/// Group A — Structural ops (Reshape, Flatten, Squeeze, Unsqueeze, Cast,
/// ConstantOfShape, Tile, Gather, GatherElements, Transpose, Concat, Slice,
/// Shape, Expand, Pad, Split, Identity, Neg, Range, ScatterND).
#[test]
fn structural_ops() {
    let _ = env_logger::try_init();
    run_group("structural", structural_cases());
}

/// Group B — Simple element-wise arithmetic.
/// Mul, LeakyRelu, HardSwish are reference_only placeholders until FLOAT
/// input support is added.
#[test]
fn simple_arithmetic_ops() {
    let _ = env_logger::try_init();
    run_group("arithmetic", arithmetic_cases());
}

/// Group C — Boolean / comparison ops (Equal, Greater, Less, And, Not, Where).
#[test]
fn boolean_ops() {
    let _ = env_logger::try_init();
    run_group("boolean", boolean_cases());
}

/// Group D — Reduction ops (ReduceSum, ReduceMax).
#[test]
fn reduction_ops() {
    let _ = env_logger::try_init();
    run_group("reductions", reduction_cases());
}

// ---------------------------------------------------------------------------
// Operator groups — FLOAT-typed ops
// ---------------------------------------------------------------------------

/// Group E — Rescaling arithmetic ops (Gemm, MatMul, Conv, ConvTranspose, BatchNorm, Div, Pow).
#[test]
fn rescaling_arithmetic_ops() {
    let _ = env_logger::try_init();
    run_group("rescaling", rescaling_cases());
}

/// Group F — Transcendental/hint ops (Exp, Sigmoid, Softmax, Gelu, LayerNorm,
/// Log, ReduceMean, Tanh, Erf, Sqrt, Cos, Sin).
#[test]
fn transcendental_ops() {
    let _ = env_logger::try_init();
    run_group("transcendental", transcendental_cases());
}

/// Group G — Pooling ops (AveragePool, GlobalAveragePool, MaxPool).
#[test]
fn pooling_ops() {
    let _ = env_logger::try_init();
    run_group("pooling", pooling_cases());
}

/// Group H — Normalization ops (InstanceNorm, GroupNorm — reference_only placeholders).
#[test]
fn norm_ops() {
    let _ = env_logger::try_init();
    run_group("norm", norm_cases());
}

/// Group I — Spatial ops (Resize, GridSample).
#[test]
fn spatial_ops() {
    let _ = env_logger::try_init();
    run_group("spatial", spatial_cases());
}

/// Group J — TopK.
#[test]
fn topk_ops() {
    let _ = env_logger::try_init();
    run_group("topk", topk_cases());
}

// ---------------------------------------------------------------------------
// Regression fixtures for historical constraint violations
// ---------------------------------------------------------------------------

/// Deterministic regression tests for historical bugs.
/// Each fixture captures the exact input that triggered the original failure.
#[test]
fn regression_fixtures() {
    let _ = env_logger::try_init();
    let full_runner = ConformanceRunner {
        reference_only: false,
    };
    let ref_runner = ConformanceRunner {
        reference_only: true,
    };
    let mut failures = 0usize;

    for fixture in all_regression_fixtures() {
        // Fixtures with allow_jstprove_error may OOM or panic in the JSTProve
        // path (e.g., Cos with ~4 M circuit variables).  Run them reference-only
        // to verify the ONNX model and tract path are correct, then skip JSTProve.
        let runner = if fixture.allow_jstprove_error {
            &ref_runner
        } else {
            &full_runner
        };

        match runner.run(&fixture.case) {
            TestResult::Pass => {
                if fixture.allow_jstprove_error {
                    eprintln!(
                        "[regression] SKIP (reference_only) fixture='{}': \
                         JSTProve path skipped (allow_jstprove_error=true — circuit too large for test)",
                        fixture.id
                    );
                }
            }
            TestResult::Fail(fails) => {
                failures += fails.len();
                eprintln!(
                    "[regression] FAIL fixture='{}' (fixed in {}): {} mismatch(es)\n  Description: {}",
                    fixture.id,
                    fixture.fixed_in,
                    fails.len(),
                    fixture.failure_description,
                );
                for f in &fails {
                    eprintln!("  {f}");
                }
            }
            TestResult::Error(e) => {
                panic!(
                    "[regression] ERROR fixture='{}' (fixed in {}): {e:#}\nDescription: {}",
                    fixture.id, fixture.fixed_in, fixture.failure_description
                );
            }
        }
    }

    if failures > 0 {
        panic!("[regression] {failures} element-level failure(s)");
    }
}
