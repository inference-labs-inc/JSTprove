use jstprove_conformance::{
    ConformanceRunner, TestCase, TestResult, Tolerance,
    onnx_builder::build_single_op_model,
};

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

    let runner = ConformanceRunner { reference_only: false };
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

    let onnx_bytes = build_single_op_model(
        "Relu",
        &[("x", &[3], 7)],
        &[("y", &[3], 7)],
        &[],
        &[],
    )
    .unwrap();

    let case = TestCase {
        op_name: "Relu",
        seed: 1,
        onnx_bytes,
        inputs: vec![vec![-5_i64, 0, 7]],
        tolerance: Tolerance::EXACT,
    };

    let runner = ConformanceRunner { reference_only: true };
    assert!(matches!(runner.run(&case), TestResult::Pass));
}
