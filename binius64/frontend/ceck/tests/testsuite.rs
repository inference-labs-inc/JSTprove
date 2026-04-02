// Copyright 2025 Irreducible Inc.
use std::{path::PathBuf, process::Command};

fn run_ceck_test(test_file: &str) -> Result<(), String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_path = PathBuf::from(manifest_dir)
        .join("testsuite")
        .join(test_file);

    // Find the ceck binary
    let exe_path = env!("CARGO_BIN_EXE_ceck");

    let mut cmd = Command::new(exe_path);
    cmd.arg(test_path);

    // If z3 is not available, we can only rely on randblast.
    #[cfg(not(feature = "z3"))]
    cmd.arg("--optimistic");

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute ceck: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Test {test_file} failed:\n{stderr}"));
    }

    Ok(())
}

#[test]
fn test_basic_and() {
    run_ceck_test("basic_and.ceck").expect("basic_and.ceck should pass");
}

#[test]
fn test_xor_operations() {
    run_ceck_test("xor.ceck").expect("xor.ceck should pass");
}

#[test]
fn test_shift_operations() {
    run_ceck_test("shift.ceck").expect("shift.ceck should pass");
}

#[test]
fn test_mul_constraints() {
    run_ceck_test("mul.ceck").expect("mul.ceck should pass");
}

#[test]
fn test_mixed_constraints() {
    run_ceck_test("mixed.ceck").expect("mixed.ceck should pass");
}

#[test]
fn test_edge_cases() {
    run_ceck_test("edge_cases.ceck").expect("edge_cases.ceck should pass");
}

#[test]
fn test_assertion_types() {
    run_ceck_test("assertion_types.ceck").expect("assertion_types.ceck should pass");
}

#[test]
fn test_smt_edge_cases() {
    run_ceck_test("smt_edge_cases.ceck").expect("smt_edge_cases.ceck should pass");
}
