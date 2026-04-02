// Copyright 2025 Irreducible Inc.
use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result};
use binius_frontend::{Circuit, CircuitStat};

/// Get the workspace root directory by using CARGO_MANIFEST_DIR
fn workspace_root() -> Result<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").context(
        "CARGO_MANIFEST_DIR environment variable not set. \
		Please run this command via 'cargo run --example <name>' or 'cargo test'.",
    )?;

    // CARGO_MANIFEST_DIR points to prover/examples, so go up two levels to reach workspace root
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .context("Failed to determine workspace root from CARGO_MANIFEST_DIR")?
        .to_path_buf();

    Ok(workspace_root)
}

/// Get the snapshot file path for a circuit example
pub fn snapshot_path(circuit_name: &str) -> Result<PathBuf> {
    let root = workspace_root()?;
    Ok(root.join(format!("crates/examples/snapshots/{}.snap", circuit_name)))
}

/// Format circuit statistics for snapshot
pub fn format_circuit_stats(circuit_name: &str, circuit: &Circuit) -> String {
    let mut output = String::new();
    output.push_str(&format!("{} circuit\n", circuit_name));
    output.push_str("--\n");

    let stat = CircuitStat::collect(circuit);
    output.push_str(&format!("{}", stat));
    output
}

/// Check if circuit statistics match snapshot
pub fn check_snapshot(circuit_name: &str, circuit: &Circuit) -> Result<()> {
    let snapshot_path = snapshot_path(circuit_name)?;

    if !snapshot_path.exists() {
        anyhow::bail!(
            "Snapshot file not found at {}\n\
            Run '{} bless-snapshot' to create it.",
            snapshot_path.display(),
            circuit_name
        );
    }

    let expected = fs::read_to_string(&snapshot_path)
        .with_context(|| format!("Failed to read snapshot file: {}", snapshot_path.display()))?;

    let actual = format_circuit_stats(circuit_name, circuit);

    if expected != actual {
        print_snapshot_diff(&expected, &actual, circuit_name);
        anyhow::bail!("Circuit statistics do not match snapshot");
    }

    println!("✓ Circuit statistics match snapshot");
    Ok(())
}

/// Update snapshot with current circuit statistics
pub fn bless_snapshot(circuit_name: &str, circuit: &Circuit) -> Result<()> {
    let snapshot_path = snapshot_path(circuit_name)?;

    // Create snapshots directory if it doesn't exist
    if let Some(parent) = snapshot_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("Failed to create snapshot directory: {}", parent.display())
        })?;
    }

    let output = format_circuit_stats(circuit_name, circuit);

    fs::write(&snapshot_path, &output)
        .with_context(|| format!("Failed to write snapshot file: {}", snapshot_path.display()))?;

    println!("✓ Snapshot updated at {}", snapshot_path.display());
    Ok(())
}

/// Print a diff between expected and actual snapshots
fn print_snapshot_diff(expected: &str, actual: &str, circuit_name: &str) {
    eprintln!("Error: Circuit statistics do not match snapshot!");
    eprintln!("\n--- Expected (from snapshot) ---");
    eprintln!("{}", expected);
    eprintln!("\n--- Actual ---");
    eprintln!("{}", actual);
    eprintln!("\n--- Diff ---");

    // Simple line-by-line diff
    let expected_lines: Vec<_> = expected.lines().collect();
    let actual_lines: Vec<_> = actual.lines().collect();

    let max_lines = expected_lines.len().max(actual_lines.len());
    for i in 0..max_lines {
        let exp_line = expected_lines.get(i).unwrap_or(&"");
        let act_line = actual_lines.get(i).unwrap_or(&"");

        if exp_line != act_line {
            eprintln!("Line {}: - {}", i + 1, exp_line);
            eprintln!("Line {}: + {}", i + 1, act_line);
        }
    }

    eprintln!(
        "\nRun '{} bless-snapshot' to update the snapshot.",
        circuit_name
    );
}
