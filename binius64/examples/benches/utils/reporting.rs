// Copyright 2025 Irreducible Inc.
//! Benchmark reporting utilities

use std::env;

use super::config::{DEFAULT_HASH_LOG_INV_RATE, DEFAULT_HASH_MAX_BYTES, DEFAULT_SIGN_LOG_INV_RATE};

/// Print benchmark header with consistent formatting
pub fn print_benchmark_header(name: &str, params: &[(String, String)]) {
    println!("\n{} Benchmark Parameters:", name);
    for (key, value) in params {
        println!("  {}: {}", key, value);
    }
    println!("=======================================\n");
}

/// Print proof size in consistent format
pub fn print_proof_size(bench_name: &str, description: &str, size_bytes: usize) {
    println!(
        "\n{} proof size for {}: {} bytes ({:.2} KiB)",
        bench_name,
        description,
        size_bytes,
        size_bytes as f64 / 1024.0
    );
}

/// Print memory statistics in consistent format
pub fn print_memory_stats(
    bench_name: &str,
    witness_bytes: usize,
    proof_bytes: usize,
    verify_bytes: usize,
) {
    println!("\n{} Peak Memory Consumption:", bench_name);
    println!("  Witness generation: {}", format_memory(witness_bytes));
    println!("  Proof generation: {}", format_memory(proof_bytes));
    println!("  Verification: {}", format_memory(verify_bytes));

    let peak_overall = witness_bytes.max(proof_bytes).max(verify_bytes);
    println!("  Peak overall: {}", format_memory(peak_overall));
}

/// Format bytes as human-readable string
fn format_memory(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;
    const KB: usize = 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Print environment variable help
pub fn print_env_help() {
    if env::var("BENCH_HELP").is_ok() {
        println!("Available environment variables:");
        println!("\nCommon:");
        println!("  LOG_INV_RATE           - Logarithmic inverse rate parameter");
        println!(
            "                           (default: {} for hash, {} for signature)",
            DEFAULT_HASH_LOG_INV_RATE, DEFAULT_SIGN_LOG_INV_RATE
        );
        println!("  BENCH_HELP             - Show this help message");
        println!("\nHash benchmarks:");
        println!(
            "  HASH_MAX_BYTES         - Maximum bytes for hash benchmarks (default: {})",
            DEFAULT_HASH_MAX_BYTES
        );
        println!("\nSignature aggregation benchmarks:");
        println!(
            "  N_SIGNATURES           - Number of signatures (default: 1 for ethsign, 4 for hashsign)"
        );
        println!("  MESSAGE_MAX_BYTES      - Max message bytes for ethsign (default: 67)");
        println!("  XMSS_TREE_HEIGHT       - Tree height for hashsign (default: 13)");
        println!("  WOTS_SPEC              - Winternitz spec for hashsign (default: 2)");
        println!("\nCriterion benchmark timing flags:");
        println!("  --warm-up-time <secs>  - Warm-up time in seconds (e.g., --warm-up-time 0.5)");
        println!(
            "  --measurement-time <s> - Measurement time in seconds (e.g., --measurement-time 2)"
        );
        println!(
            "  --sample-size <n>      - Number of samples to collect (min: 10, e.g., --sample-size 10)"
        );
        println!("\nExample usage:");
        println!(
            "  HASH_MAX_BYTES=256 cargo bench --bench keccak -- --warm-up-time 0.1 --measurement-time 0.5 --sample-size 10"
        );
        std::process::exit(0);
    }
}
