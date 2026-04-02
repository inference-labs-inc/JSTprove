// Copyright 2025 Irreducible Inc.
use std::env;

fn main() {
    // Pass build configuration through environment variables
    println!(
        "cargo:rustc-env=BUILD_TARGET={}",
        env::var("TARGET").unwrap()
    );
    println!("cargo:rustc-env=BUILD_HOST={}", env::var("HOST").unwrap());
    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        env::var("PROFILE").unwrap()
    );

    // Handle RUSTFLAGS - Cargo may use CARGO_ENCODED_RUSTFLAGS instead of RUSTFLAGS
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .or_else(|_| env::var("RUSTFLAGS"))
        .unwrap_or_default()
        .replace('\x1f', " "); // CARGO_ENCODED_RUSTFLAGS uses 0x1f as separator
    println!("cargo:rustc-env=BUILD_RUSTFLAGS={rustflags}");

    // Collect target features
    let target_features = if let Ok(features) = env::var("CARGO_CFG_TARGET_FEATURE") {
        if features.is_empty() {
            Vec::new()
        } else {
            features.split(',').map(|s| s.to_string()).collect()
        }
    } else {
        Vec::new()
    };

    // Pass target features as comma-separated list
    println!(
        "cargo:rustc-env=COMPILE_TIME_FEATURES={}",
        target_features.join(",")
    );
}
