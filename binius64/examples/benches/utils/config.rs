// Copyright 2025 Irreducible Inc.
//! Benchmark configuration utilities

use std::env;

/// Default HASH_MAX_BYTES for hash benchmarks (1KiB)
pub const DEFAULT_HASH_MAX_BYTES: usize = 1024;

/// Default LOG_INV_RATE for hash benchmarks
pub const DEFAULT_HASH_LOG_INV_RATE: usize = 1;

/// Default LOG_INV_RATE for signature benchmarks
pub const DEFAULT_SIGN_LOG_INV_RATE: usize = 2;

/// Common configuration for hash benchmarks
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HashBenchConfig {
    pub max_bytes: usize,
    pub log_inv_rate: usize,
}

#[allow(dead_code)]
impl HashBenchConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let max_bytes = env::var("HASH_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_HASH_MAX_BYTES);

        let log_inv_rate = env::var("LOG_INV_RATE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_HASH_LOG_INV_RATE);

        Self {
            max_bytes,
            log_inv_rate,
        }
    }
}

/// Common configuration for signature benchmarks
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SignBenchConfig {
    pub n_signatures: usize,
    pub log_inv_rate: usize,
}

#[allow(dead_code)]
impl SignBenchConfig {
    /// Create configuration from environment variables
    pub fn from_env(default_signatures: usize) -> Self {
        let n_signatures = env::var("N_SIGNATURES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default_signatures);

        let log_inv_rate = env::var("LOG_INV_RATE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SIGN_LOG_INV_RATE);

        Self {
            n_signatures,
            log_inv_rate,
        }
    }
}
