// Copyright 2025 Irreducible Inc.
//! SHA-256 hash benchmark

mod utils;

use std::alloc::System;

use binius_examples::circuits::sha256::{Instance, Params, Sha256Example};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakAlloc;
use utils::{ExampleBenchmark, HashBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static SHA256_PEAK_ALLOC: PeakAlloc<System> = PeakAlloc::new(System);

struct Sha256Benchmark {
    config: HashBenchConfig,
}

impl Sha256Benchmark {
    fn new() -> Self {
        let config = HashBenchConfig::from_env();
        Self { config }
    }
}

impl ExampleBenchmark for Sha256Benchmark {
    type Params = Params;
    type Instance = Instance;
    type Example = Sha256Example;

    fn create_params(&self) -> Self::Params {
        Params {
            max_len_bytes: Some(self.config.max_bytes),
            exact_len: true,
        }
    }

    fn create_instance(&self) -> Self::Instance {
        Instance {
            message_len: Some(self.config.max_bytes),
            message_string: None,
        }
    }

    fn bench_name(&self) -> String {
        format!("message_bytes_{}", self.config.max_bytes)
    }

    fn throughput(&self) -> Throughput {
        Throughput::Bytes(self.config.max_bytes as u64)
    }

    fn proof_description(&self) -> String {
        format!("{} bytes message", self.config.max_bytes)
    }

    fn log_inv_rate(&self) -> usize {
        self.config.log_inv_rate
    }

    fn print_params(&self) {
        let blocks = self.config.max_bytes.div_ceil(64);
        let params_list = vec![
            (
                "Circuit capacity".to_string(),
                format!(
                    "{} bytes ({} blocks × 64 bytes/block)",
                    self.config.max_bytes, blocks
                ),
            ),
            (
                "Message length".to_string(),
                format!("{} bytes (using full capacity)", self.config.max_bytes),
            ),
            (
                "Log inverse rate".to_string(),
                self.config.log_inv_rate.to_string(),
            ),
        ];
        print_benchmark_header("SHA-256", &params_list);
    }
}

fn bench_sha256_hash(c: &mut Criterion) {
    let benchmark = Sha256Benchmark::new();
    run_cs_benchmark(c, benchmark, "sha256", &SHA256_PEAK_ALLOC);
}

criterion_group!(benches, bench_sha256_hash);
criterion_main!(benches);
