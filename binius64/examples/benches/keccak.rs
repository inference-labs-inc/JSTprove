// Copyright 2025 Irreducible Inc.

mod utils;

use std::alloc::System;

use binius_examples::circuits::keccak::{Instance, KeccakExample, Params};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakAlloc;
use utils::{ExampleBenchmark, HashBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static KECCAK_PEAK_ALLOC: PeakAlloc<System> = PeakAlloc::new(System);

struct KeccakBenchmark {
    config: HashBenchConfig,
}

impl KeccakBenchmark {
    fn new() -> Self {
        let config = HashBenchConfig::from_env();
        Self { config }
    }
}

impl ExampleBenchmark for KeccakBenchmark {
    type Params = Params;
    type Instance = Instance;
    type Example = KeccakExample;

    fn create_params(&self) -> Self::Params {
        Params {
            max_len_bytes: Some(self.config.max_bytes),
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
        const KECCAK_256_RATE: usize = 136;
        let n_permutations = self.config.max_bytes.div_ceil(KECCAK_256_RATE);
        let params = vec![
            (
                "Message size".to_string(),
                format!("{} bytes", self.config.max_bytes),
            ),
            (
                "Permutations required".to_string(),
                format!(
                    "{} (for {} bytes at {} bytes/permutation)",
                    n_permutations, self.config.max_bytes, KECCAK_256_RATE
                ),
            ),
            (
                "Log inverse rate".to_string(),
                self.config.log_inv_rate.to_string(),
            ),
        ];
        print_benchmark_header("Keccak-256", &params);
    }
}

fn bench_keccak(c: &mut Criterion) {
    let benchmark = KeccakBenchmark::new();
    run_cs_benchmark(c, benchmark, "keccak", &KECCAK_PEAK_ALLOC);
}

criterion_group!(keccak, bench_keccak);
criterion_main!(keccak);
