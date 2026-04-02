// Copyright 2025 Irreducible Inc.

mod utils;

use std::{alloc::System, env};

use binius_examples::circuits::hashsign::{HashBasedSigExample, Instance, Params};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakAlloc;
use utils::{ExampleBenchmark, SignBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static HASHSIGN_PEAK_ALLOC: PeakAlloc<System> = PeakAlloc::new(System);

struct HashSignBenchmark {
    config: SignBenchConfig,
    tree_height: usize,
    spec: u8,
}

impl HashSignBenchmark {
    fn new() -> Self {
        let config = SignBenchConfig::from_env(4); // default: 4 signatures

        // Parse XMSS/WOTS parameters from environment variables
        let tree_height = env::var("XMSS_TREE_HEIGHT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(13);

        let spec = env::var("WOTS_SPEC")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(2);

        Self {
            config,
            tree_height,
            spec,
        }
    }
}

impl ExampleBenchmark for HashSignBenchmark {
    type Params = Params;
    type Instance = Instance;
    type Example = HashBasedSigExample;

    fn create_params(&self) -> Self::Params {
        Params {
            num_validators: self.config.n_signatures,
            tree_height: self.tree_height,
            spec: self.spec,
        }
    }

    fn create_instance(&self) -> Self::Instance {
        Instance {}
    }

    fn bench_name(&self) -> String {
        format!("sig_{}_tree_{}", self.config.n_signatures, self.tree_height)
    }

    fn throughput(&self) -> Throughput {
        Throughput::Elements(self.config.n_signatures as u64)
    }

    fn proof_description(&self) -> String {
        format!(
            "{} signatures (tree height {})",
            self.config.n_signatures, self.tree_height
        )
    }

    fn log_inv_rate(&self) -> usize {
        self.config.log_inv_rate
    }

    fn print_params(&self) {
        let params_list = vec![
            (
                "Signatures".to_string(),
                self.config.n_signatures.to_string(),
            ),
            (
                "XMSS tree height".to_string(),
                format!(
                    "{} (2^{} = {} slots)",
                    self.tree_height,
                    self.tree_height,
                    1 << self.tree_height
                ),
            ),
            ("WOTS spec".to_string(), self.spec.to_string()),
            ("Message size".to_string(), "32 bytes (fixed)".to_string()),
            (
                "Log inverse rate".to_string(),
                self.config.log_inv_rate.to_string(),
            ),
        ];
        print_benchmark_header("Hashsign", &params_list);
    }
}

fn bench_hashsign(c: &mut Criterion) {
    let benchmark = HashSignBenchmark::new();
    run_cs_benchmark(c, benchmark, "hashsign", &HASHSIGN_PEAK_ALLOC);
}

criterion_group!(hashsign, bench_hashsign);
criterion_main!(hashsign);
