// Copyright 2025 Irreducible Inc.

mod utils;

use std::{alloc::System, env};

use binius_examples::circuits::ethsign::{EthSignExample, Instance, Params};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakAlloc;
use utils::{ExampleBenchmark, SignBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static ETHSIGN_PEAK_ALLOC: PeakAlloc<System> = PeakAlloc::new(System);

struct EthSignBenchmark {
    config: SignBenchConfig,
    max_msg_len_bytes: u16,
}

impl EthSignBenchmark {
    fn new() -> Self {
        let config = SignBenchConfig::from_env(1); // default: 1 signature

        // Parse message size from environment variable
        let max_msg_len_bytes = env::var("MESSAGE_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(67);

        Self {
            config,
            max_msg_len_bytes,
        }
    }
}

impl ExampleBenchmark for EthSignBenchmark {
    type Params = Params;
    type Instance = Instance;
    type Example = EthSignExample;

    fn create_params(&self) -> Self::Params {
        Params {
            n_signatures: self.config.n_signatures,
            max_msg_len_bytes: self.max_msg_len_bytes,
        }
    }

    fn create_instance(&self) -> Self::Instance {
        Instance {}
    }

    fn bench_name(&self) -> String {
        format!(
            "sig_{}_msg_{}",
            self.config.n_signatures, self.max_msg_len_bytes
        )
    }

    fn throughput(&self) -> Throughput {
        Throughput::Elements(self.config.n_signatures as u64)
    }

    fn proof_description(&self) -> String {
        format!(
            "{} signatures, {} max bytes",
            self.config.n_signatures, self.max_msg_len_bytes
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
                "Max message length".to_string(),
                format!("{} bytes", self.max_msg_len_bytes),
            ),
            (
                "Log inverse rate".to_string(),
                self.config.log_inv_rate.to_string(),
            ),
        ];
        print_benchmark_header("EthSign", &params_list);
    }
}

fn bench_ethsign_signatures(c: &mut Criterion) {
    let benchmark = EthSignBenchmark::new();
    run_cs_benchmark(c, benchmark, "ethsign", &ETHSIGN_PEAK_ALLOC);
}

criterion_group!(ethsign, bench_ethsign_signatures);
criterion_main!(ethsign);
