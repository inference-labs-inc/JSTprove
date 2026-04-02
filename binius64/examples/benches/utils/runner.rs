// Copyright 2025 Irreducible Inc.
//! Common benchmark runner for constraint system benchmarks

use std::error::Error;

use binius_examples::{ExampleCircuit, setup_sha256};
use binius_frontend::CircuitBuilder;
use binius_utils::platform_diagnostics::PlatformDiagnostics;
use binius_verifier::{
    config::StdChallenger,
    transcript::{ProverTranscript, VerifierTranscript},
};
use criterion::{BenchmarkId, Criterion, Throughput};
use peakmem_alloc::PeakAllocTrait;

/// Trait for standardized constraint system benchmarks
pub trait ExampleBenchmark {
    /// Type for circuit parameters
    type Params: Clone;
    /// Type for circuit instance
    type Instance: Clone;
    /// Type for the example circuit
    type Example: ExampleCircuit<Instance = Self::Instance, Params = Self::Params>;

    /// Create benchmark parameters from environment/config
    fn create_params(&self) -> Self::Params;

    /// Create benchmark instance
    fn create_instance(&self) -> Self::Instance;

    /// Build the example circuit - has default implementation that calls Example::build
    fn build_example_circuit(
        params: Self::Params,
        builder: &mut CircuitBuilder,
    ) -> Result<Self::Example, Box<dyn Error>> {
        Self::Example::build(params, builder).map_err(Into::into)
    }

    /// Get benchmark name for reporting
    fn bench_name(&self) -> String;

    /// Get throughput for benchmarking (e.g., bytes, elements)
    fn throughput(&self) -> Throughput;

    /// Get description for proof size reporting
    fn proof_description(&self) -> String;

    /// Get log inverse rate
    fn log_inv_rate(&self) -> usize;

    /// Print benchmark-specific parameters
    fn print_params(&self);
}

/// Run a complete benchmark suite for a constraint system
pub fn run_cs_benchmark<B: ExampleBenchmark>(
    c: &mut Criterion,
    benchmark: B,
    group_prefix: &str,
    peak_alloc: &impl PeakAllocTrait,
) {
    use super::reporting::{print_env_help, print_memory_stats, print_proof_size};

    // Check for help
    print_env_help();

    // Gather and print platform diagnostics
    let diagnostics = PlatformDiagnostics::gather();
    diagnostics.print();

    // Print benchmark-specific parameters
    benchmark.print_params();

    // Setup phase
    let params = benchmark.create_params();
    let instance = benchmark.create_instance();

    let mut builder = CircuitBuilder::new();
    let example = B::build_example_circuit(params.clone(), &mut builder).unwrap();
    let circuit = builder.build();
    let cs = circuit.constraint_system().clone();
    let (verifier, prover) = setup_sha256(cs, benchmark.log_inv_rate(), None).unwrap();

    // Track memory for witness generation
    peak_alloc.reset_peak_memory();
    let mut filler = circuit.new_witness_filler();
    example
        .populate_witness(instance.clone(), &mut filler)
        .unwrap();
    circuit.populate_wire_witness(&mut filler).unwrap();
    let witness = filler.into_value_vec();
    let witness_peak_bytes = peak_alloc.get_peak_memory();

    // Track memory for proof generation
    peak_alloc.reset_peak_memory();
    let mut prover_transcript_mem = ProverTranscript::new(StdChallenger::default());
    prover
        .prove(witness.clone(), &mut prover_transcript_mem)
        .unwrap();
    let proof_bytes_mem = prover_transcript_mem.finalize();
    let proof_peak_bytes = peak_alloc.get_peak_memory();

    // Track memory for verification
    peak_alloc.reset_peak_memory();
    let mut verifier_transcript_mem =
        VerifierTranscript::new(StdChallenger::default(), proof_bytes_mem.clone());
    verifier
        .verify(witness.public(), &mut verifier_transcript_mem)
        .unwrap();
    verifier_transcript_mem.finalize().unwrap();
    let verify_peak_bytes = peak_alloc.get_peak_memory();

    let feature_suffix = diagnostics.get_feature_suffix();
    let bench_name = format!("{}_{}", benchmark.bench_name(), feature_suffix);

    // Benchmark witness generation
    {
        let mut group = c.benchmark_group(format!("{}_witness_generation", group_prefix));
        group.throughput(benchmark.throughput());
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_secs(2));

        group.bench_function(BenchmarkId::from_parameter(&bench_name), |b| {
            b.iter(|| {
                let mut filler = circuit.new_witness_filler();
                example
                    .populate_witness(instance.clone(), &mut filler)
                    .unwrap();
                circuit.populate_wire_witness(&mut filler).unwrap();
                filler.into_value_vec()
            })
        });

        group.finish();
    }

    // Benchmark proof generation
    {
        let mut group = c.benchmark_group(format!("{}_proof_generation", group_prefix));
        group.throughput(benchmark.throughput());
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_secs(2));

        group.bench_function(BenchmarkId::from_parameter(&bench_name), |b| {
            b.iter(|| {
                let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
                prover
                    .prove(witness.clone(), &mut prover_transcript)
                    .unwrap();
                prover_transcript
            })
        });

        group.finish();
    }

    // Generate proof for verification and size measurement
    let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
    prover
        .prove(witness.clone(), &mut prover_transcript)
        .unwrap();
    let proof_bytes = prover_transcript.finalize();
    let proof_size = proof_bytes.len();

    // Benchmark proof verification
    {
        let mut group = c.benchmark_group(format!("{}_proof_verification", group_prefix));
        group.throughput(benchmark.throughput());
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_secs(2));

        group.bench_function(BenchmarkId::from_parameter(&bench_name), |b| {
            b.iter(|| {
                let mut verifier_transcript =
                    VerifierTranscript::new(StdChallenger::default(), proof_bytes.clone());
                verifier
                    .verify(witness.public(), &mut verifier_transcript)
                    .unwrap();
                verifier_transcript.finalize().unwrap()
            })
        });

        group.finish();
    }

    // Report proof size
    print_proof_size(
        &group_prefix.replace('_', " ").to_uppercase(),
        &benchmark.proof_description(),
        proof_size,
    );

    // Print memory statistics
    print_memory_stats(
        &group_prefix.replace('_', " ").to_uppercase(),
        witness_peak_bytes,
        proof_peak_bytes,
        verify_peak_bytes,
    );
}
