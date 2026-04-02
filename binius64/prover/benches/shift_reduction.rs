// Copyright 2025 Irreducible Inc.

use binius_circuits::sha256::Sha256;
use binius_core::{ValueVec, constraint_system::ConstraintSystem};
use binius_frontend::CircuitBuilder;
use binius_prover::protocols::shift::{OperatorData, build_key_collection, prove};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::strict_log_2;
use binius_verifier::{
    config::StdChallenger,
    protocols::shift::{OperatorData as VerifierOperatorData, verify},
};
use criterion::{Criterion, criterion_group, criterion_main};
use sha2::{Digest, Sha256 as Sha256Hasher};

pub fn create_sha256_cs_with_witness(
    log_message_len_bytes: usize,
    rng: &mut impl rand::Rng,
) -> (ConstraintSystem, ValueVec) {
    let builder = CircuitBuilder::new();
    let message_len_bytes: usize = 1 << log_message_len_bytes; // 2^log_message_len

    // Create wires for the SHA256 circuit
    let len = builder.add_witness(); // Actual message length
    let digest = [
        builder.add_inout(), // Expected digest as 4x64-bit words
        builder.add_inout(),
        builder.add_inout(),
        builder.add_inout(),
    ];
    let message: Vec<binius_frontend::Wire> = (0..message_len_bytes.div_ceil(8usize))
        .map(|_| builder.add_witness())
        .collect();

    // Create the SHA256 circuit
    let sha256 = Sha256::new(&builder, len, digest, message);

    let circuit = builder.build();
    let mut witness_filler = circuit.new_witness_filler();

    // Generate random message bytes of specified length
    let mut message_bytes = vec![0u8; message_len_bytes];
    rng.fill_bytes(&mut message_bytes);
    sha256.populate_len_bytes(&mut witness_filler, message_bytes.len());
    sha256.populate_message(&mut witness_filler, &message_bytes);

    // Calculate SHA256 digest of the message dynamically
    let hash = Sha256Hasher::digest(&message_bytes);
    let expected_digest: [u8; 32] = hash.into();
    sha256.populate_digest(&mut witness_filler, expected_digest);

    // Get the witness vector
    circuit.populate_wire_witness(&mut witness_filler).unwrap();

    (
        circuit.constraint_system().clone(),
        witness_filler.into_value_vec(),
    )
}

fn bench_prove_and_verify(c: &mut Criterion) {
    use binius_field::{BinaryField128bGhash, PackedBinaryGhash1x128b, Random};
    type F = BinaryField128bGhash;
    type P = PackedBinaryGhash1x128b;
    let mut rng = rand::rng();

    // Configurable log message lengths to benchmark (actual lengths will be 2^log_len)
    let log_message_lengths_bytes = [8, 12, 16]; // Actual lengths: 256, 4096, 65536 bytes

    for &log_message_len_bytes in &log_message_lengths_bytes {
        let message_len_bytes = 1 << log_message_len_bytes;
        let (mut cs, value_vec) = create_sha256_cs_with_witness(log_message_len_bytes, &mut rng);
        cs.validate_and_prepare().unwrap();

        // Sample multilinear eval points
        let r_x_prime_bitand = {
            let log_bitand_constraint_count = strict_log_2(cs.and_constraints.len()).unwrap();
            (0..log_bitand_constraint_count as u128)
                .map(F::new)
                .collect::<Vec<_>>()
        };
        let r_x_prime_intmul = {
            let log_intmul_constraint_count = strict_log_2(cs.mul_constraints.len()).unwrap();
            (0..log_intmul_constraint_count as u128)
                .map(F::new)
                .collect::<Vec<_>>()
        };

        // Sample univaraite eval point
        let r_zhat_prime_bitand = F::random(&mut rng);
        let r_zhat_prime_intmul = F::random(&mut rng);

        let bitand_evals = [F::random(&mut rng); 3];
        let intmul_evals = [F::random(&mut rng); 4];
        let key_collection = build_key_collection(&cs);

        let inout_n_vars = strict_log_2(cs.value_vec_layout.offset_witness).unwrap();

        let mut group = c.benchmark_group(format!(
            "shift_reduction_log2_{log_message_len_bytes}_bytes_{message_len_bytes}"
        ));
        group.sample_size(10);

        group.bench_function("prove", |b| {
            b.iter(|| {
                let prover_bitand_data = OperatorData {
                    evals: bitand_evals.to_vec(),
                    r_zhat_prime: r_zhat_prime_bitand,
                    r_x_prime: r_x_prime_bitand.clone(),
                };
                let prover_intmul_data = OperatorData {
                    evals: intmul_evals.to_vec(),
                    r_zhat_prime: r_zhat_prime_intmul,
                    r_x_prime: r_x_prime_intmul.clone(),
                };

                let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

                prove::<F, P, _>(
                    inout_n_vars,
                    &key_collection,
                    value_vec.combined_witness(),
                    prover_bitand_data,
                    prover_intmul_data,
                    &mut prover_transcript,
                )
                .unwrap()
            })
        });

        // Pre-run the prover to get the transcript for verifier benchmarking
        let prover_bitand_data = OperatorData {
            evals: bitand_evals.to_vec(),
            r_zhat_prime: r_zhat_prime_bitand,
            r_x_prime: r_x_prime_bitand.clone(),
        };
        let prover_intmul_data = OperatorData {
            evals: intmul_evals.to_vec(),
            r_zhat_prime: r_zhat_prime_intmul,
            r_x_prime: r_x_prime_intmul.clone(),
        };

        let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

        prove::<F, P, _>(
            inout_n_vars,
            &key_collection,
            value_vec.combined_witness(),
            prover_bitand_data,
            prover_intmul_data,
            &mut prover_transcript,
        )
        .unwrap();

        let setup_verifier_transcript = prover_transcript.into_verifier();

        group.bench_function("verify", |b| {
            b.iter(|| {
                let mut verifier_transcript = setup_verifier_transcript.clone();

                let verifier_bitand_data = VerifierOperatorData::new(
                    r_zhat_prime_bitand,
                    r_x_prime_bitand.clone(),
                    bitand_evals,
                );
                let verifier_intmul_data = VerifierOperatorData::new(
                    r_zhat_prime_intmul,
                    r_x_prime_intmul.clone(),
                    intmul_evals,
                );

                verify(
                    &cs,
                    value_vec.public(),
                    &verifier_bitand_data,
                    &verifier_intmul_data,
                    &mut verifier_transcript,
                )
                .unwrap();
            })
        });
    }
}

criterion_group!(benches, bench_prove_and_verify);
criterion_main!(benches);
