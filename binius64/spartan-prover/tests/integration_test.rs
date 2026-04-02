// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{BinaryField128bGhash as B128, Random, arch::OptimalPackedB128};
use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
use binius_spartan_frontend::{
    circuit_builder::{CircuitBuilder, ConstraintBuilder, WitnessGenerator},
    circuits::powers,
    compiler::compile,
};
use binius_spartan_prover::Prover;
use binius_spartan_verifier::{Verifier, config::StdChallenger};
use binius_transcript::ProverTranscript;
use rand::{SeedableRng, rngs::StdRng};

// Build a power7 circuit: assert that x^7 = y
fn power7_circuit<Builder: CircuitBuilder>(
    builder: &mut Builder,
    x_wire: Builder::Wire,
    y_wire: Builder::Wire,
) {
    let powers_vec = powers(builder, x_wire, 7);
    let x7 = powers_vec[6]; // x^7 is the 7th element (0-indexed)
    builder.assert_eq(x7, y_wire);
}

#[test]
fn test_power7_circuit_prover_verifier() {
    // Build the constraint system
    let mut constraint_builder = ConstraintBuilder::new();
    let x_wire = constraint_builder.alloc_inout();
    let y_wire = constraint_builder.alloc_inout();
    power7_circuit(&mut constraint_builder, x_wire, y_wire);
    let (cs, layout) = compile(constraint_builder);

    // Setup prover and verifier
    let log_inv_rate = 1;
    let compression = StdCompression::default();
    let verifier = Verifier::<_, StdDigest, _>::setup(cs, log_inv_rate, compression.clone())
        .expect("verifier setup failed");
    let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(
        verifier.clone(),
        ParallelCompressionAdaptor::new(compression),
    )
    .expect("prover setup failed");

    let cs = verifier.constraint_system();
    let layout = layout.with_blinding(cs.blinding_info().clone());

    // Choose test values: x = random, y = x^7
    let mut rng = StdRng::seed_from_u64(0);
    let x_val = B128::random(&mut rng);
    let y_val = x_val * x_val * x_val * x_val * x_val * x_val * x_val; // x^7

    // Generate witness
    let mut witness_gen = WitnessGenerator::new(&layout);
    let x_assigned = witness_gen.write_inout(x_wire, x_val);
    let y_assigned = witness_gen.write_inout(y_wire, y_val);
    power7_circuit(&mut witness_gen, x_assigned, y_assigned);
    let witness = witness_gen.build().expect("failed to build witness");

    // Validate witness satisfies constraints
    cs.validate(&witness);

    // Extract public inputs (constants + inout, padded to 2^log_public)
    let public = &witness[..1 << cs.log_public()];

    // Generate proof
    let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
    prover
        .prove(&witness, &mut rng, &mut prover_transcript)
        .expect("prove failed");

    // Verify proof
    let mut verifier_transcript = prover_transcript.into_verifier();
    verifier
        .verify(public, &mut verifier_transcript)
        .expect("verify failed");
    verifier_transcript.finalize().expect("finalize failed");
}
