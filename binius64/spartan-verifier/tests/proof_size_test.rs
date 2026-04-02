// Copyright 2026 The Binius Developers

use binius_field::BinaryField128bGhash as B128;
use binius_spartan_frontend::{
    circuit_builder::{CircuitBuilder, ConstraintBuilder},
    circuits::powers,
    compiler::compile,
};
use binius_spartan_verifier::{
    Verifier,
    config::{StdCompression, StdDigest},
};

// Build a power circuit: assert that x^n = y
fn power_circuit<Builder: CircuitBuilder>(
    builder: &mut Builder,
    x_wire: Builder::Wire,
    y_wire: Builder::Wire,
    n: usize,
) {
    let powers_vec = powers(builder, x_wire, n);
    let xn = powers_vec[n - 1];
    builder.assert_eq(xn, y_wire);
}

#[test]
fn test_ip_proof_size() {
    // Build the constraint system
    let mut constraint_builder = ConstraintBuilder::new();
    let x_wire = constraint_builder.alloc_inout();
    let y_wire = constraint_builder.alloc_inout();
    power_circuit(&mut constraint_builder, x_wire, y_wire, 7);
    let (cs, _layout) = compile(constraint_builder);

    // Setup verifier
    let log_inv_rate = 3;
    let compression = StdCompression::default();
    let verifier = Verifier::<_, StdDigest, _>::setup(cs, log_inv_rate, compression)
        .expect("verifier setup failed");

    let cs = verifier.constraint_system();

    // Create size tracking channel and run verify_iop with dummy public inputs
    // (SizeTrackingChannel ignores values).
    let mut channel = verifier.iop_compiler().create_size_tracking_channel();
    let public = vec![B128::default(); 1 << cs.log_public()];
    verifier
        .verify_iop(&public, &mut channel)
        .expect("verify_iop with size tracking channel should succeed");
    let proof_size = channel.proof_size();

    // Hardcoded expected value to detect proof size regressions.
    // This measures IP-layer bytes (sumcheck rounds, oracle commitments, evaluations) plus
    // FRI proof sizes. It is a slight underestimate because it does not account for some
    // smaller BaseFold components (e.g. sumcheck coefficients within BaseFold, blinding
    // elements for ZK).
    assert_eq!(proof_size, 91152, "proof size regression");
}
