// Copyright 2025 Irreducible Inc.

//! Masked AND
//!
//! Example showing how to create a circuit that proves knowledge of a value
//! that produces a specific result when AND'd with a mask.
//!
//! Guide: https://www.binius.xyz/building/

use binius_core::{verify::verify_constraints, word::Word};
use binius_frontend::{CircuitBuilder, stat::CircuitStat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Masked AND Example ===\n");

    // Phase 1: Circuit Building
    let builder = CircuitBuilder::new();

    let mask = builder.add_constant_64(0xFF00);
    let private = builder.add_witness();
    let result = builder.band(private, mask);

    // Public output for verification
    let output = builder.add_inout();
    builder.assert_eq("masked_result", result, output);

    let circuit = builder.build();

    // Phase 2: Witness Generation
    let mut w = circuit.new_witness_filler();
    w[private] = Word(0x1234);
    w[output] = Word(0x1200);

    circuit.populate_wire_witness(&mut w)?;

    // Phase 3: Constraint Verification
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec())?;

    println!("✓ Proof verified: Someone knows a value that ANDs with 0xFF00 to produce 0x1200");
    let stat = CircuitStat::collect(&circuit);
    println!("  Circuit used {} AND constraints", stat.n_and_constraints);

    Ok(())
}
