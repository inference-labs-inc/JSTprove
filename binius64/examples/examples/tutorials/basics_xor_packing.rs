// Copyright 2025 Irreducible Inc.

//! XOR Packing Example
//!
//! Demonstrates how multiple XOR operations can be packed into a single constraint.
//!
//! Guide: https://www.binius.xyz/building/

use binius_core::{verify::verify_constraints, word::Word};
use binius_frontend::CircuitBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First show the naive approach
    println!("Naive approach: chaining individual XORs");
    let builder_naive = CircuitBuilder::new();

    let a = builder_naive.add_constant_64(0x1111);
    let b = builder_naive.add_constant_64(0x2222);
    let c = builder_naive.add_constant_64(0x4444);
    let d = builder_naive.add_constant_64(0x8888);

    // Naive: 3 separate XOR gates, 3 constraints
    let temp1 = builder_naive.bxor(a, b); // Constraint 1
    let temp2 = builder_naive.bxor(temp1, c); // Constraint 2
    let result_naive = builder_naive.bxor(temp2, d); // Constraint 3

    let out_naive = builder_naive.add_inout();
    builder_naive.assert_eq("naive_result", result_naive, out_naive);

    let circuit_naive = builder_naive.build();
    let mut witness_naive = circuit_naive.new_witness_filler();
    witness_naive[out_naive] = Word(0x1111 ^ 0x2222 ^ 0x4444 ^ 0x8888);
    circuit_naive.populate_wire_witness(&mut witness_naive)?;

    let cs_naive = circuit_naive.constraint_system();
    verify_constraints(cs_naive, &witness_naive.into_value_vec())?;

    println!(
        "  Used {} AND constraints (for XOR operations)",
        cs_naive.n_and_constraints()
    );

    // Now show the optimized approach
    println!("\nOptimized approach: multi-XOR packing");
    let builder_opt = CircuitBuilder::new();

    let a = builder_opt.add_constant_64(0x1111);
    let b = builder_opt.add_constant_64(0x2222);
    let c = builder_opt.add_constant_64(0x4444);
    let d = builder_opt.add_constant_64(0x8888);

    // Optimized: single constraint for all XORs
    let result_opt = builder_opt.bxor_multi(&[a, b, c, d]);

    let out_opt = builder_opt.add_inout();
    builder_opt.assert_eq("optimized_result", result_opt, out_opt);

    let circuit_opt = builder_opt.build();
    let mut witness_opt = circuit_opt.new_witness_filler();
    witness_opt[out_opt] = Word(0x1111 ^ 0x2222 ^ 0x4444 ^ 0x8888);
    circuit_opt.populate_wire_witness(&mut witness_opt)?;

    // Save result before moving witness
    let result_val = witness_opt[out_opt].0;

    let cs_opt = circuit_opt.constraint_system();
    verify_constraints(cs_opt, &witness_opt.into_value_vec())?;

    println!(
        "  Used {} AND constraint (all XORs packed)",
        cs_opt.n_and_constraints()
    );

    println!("\n✓ Example 4: XOR packing optimization");
    println!(
        "  Naive: {} constraints, Optimized: {} constraint",
        cs_naive.n_and_constraints(),
        cs_opt.n_and_constraints()
    );
    println!("  Result: 0x{:04X}", result_val);

    Ok(())
}
