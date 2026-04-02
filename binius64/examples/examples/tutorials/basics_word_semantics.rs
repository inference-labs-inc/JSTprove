// Copyright 2025 Irreducible Inc.

//! Word Semantics Example
//!
//! Shows how Binius64 performs different operations on 64-bit words,
//! demonstrating various gate semantics.
//!
//! Guide: https://www.binius.xyz/building/

use binius_core::verify::verify_constraints;
use binius_frontend::CircuitBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let builder = CircuitBuilder::new();

    // Create some test values
    let a = builder.add_constant_64(0xFF00FF00);
    let b = builder.add_constant_64(0x00FF00FF);
    let x = builder.add_constant_64(100);
    let y = builder.add_constant_64(200);
    let p = builder.add_constant_64(50);
    let q = builder.add_constant_64(100);
    let v = builder.add_constant_64(0xABCDEF12);
    let mask = builder.add_constant_64(0xFFFF0000);

    // Same 64-bit words, different operations
    let xor_result = builder.bxor(a, b);
    let zero = builder.add_constant_64(0);
    let (int_sum, _) = builder.iadd_cin_cout(x, y, zero);
    let cmp_result = builder.icmp_ult(p, q);
    let bit_pattern = builder.band(v, mask);

    let circuit = builder.build();

    // Populate witness
    let mut witness = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut witness)?;

    // Save computed values
    let val1 = witness[xor_result].0;
    let val2 = witness[int_sum].0;
    let val3 = witness[cmp_result].0;
    let val4 = witness[bit_pattern].0;

    // Verify
    let cs = circuit.constraint_system();
    verify_constraints(cs, &witness.into_value_vec())?;

    println!("✓ Example 2: Word operations demonstration");
    println!("  XOR result: 0x{:016X}", val1);
    println!("  Integer sum: {}", val2);
    println!("  Comparison (p < q): 0x{:016X}", val3);
    println!("  AND result: 0x{:016X}", val4);
    println!(
        "  Note: Comparison returns 0x{:X} (true = -1 in two's complement)",
        val3
    );

    Ok(())
}
