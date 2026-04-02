// Copyright 2025 Irreducible Inc.

//! Non-Deterministic Values Example
//!
//! Demonstrates modular arithmetic verification using hints for quotient/remainder.
//!
//! Guide: https://www.binius.xyz/building/

use binius_core::verify::verify_constraints;
use binius_frontend::CircuitBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let builder = CircuitBuilder::new();

    // To verify: 100 mod 83 = 17

    // BigUint division for multi-limb numbers
    let dividend = vec![builder.add_constant_64(100)];
    let divisor = vec![builder.add_constant_64(83)];

    // Hint computes quotient and remainder externally
    let (quotient, remainder) = builder.biguint_divide_hint(&dividend, &divisor);

    // Verify: dividend = divisor × quotient + remainder
    let (_hi, lo) = builder.imul(divisor[0], quotient[0]);
    let zero_wire = builder.add_constant_64(0);
    let sum = builder.iadd_cin_cout(lo, remainder[0], zero_wire).0;
    builder.assert_eq("modulo_check", sum, dividend[0]);

    // Also verify remainder < divisor (for completeness)
    let rem_less_than_div = builder.icmp_ult(remainder[0], divisor[0]);

    let circuit = builder.build();

    // The prover computes 100 ÷ 83 = 1 remainder 17 and fills these wires
    let mut witness = circuit.new_witness_filler();
    circuit.populate_wire_witness(&mut witness)?;

    // Check the boolean value
    let bool_val = witness[rem_less_than_div].0;

    // Verify constraints
    let cs = circuit.constraint_system();
    verify_constraints(cs, &witness.into_value_vec())?;

    println!("✓ Example 3: Non-deterministic modulo verification");
    println!("  Verified: 100 = 83 × 1 + 17");
    println!("  Quotient and remainder computed via hints");
    println!("  Remainder < divisor: 0x{:016X}", bool_val);

    Ok(())
}
