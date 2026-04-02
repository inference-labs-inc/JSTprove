// Copyright 2025 Irreducible Inc.

//! Gadgets Example
//!
//! This example shows:
//! - Byte swapping gadgets for endianness conversion
//! - Popcount gadget for counting 1-bits
//! - How gadgets compose multiple gates internally
//!
//! Guide: https://www.binius.xyz/building/

use anyhow::Result;
use binius_circuits::{bytes, popcount};
use binius_core::{verify::verify_constraints, word::Word};
use binius_frontend::CircuitBuilder;

fn main() -> Result<()> {
    // Example 1: Byte swapping gadgets
    println!("=== Byte Swapping Gadgets ===\n");
    demo_byte_swapping()?;

    // Example 2: Popcount gadget
    println!("\n=== Popcount Gadget ===\n");
    demo_popcount()?;

    // Example 3: Combined gadget usage
    println!("\n=== Combined Gadget Usage ===\n");
    demo_combined()?;

    Ok(())
}

fn demo_byte_swapping() -> Result<()> {
    let builder = CircuitBuilder::new();

    // Input value: 0x0123456789ABCDEF
    let input = builder.add_witness();

    // Apply swap_bytes_32 gadget - swaps bytes within 32-bit halves
    // This uses ~8 gates internally (masks, shifts, XORs)
    let swapped_32 = bytes::swap_bytes_32(&builder, input);

    // Apply full swap_bytes gadget - complete byte reversal
    // This uses ~12 gates internally (calls swap_bytes_32 + rotation)
    let swapped_full = bytes::swap_bytes(&builder, input);

    // Expected outputs
    let expected_32 = builder.add_witness();
    let expected_full = builder.add_witness();

    builder.assert_eq("swap_32_result", swapped_32, expected_32);
    builder.assert_eq("swap_full_result", swapped_full, expected_full);

    let circuit = builder.build();

    // Fill witness values
    let mut w = circuit.new_witness_filler();
    w[input] = Word(0x0123456789ABCDEF);

    // swap_bytes_32: bytes within each half are reversed
    // 0x01234567_89ABCDEF -> 0x67452301_EFCDAB89
    w[expected_32] = Word(0x67452301EFCDAB89);

    // swap_bytes: complete reversal
    // 0x0123456789ABCDEF -> 0xEFCDAB8967452301
    w[expected_full] = Word(0xEFCDAB8967452301);

    circuit.populate_wire_witness(&mut w)?;

    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Byte swapping verified");
    println!("  Input:            0x{:016X}", 0x0123456789ABCDEFu64);
    println!(
        "  swap_bytes_32:    0x{:016X} (swapped within 32-bit halves)",
        0x67452301EFCDAB89u64
    );
    println!(
        "  swap_bytes:       0x{:016X} (fully reversed)",
        0xEFCDAB8967452301u64
    );

    // Show gate usage
    println!("\nGate usage for byte swapping:");
    println!("  AND constraints: {}", cs.n_and_constraints());
    println!("  MUL constraints: {}", cs.n_mul_constraints());

    Ok(())
}

fn demo_popcount() -> Result<()> {
    let mut builder = CircuitBuilder::new();

    // Test several input values
    let test_cases = vec![
        (0xFF, 8),                // 8 bits set
        (0x5555555555555555, 32), // Every other bit
        (0xFFFFFFFFFFFFFFFF, 64), // All bits set
        (0x0, 0),                 // No bits set
        (0x8000000000000001, 2),  // Just MSB and LSB
    ];

    for (input_val, expected_count) in test_cases {
        let input = builder.add_witness();
        let output = builder.add_witness();

        // Apply popcount gadget
        // Uses SWAR (SIMD Within A Register) algorithm
        // Composes ~20 gates internally
        let count = popcount::popcount(&mut builder, input);

        builder.assert_eq(format!("popcount_{:016X}", input_val), count, output);

        let circuit = builder.build();

        let mut w = circuit.new_witness_filler();
        w[input] = Word(input_val);
        w[output] = Word(expected_count);

        circuit.populate_wire_witness(&mut w)?;

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

        println!("✓ Popcount(0x{:016X}) = {} bits", input_val, expected_count);

        // Reset builder for next test
        builder = CircuitBuilder::new();
    }

    // Show gate usage for one popcount
    let input = builder.add_witness();
    let _count = popcount::popcount(&mut builder, input);
    let circuit = builder.build();
    let cs = circuit.constraint_system();

    println!("\nGate usage for single popcount:");
    println!("  AND constraints: {}", cs.n_and_constraints());
    println!("  Uses SWAR algorithm for parallel bit counting");

    Ok(())
}

fn demo_combined() -> Result<()> {
    let mut builder = CircuitBuilder::new();

    // Combine multiple gadgets in a single circuit
    // Example: Count bits in byte-swapped value

    let input = builder.add_witness();

    // First swap the bytes
    let swapped = bytes::swap_bytes(&builder, input);

    // Then count the bits
    let count = popcount::popcount(&mut builder, swapped);

    let expected_count = builder.add_witness();
    builder.assert_eq("combined_result", count, expected_count);

    let circuit = builder.build();

    // Test with value that has different bit patterns in each byte
    let test_val = 0xFF00FF0000FF00FF; // 32 bits set

    let mut w = circuit.new_witness_filler();
    w[input] = Word(test_val);
    w[expected_count] = Word(32); // Bit count doesn't change with byte swapping

    circuit.populate_wire_witness(&mut w)?;

    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Combined gadgets:");
    println!("  Input:        0x{:016X}", test_val);
    println!("  After swap:   0x{:016X}", test_val.swap_bytes());
    println!("  Bit count:    {} (unchanged by byte swap)", 32);

    println!("\nTotal gates for combined circuit:");
    println!("  AND constraints: {}", cs.n_and_constraints());
    println!("  = byte swap (~12) + popcount (~20) gates");

    Ok(())
}
