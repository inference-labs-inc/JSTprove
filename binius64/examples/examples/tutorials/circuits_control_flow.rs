// Copyright 2025 Irreducible Inc.

//! Control Flow Patterns
//!
//! This example shows:
//! - Conditional selection without branching
//! - Array access with multiplexers
//! - Loop patterns and early exit simulation
//! - Variable-length data handling
//!
//! Guide: https://www.binius.xyz/building/

use anyhow::Result;
use binius_circuits::multiplexer::{multi_wire_multiplex, single_wire_multiplex};
use binius_core::{verify::verify_constraints, word::Word};
use binius_frontend::CircuitBuilder;

fn main() -> Result<()> {
    println!("=== Control Flow Examples ===\n");

    demo_conditional_selection()?;
    demo_array_access()?;
    demo_early_exit()?;
    demo_variable_length()?;

    Ok(())
}

fn demo_conditional_selection() -> Result<()> {
    println!("1. Conditional Selection\n");

    let builder = CircuitBuilder::new();

    // Inputs
    let a = builder.add_witness();
    let b = builder.add_witness();
    let condition = builder.add_witness();

    // Both paths execute
    let path_a = builder.iadd_32(a, builder.add_constant_64(100));
    let (_, path_b) = builder.imul(b, builder.add_constant_64(2));

    // Select based on MSB of condition
    let result = builder.select(condition, path_a, path_b);

    let expected = builder.add_witness();
    builder.assert_eq("conditional_result", result, expected);

    let circuit = builder.build();

    // Test with MSB set (selects first argument)
    let mut w = circuit.new_witness_filler();
    w[a] = Word(50);
    w[b] = Word(75);
    w[condition] = Word(0x8000000000000000);
    w[expected] = Word(150);

    circuit.populate_wire_witness(&mut w)?;
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Selected path A (a + 100) when condition true");

    // Test with MSB clear (selects second argument)
    let builder2 = CircuitBuilder::new();
    let a2 = builder2.add_witness();
    let b2 = builder2.add_witness();
    let condition2 = builder2.add_witness();

    let path_a2 = builder2.iadd_32(a2, builder2.add_constant_64(100));
    let (_, path_b2) = builder2.imul(b2, builder2.add_constant_64(2));
    let result2 = builder2.select(condition2, path_a2, path_b2);

    let expected2 = builder2.add_witness();
    builder2.assert_eq("conditional_result2", result2, expected2);

    let circuit2 = builder2.build();

    let mut w2 = circuit2.new_witness_filler();
    w2[a2] = Word(50);
    w2[b2] = Word(75);
    w2[condition2] = Word(0);
    w2[expected2] = Word(150);

    circuit2.populate_wire_witness(&mut w2)?;
    let cs2 = circuit2.constraint_system();
    verify_constraints(cs2, &w2.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Selected path B (b * 2) when condition false\n");

    Ok(())
}

fn demo_array_access() -> Result<()> {
    println!("2. Dynamic Array Access\n");

    let builder = CircuitBuilder::new();

    // Create array of values
    let values = [
        builder.add_witness(),
        builder.add_witness(),
        builder.add_witness(),
        builder.add_witness(),
    ];

    // Dynamic index
    let index = builder.add_witness();

    // Use multiplexer for dynamic access
    let selected = single_wire_multiplex(&builder, &values, index);

    let expected = builder.add_witness();
    builder.assert_eq("array_access", selected, expected);

    let circuit = builder.build();

    // Test selecting index 2
    let mut w = circuit.new_witness_filler();
    w[values[0]] = Word(10);
    w[values[1]] = Word(20);
    w[values[2]] = Word(30);
    w[values[3]] = Word(40);
    w[index] = Word(2);
    w[expected] = Word(30);

    circuit.populate_wire_witness(&mut w)?;
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Selected array[2] = 30 using multiplexer");

    let builder = CircuitBuilder::new();

    let red = [builder.add_witness(), builder.add_witness()];
    let green = [builder.add_witness(), builder.add_witness()];
    let blue = [builder.add_witness(), builder.add_witness()];

    let groups = vec![red.as_slice(), green.as_slice(), blue.as_slice()];
    let selector = builder.add_witness();
    let selected_group = multi_wire_multiplex(&builder, &groups, selector);

    let expected_val = builder.add_witness();
    let expected_intensity = builder.add_witness();
    builder.assert_eq("color_value", selected_group[0], expected_val);
    builder.assert_eq("color_intensity", selected_group[1], expected_intensity);

    let circuit = builder.build();

    let mut w = circuit.new_witness_filler();
    w[red[0]] = Word(0xFF0000);
    w[red[1]] = Word(100);
    w[green[0]] = Word(0x00FF00);
    w[green[1]] = Word(200);
    w[blue[0]] = Word(0x0000FF);
    w[blue[1]] = Word(150);
    w[selector] = Word(1);
    w[expected_val] = Word(0x00FF00);
    w[expected_intensity] = Word(200);

    circuit.populate_wire_witness(&mut w)?;
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Selected green color group using multi-wire multiplexer\n");

    Ok(())
}

fn demo_early_exit() -> Result<()> {
    println!("3. Early Exit Simulation\n");

    let builder = CircuitBuilder::new();

    let data = [
        builder.add_witness(),
        builder.add_witness(),
        builder.add_witness(),
        builder.add_witness(),
    ];
    let mut sum = builder.add_constant(Word::ZERO);
    let mut found_zero = builder.add_constant(Word::ZERO);
    let zero = builder.add_constant(Word::ZERO);

    for &value in &data {
        let is_zero = builder.icmp_eq(value, zero);

        // Mask value based on found_zero flag
        let to_add = builder.select(found_zero, zero, value);
        let (new_sum, _) = builder.iadd_cin_cout(sum, to_add, zero);
        sum = new_sum;

        // Sticky flag - once set, stays set
        found_zero = builder.bor(found_zero, is_zero);
    }

    let expected = builder.add_witness();
    builder.assert_eq("sum_until_zero", sum, expected);

    let circuit = builder.build();

    let mut w = circuit.new_witness_filler();
    w[data[0]] = Word(10);
    w[data[1]] = Word(20);
    w[data[2]] = Word(0);
    w[data[3]] = Word(30);
    w[expected] = Word(30);

    circuit.populate_wire_witness(&mut w)?;
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Sum until zero: [10, 20, 0, 30] = 30\n");

    Ok(())
}

fn demo_variable_length() -> Result<()> {
    println!("4. Variable-Length Data\n");

    let builder = CircuitBuilder::new();

    // Note: The framework provides FixedByteVec for variable-length data,
    // but here we demonstrate the underlying masking pattern manually
    const MAX_SIZE: usize = 8;
    let data: [_; MAX_SIZE] = core::array::from_fn(|_| builder.add_witness());
    let actual_len = builder.add_witness();
    let mut sum = builder.add_constant(Word::ZERO);
    let zero = builder.add_constant(Word::ZERO);

    for (i, &data_wire) in data.iter().enumerate() {
        let i_wire = builder.add_constant_64(i as u64);
        let in_bounds = builder.icmp_ult(i_wire, actual_len);

        let value = builder.select(in_bounds, data_wire, zero);

        let (new_sum, _) = builder.iadd_cin_cout(sum, value, zero);
        sum = new_sum;
    }

    let expected = builder.add_witness();
    builder.assert_eq("variable_sum", sum, expected);

    let circuit = builder.build();

    let mut w = circuit.new_witness_filler();
    for i in 0..MAX_SIZE {
        w[data[i]] = Word((i + 1) as u64 * 10);
    }
    w[actual_len] = Word(3);
    w[expected] = Word(60);

    circuit.populate_wire_witness(&mut w)?;
    let cs = circuit.constraint_system();
    verify_constraints(cs, &w.into_value_vec()).map_err(|e| anyhow::anyhow!(e))?;

    println!("✓ Variable-length sum: [10,20,30,40,50,60,70,80] with len=3 = 60\n");

    println!("Circuit Statistics:");
    println!("  AND constraints: {}", cs.n_and_constraints());
    println!("  Used for comparisons and selections");

    Ok(())
}
