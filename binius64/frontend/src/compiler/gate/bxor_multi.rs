// Copyright 2025 Irreducible Inc.
//! N-way bitwise XOR operation.
//!
//! Returns `z = x0 ^ x1 ^ ... ^ xn`.
//!
//! # Constraints
//!
//! The gate generates 1 linear constraint:
//! - `x0 ⊕ x1 ⊕ ... ⊕ xn = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, WireExprTerm, xor_multi},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape(dimensions: &[usize]) -> OpcodeShape {
    let [n_inputs] = dimensions else {
        unreachable!()
    };
    OpcodeShape {
        const_in: &[Word::ALL_ONE],
        n_in: *n_inputs,
        n_out: 1,
        n_aux: 0,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        inputs, outputs, ..
    } = data.gate_param();
    let [z] = outputs else { unreachable!() };

    // Constraint: N-way Bitwise XOR (linear)
    //
    // (x0 ⊕ x1 ⊕ ... ⊕ xn) = z
    let terms: Vec<WireExprTerm> = inputs.iter().map(|&w| w.into()).collect();
    builder.linear().rhs(xor_multi(terms)).dst(*z).build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam {
        inputs, outputs, ..
    } = data.gate_param();
    let [z] = outputs else { unreachable!() };

    let input_regs: Vec<u32> = inputs.iter().map(|&wire| wire_to_reg(wire)).collect();
    builder.emit_bxor_multi(wire_to_reg(*z), &input_regs);
}

#[cfg(test)]
mod tests {
    use binius_core::word::Word;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::compiler::CircuitBuilder;

    #[test]
    fn test_bxor_multi() {
        // Test the n-way XOR gate with different input sizes
        let builder = CircuitBuilder::new();

        // Test with 3 inputs
        let a = builder.add_inout();
        let b = builder.add_inout();
        let c = builder.add_inout();
        let result_3 = builder.bxor_multi(&[a, b, c]);
        let expected_3 = builder.add_inout();
        builder.assert_eq("xor3", result_3, expected_3);

        // Test with 4 inputs
        let d = builder.add_inout();
        let result_4 = builder.bxor_multi(&[a, b, c, d]);
        let expected_4 = builder.add_inout();
        builder.assert_eq("xor4", result_4, expected_4);

        // Test with 5 inputs
        let e = builder.add_inout();
        let result_5 = builder.bxor_multi(&[a, b, c, d, e]);
        let expected_5 = builder.add_inout();
        builder.assert_eq("xor5", result_5, expected_5);

        let circuit = builder.build();

        // Test with random values
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..1000 {
            let mut w = circuit.new_witness_filler();
            w[a] = Word(rng.random::<u64>());
            w[b] = Word(rng.random::<u64>());
            w[c] = Word(rng.random::<u64>());
            w[d] = Word(rng.random::<u64>());
            w[e] = Word(rng.random::<u64>());

            // Expected results
            w[expected_3] = Word(w[a].0 ^ w[b].0 ^ w[c].0);
            w[expected_4] = Word(w[a].0 ^ w[b].0 ^ w[c].0 ^ w[d].0);
            w[expected_5] = Word(w[a].0 ^ w[b].0 ^ w[c].0 ^ w[d].0 ^ w[e].0);

            w.circuit.populate_wire_witness(&mut w).unwrap();
        }
    }

    #[test]
    fn test_bxor_multi_edge_cases() {
        let builder = CircuitBuilder::new();

        // Test with single input (should return the input itself)
        let single = builder.add_inout();
        let result_single = builder.bxor_multi(&[single]);
        assert_eq!(result_single, single, "Single input should return itself");

        // Test with two inputs (should use regular bxor)
        let a = builder.add_inout();
        let b = builder.add_inout();
        let result_2 = builder.bxor_multi(&[a, b]);
        let expected_2 = builder.add_inout();
        builder.assert_eq("xor2", result_2, expected_2);

        let circuit = builder.build();

        // Verify two-input case works correctly
        let mut rng = StdRng::seed_from_u64(456);
        for _ in 0..100 {
            let mut w = circuit.new_witness_filler();
            w[a] = Word(rng.random::<u64>());
            w[b] = Word(rng.random::<u64>());
            w[expected_2] = Word(w[a].0 ^ w[b].0);
            w[single] = Word(rng.random::<u64>());

            w.circuit.populate_wire_witness(&mut w).unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "bxor_multi requires at least one input")]
    fn test_bxor_multi_empty_panic() {
        let builder = CircuitBuilder::new();
        builder.bxor_multi(&[]);
    }
}
