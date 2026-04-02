// Copyright 2025 Irreducible Inc.
//! Assert that a wire isn't zero.
//!
//! Enforces `x ≠ 0`.
//!
//! # Algorithm
//!
//! The idea is similar to `icmp_eq`, but actually simpler.
//! First off, we only have one operand, not two;
//! secondly, we don't need to negate the MSB of the result.
//!
//! The gate exploits the property that when adding `all-1` to a value:
//! - If the value is 0: `0 + all-1 = all-1` with no carry out (MSB of cout = 0)
//! - If the value is non-zero: `value + all-1` wraps around with carry out (MSB of cout = 1)
//!
//! The algorithm is as follows:
//! 1. Compute carry bits `cout` from `x + all-1` using the constraint: `(x ⊕ cin) ∧ (all-1 ⊕ cin) =
//!    cin ⊕ cout` where `cin = cout << 1`
//! 2. The MSB of `cout` tells us whether x ≠ 0; i.e.,
//!    - MSB = 0: no carry out, meaning `x = 0`
//!    - MSB = 1: carry out occurred, meaning `x ≠ 0`
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sll, xor2},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
    pathspec::PathSpec,
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE, Word::ZERO], // Need zero constant for cin
        n_in: 1,
        n_out: 0,
        n_aux: 1,
        n_scratch: 1,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        constants,
        inputs,
        aux,
        ..
    } = data.gate_param();
    let [all_one, _zero] = constants else {
        unreachable!()
    };
    let [x] = inputs else { unreachable!() };
    let [cout] = aux else { unreachable!() };

    let cin = sll(*cout, 1);

    // Constraint 1: Constrain carry-out
    // (x ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout
    builder
        .and()
        .a(xor2(*x, cin))
        .b(xor2(*all_one, cin))
        .c(xor2(cin, *cout))
        .build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    assertion_path: PathSpec,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam {
        constants,
        inputs,
        aux,
        scratch,
        ..
    } = data.gate_param();
    let [all_one, zero] = constants else {
        unreachable!()
    };
    let [x] = inputs else { unreachable!() };
    let [cout] = aux else { unreachable!() };
    let [scratch_sum_unused] = scratch else {
        unreachable!()
    };

    // Compute carry bits from all_one + diff
    builder.emit_iadd_cin_cout(
        wire_to_reg(*scratch_sum_unused), // sum (unused)
        wire_to_reg(*cout),               // cout
        wire_to_reg(*all_one),            // all_one
        wire_to_reg(*x),                  // x
        wire_to_reg(*zero),               // cin = 0
    );

    builder.emit_assert_non_zero(wire_to_reg(*cout), assertion_path.as_u32());
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use rand::{RngCore, SeedableRng, rngs::StdRng};

    use crate::compiler::CircuitBuilder;

    #[test]
    fn test_assert_non_zero_basic() {
        // Build a circuit with assert_non_zero gate
        let builder = CircuitBuilder::new();
        let x = builder.add_inout();
        builder.assert_non_zero("non_zero", x);
        let circuit = builder.build();

        // Test specific non-zero cases
        let test_cases = [
            1_u64,
            0xFFFFFFFFFFFFFFFF_u64,
            0x1234567890ABCDEF_u64,
            0x8000000000000000_u64,
            0x0000000000000001_u64,
        ];

        for x_val in test_cases {
            let mut w = circuit.new_witness_filler();
            w[x] = Word(x_val);
            w.circuit.populate_wire_witness(&mut w).unwrap();

            // Verify constraints pass for non-zero values
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_assert_non_zero_random() {
        // Build a circuit with assert_non_zero gate
        let builder = CircuitBuilder::new();
        let x = builder.add_inout();
        builder.assert_non_zero("non_zero", x);
        let circuit = builder.build();

        // Test with random non-zero values
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let mut x_val = rng.next_u64();
            // Ensure we don't test with zero
            if x_val == 0 {
                x_val = 1;
            }

            let mut w = circuit.new_witness_filler();
            w[x] = Word(x_val);
            w.circuit.populate_wire_witness(&mut w).unwrap();

            // Verify constraints pass
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "Word(0x0000000000000000) == 0")]
    fn test_assert_non_zero_fails_on_zero() {
        // Build a circuit with assert_non_zero gate
        let builder = CircuitBuilder::new();
        let x = builder.add_inout();
        builder.assert_non_zero("non_zero", x);
        let circuit = builder.build();

        // Test with zero value (should panic)
        let mut w = circuit.new_witness_filler();
        w[x] = Word(0);
        // This should panic when trying to assert non-zero on zero
        w.circuit.populate_wire_witness(&mut w).unwrap();
    }
}
