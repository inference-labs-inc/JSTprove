// Copyright 2025 Irreducible Inc.
//! 64-bit unsigned integer addition with carry propagation.
//!
//! # Wires
//!
//! - `a`, `b`: Input wires for the summands
//! - `cin` (carry-in): Input wire for the previous carry word. Only the MSB is used as the actual
//!   carry bit
//! - `sum`: Output wire containing the resulting sum = a + b + carry_bit
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition.
//!
//! ## Carry-out Computation
//!
//! The carry-out is computed as: `cout = (a & b) | ((a ^ b) & ¬sum)`
//!
//! For example:
//! - `0x0000000000000003 + 0x0000000000000001 = 0x0000000000000004` with `cout =
//!   0x0000000000000003` (carries at bits 0 and 1)
//! - `0xFFFFFFFFFFFFFFFF + 0x0000000000000001 = 0x0000000000000000` with `cout =
//!   0xFFFFFFFFFFFFFFFF` (carries at all bit positions)
//!
//! # Constraints
//!
//! The gate generates two AND constraints:
//!
//! 1. **Carry generation constraint**: Ensures correct carry propagation
//! 2. **Sum constraint**: Ensures the sum equals `a ^ b ^ (cout << 1) ^ cin_msb`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sll, srl, xor3, xor4},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE],
        n_in: 3,
        n_out: 2,
        n_aux: 1,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        inputs, outputs, ..
    } = data.gate_param();
    let [a, b, cin] = inputs else { unreachable!() };
    let [sum, cout] = outputs else { unreachable!() };

    let cout_sll_1 = sll(*cout, 1);
    let cin_msb = srl(*cin, 63);

    // Constraint 1: Carry propagation
    //
    // (a ⊕ (cout << 1) ⊕ cin_msb) ∧ (b ⊕ (cout << 1) ⊕ cin_msb) = cout ⊕ (cout << 1) ⊕ cin_msb
    builder
        .and()
        .a(xor3(*a, cout_sll_1, cin_msb))
        .b(xor3(*b, cout_sll_1, cin_msb))
        .c(xor3(*cout, cout_sll_1, cin_msb))
        .build();

    // Constraint 2: Sum equality (linear)
    //
    // (a ⊕ b ⊕ (cout << 1) ⊕ cin_msb) = sum
    builder
        .linear()
        .rhs(xor4(*a, *b, cout_sll_1, cin_msb))
        .dst(*sum)
        .build();
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
    let [a, b, cin] = inputs else { unreachable!() };
    let [sum, cout] = outputs else { unreachable!() };
    builder.emit_iadd_cin_cout(
        wire_to_reg(*sum),
        wire_to_reg(*cout),
        wire_to_reg(*a),
        wire_to_reg(*b),
        wire_to_reg(*cin),
    );
}
