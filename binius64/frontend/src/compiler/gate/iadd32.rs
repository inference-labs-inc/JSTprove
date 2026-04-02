// Copyright 2025 Irreducible Inc.
//! 32-bit unsigned integer addition with carry propagation.
//!
//! Returns `z = (x + y) & MASK_32` and `cout` containing carry bits.
//!
//! # Algorithm
//!
//! Performs 32-bit addition by computing the full 64-bit result and masking:
//! 1. Compute carry bits `cout` from `x + y` using carry propagation
//! 2. Extract the lower 32 bits: `z = (x ⊕ y ⊕ (cout << 1)) ∧ MASK_32`
//!
//! # Constraints
//!
//! The gate generates 2 AND constraints:
//! 1. Carry propagation: `(x ⊕ (cout << 1)) ∧ (y ⊕ (cout << 1)) = cout ⊕ (cout << 1)`
//! 2. Result masking: `(x ⊕ y ⊕ (cout << 1)) ∧ MASK_32 = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sll, xor2, xor3},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::MASK_32],
        n_in: 2,
        n_out: 1,
        n_aux: 1,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        constants,
        inputs,
        outputs,
        aux,
        ..
    } = data.gate_param();
    let [mask32] = constants else { unreachable!() };
    let [x, y] = inputs else { unreachable!() };
    let [z] = outputs else { unreachable!() };
    let [cout] = aux else { unreachable!() };

    let cout_sll_1 = sll(*cout, 1);

    // Constraint 1: Carry propagation
    //
    // (x ⊕ (cout << 1)) ∧ (y ⊕ (cout << 1)) = cout ⊕ (cout << 1)
    builder
        .and()
        .a(xor2(*x, cout_sll_1))
        .b(xor2(*y, cout_sll_1))
        .c(xor2(*cout, cout_sll_1))
        .build();

    // Constraint 2: Result masking
    //
    // (x ⊕ y ⊕ (cout << 1)) ∧ MASK_32 = z
    builder
        .and()
        .a(xor3(*x, *y, cout_sll_1))
        .b(*mask32)
        .c(*z)
        .build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam {
        inputs,
        outputs,
        aux,
        ..
    } = data.gate_param();
    let [a, b] = inputs else { unreachable!() };
    let [sum] = outputs else { unreachable!() };
    let [cout] = aux else { unreachable!() };
    builder.emit_iadd_cout32(
        wire_to_reg(*sum),
        wire_to_reg(*cout),
        wire_to_reg(*a),
        wire_to_reg(*b),
    );
}
