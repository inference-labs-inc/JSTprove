// Copyright 2025 Irreducible Inc.
//! Equality assertion.
//!
//! Enforces `x = y` using an AND constraint.
//!
//! # Algorithm
//!
//! Uses the property that `x = y` iff `x ^ y = 0`.
//! This is enforced as `(x ⊕ y) ∧ all-1 = 0`.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x ⊕ y) ∧ all-1 = 0`
use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, empty, xor2},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
    pathspec::PathSpec,
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE],
        n_in: 2,
        n_out: 0,
        n_aux: 0,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        constants, inputs, ..
    } = data.gate_param();
    let [all_one] = constants else { unreachable!() };
    let [x, y] = inputs else { unreachable!() };

    // Constraint: (x ⊕ y) ∧ all-1 = 0
    builder.and().a(xor2(*x, *y)).b(*all_one).c(empty()).build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    assertion_path: PathSpec,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam { inputs, .. } = data.gate_param();
    let [x, y] = inputs else { unreachable!() };
    builder.emit_assert_eq(wire_to_reg(*x), wire_to_reg(*y), assertion_path.as_u32());
}
