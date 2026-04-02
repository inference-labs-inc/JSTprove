// Copyright 2025 Irreducible Inc.
//! Assert that a wire, interpreted as a MSB-bool, is false.
//! i.e., we are checking whether its most-significant bit is 0. all lower bits get ignored.
//!
//! Enforces `x & 0x8000000000000000 = 0` using an AND constraint.
//!
//! # Algorithm
//!
//! Uses the constraint `x ∧ 0x8000000000000000 = 0`.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x ∧ 0x8000000000000000 = 0`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, empty},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
    pathspec::PathSpec,
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::MSB_ONE],
        n_in: 1,
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
    let [msb_one] = constants else { unreachable!() };
    let [x] = inputs else { unreachable!() };

    // Constraint: x ∧ msb_one = msb_one
    builder.and().a(*x).b(*msb_one).c(empty()).build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    assertion_path: PathSpec,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam { inputs, .. } = data.gate_param();
    let [x] = inputs else { unreachable!() };
    builder.emit_assert_false(wire_to_reg(*x), assertion_path.as_u32());
}
