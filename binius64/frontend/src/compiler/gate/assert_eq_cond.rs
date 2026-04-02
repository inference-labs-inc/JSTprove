// Copyright 2025 Irreducible Inc.
//! Conditional equality assertion.
//!
//! Enforces `x = y` when the MSB-bool value of `cond` is true, and no constraint otherwise.
//!
//! # Algorithm
//!
//! Uses a mask to conditionally enforce equality: `(x ^ y) & (cond ~>> 63) = 0`.
//! When `cond` is MSB-bool-true, this enforces `x = y`. otherwise, the constraint is satisfied
//! trivially.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x ⊕ y) ∧ (cond ~>> 63) = 0`

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, empty, sar, xor2},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
    pathspec::PathSpec,
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[],
        n_in: 3,
        n_out: 0,
        n_aux: 0,
        n_scratch: 1,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam { inputs, .. } = data.gate_param();
    let [x, y, cond] = inputs else { unreachable!() };

    // Constraint: (x ⊕ y) ∧ (cond ~>> 63) = 0
    let mask = sar(*cond, 63);
    builder.and().a(xor2(*x, *y)).b(mask).c(empty()).build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    assertion_path: PathSpec,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam {
        inputs, scratch, ..
    } = data.gate_param();
    let [x, y, cond] = inputs else { unreachable!() };
    let [mask] = scratch else { unreachable!() };

    // Broadcast MSB: mask = cond >> 63 (arithmetic)
    builder.emit_sar(wire_to_reg(*mask), wire_to_reg(*cond), 63);

    builder.emit_assert_eq_cond(
        wire_to_reg(*mask),
        wire_to_reg(*x),
        wire_to_reg(*y),
        assertion_path.as_u32(),
    );
}
