// Copyright 2025 Irreducible Inc.
//! Imul gate implements 64-bit × 64-bit → 128-bit unsigned multiplication.
//! Uses the MulConstraint: X * Y = (HI << 64) | LO

use crate::compiler::{
    constraint_builder::ConstraintBuilder,
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[],
        n_in: 2,
        n_out: 2,
        n_aux: 0,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        inputs, outputs, ..
    } = data.gate_param();
    let [x, y] = inputs else { unreachable!() };
    let [hi, lo] = outputs else { unreachable!() };

    // Create MulConstraint: X * Y = (HI << 64) | LO
    builder.mul().a(*x).b(*y).hi(*hi).lo(*lo).build();
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
    let [x, y] = inputs else { unreachable!() };
    let [hi, lo] = outputs else { unreachable!() };
    builder.emit_imul(
        wire_to_reg(*hi),
        wire_to_reg(*lo),
        wire_to_reg(*x),
        wire_to_reg(*y),
    );
}
