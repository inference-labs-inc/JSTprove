// Copyright 2026 The Binius Developers
//! 32-bit half-wise logical left shift.
//!
//! Returns `z = x SLL32 n`.
//!
//! # Algorithm
//!
//! Performs independent logical left shifts on the upper and lower 32-bit halves
//! of the input word. This exposes the core `ShiftVariant::Sll32` primitive in
//! the frontend.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x SLL32 n) ∧ all-1 = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sll32},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE],
        n_in: 1,
        n_out: 1,
        n_aux: 0,
        n_scratch: 0,
        n_imm: 1,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        inputs,
        outputs,
        imm,
        ..
    } = data.gate_param();
    let [x] = inputs else { unreachable!() };
    let [z] = outputs else { unreachable!() };
    let [n] = imm else { unreachable!() };

    builder.linear().rhs(sll32(*x, *n)).dst(*z).build();
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
        imm,
        ..
    } = data.gate_param();
    let [x] = inputs else { unreachable!() };
    let [z] = outputs else { unreachable!() };
    let [n] = imm else { unreachable!() };
    builder.emit_sll32(wire_to_reg(*z), wire_to_reg(*x), *n as u8);
}
