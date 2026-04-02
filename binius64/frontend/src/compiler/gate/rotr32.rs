// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! 32-bit half-wise rotate right.
//!
//! Returns `z = x ROTR32 n`.
//!
//! # Algorithm
//!
//! Performs independent rotate-right operations on the upper and lower 32-bit
//! halves of the input word. Bits do not cross the 32-bit lane boundary.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x ROTR32 n) ∧ all-1 = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, rotr32},
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

    builder.linear().rhs(rotr32(*x, *n)).dst(*z).build();
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
    builder.emit_rotr32(wire_to_reg(*z), wire_to_reg(*x), *n as u8);
}
