// Copyright 2025 Irreducible Inc.
//! Arithmetic right shift.
//!
//! Returns `z = x SAR n`.
//!
//! # Algorithm
//!
//! Performs an arithmetic right shift by `n` bits. The constraint system allows
//! referencing shifted versions of values directly without additional gates.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x SAR n) ∧ all-1 = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sar},
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

    // Constraint: Arithmetic right shift (linear)
    // (x SAR n) = z
    builder.linear().rhs(sar(*x, *n)).dst(*z).build();
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
    builder.emit_sar(wire_to_reg(*z), wire_to_reg(*x), *n as u8);
}
