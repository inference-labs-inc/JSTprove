// Copyright 2025 Irreducible Inc.
//! Bitwise XOR operation.
//!
//! Returns `z = x ^ y`.
//!
//! # Algorithm
//!
//! Computes the bitwise XOR using a linear constraint.
//!
//! # Constraints
//!
//! The gate generates 1 linear constraint:
//! - `x ⊕ y = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, xor2},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE],
        n_in: 2,
        n_out: 1,
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
    let [z] = outputs else { unreachable!() };

    // Constraint: Bitwise XOR (linear)
    //
    // (x ⊕ y) = z
    builder.linear().rhs(xor2(*x, *y)).dst(*z).build();
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
    let [z] = outputs else { unreachable!() };

    builder.emit_bxor(wire_to_reg(*z), wire_to_reg(*x), wire_to_reg(*y));
}
