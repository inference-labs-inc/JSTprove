// Copyright 2025 Irreducible Inc.
//! Fused AND-XOR operation.
//!
//! Returns `z = (x & y) ^ w`.
//!
//! # Algorithm
//!
//! Computes the bitwise AND of two words followed by XOR with a third word.
//! This common pattern is fused into a single gate for efficiency.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x & y = t` where `t ^ w = z`

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, xor2},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[],
        n_in: 3,
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
    let [x, y, w] = inputs else { unreachable!() };
    let [z] = outputs else { unreachable!() };

    // Constraint: Fused AND-XOR
    //
    // x & y = t, where t ^ w = z
    // This can be written as: x & y ^ w = z
    builder.and().a(*x).b(*y).c(xor2(*z, *w)).build();
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
    let [x, y, w] = inputs else { unreachable!() };
    let [z] = outputs else { unreachable!() };

    builder.emit_fax(
        wire_to_reg(*z),
        wire_to_reg(*x),
        wire_to_reg(*y),
        wire_to_reg(*w),
    );
}
