// Copyright 2025 Irreducible Inc.
//! 64-bit rotate right.
//!
//! Returns `z = ((x >> n) | (x << (64 - n))) & MASK_64`
//!
//! # Algorithm
//!
//! Rotates a 64-bit value right by `n` positions:
//! 1. Shift right by n: `t1 = x >> n`
//! 2. Shift left by 64-n: `t2 = x << (64-n)`
//! 3. Combine with XOR: Since the shifted ranges don't overlap, `t1 | t2 = t1 ^ t2`
//! 4. Mask to 64 bits: `z = (t1 ^ t2) & MASK_64`
//!
//! The non-overlapping property is crucial: right-shifted bits occupy positions 0-(63-n),
//! while left-shifted bits occupy positions (64-n)-63, with no overlap.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `((x >> n) ⊕ (x << (64-n))) ∧ MASK_64 = z`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, rotr},
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

    // Constraint: Rotate right (linear)
    // rotr(x, n) = z
    builder.linear().rhs(rotr(*x, *n)).dst(*z).build();
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
    builder.emit_rotr(wire_to_reg(*z), wire_to_reg(*x), *n as u8);
}
