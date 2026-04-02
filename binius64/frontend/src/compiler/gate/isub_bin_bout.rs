// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sll, srl, xor3, xor4},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE],
        n_in: 3,
        n_out: 2,
        n_aux: 1,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
    let GateParam {
        constants,
        inputs,
        outputs,
        ..
    } = data.gate_param();
    let [all_one] = constants else { unreachable!() };
    let [a, b, bin] = inputs else { unreachable!() };
    let [diff, bout] = outputs else {
        unreachable!()
    };

    let bout_sll_1 = sll(*bout, 1);
    let bin_msb = srl(*bin, 63);

    // Constraint 1: Borrow propagation
    //
    // (¬a ⊕ (bout << 1) ⊕ bin_msb) ∧ (b ⊕ (bout << 1) ⊕ bin_msb) = bout ⊕ (bout << 1) ⊕ bin_msb
    builder
        .and()
        .a(xor4(*all_one, *a, bout_sll_1, bin_msb))
        .b(xor3(*b, bout_sll_1, bin_msb))
        .c(xor3(*bout, bout_sll_1, bin_msb))
        .build();

    // Constraint 2: Diff equality (linear)
    //
    // (a ⊕ b ⊕ (bout << 1) ⊕ bin_msb) = diff
    builder
        .linear()
        .rhs(xor4(*a, *b, bout_sll_1, bin_msb))
        .dst(*diff)
        .build();
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
    let [a, b, bin] = inputs else { unreachable!() };
    let [diff, bout] = outputs else {
        unreachable!()
    };
    builder.emit_isub_bin_bout(
        wire_to_reg(*diff),
        wire_to_reg(*bout),
        wire_to_reg(*a),
        wire_to_reg(*b),
        wire_to_reg(*bin),
    );
}
