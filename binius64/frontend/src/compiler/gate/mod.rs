// Copyright 2025 Irreducible Inc.
use crate::compiler::{
    constraint_builder::ConstraintBuilder,
    eval_form::BytecodeBuilder,
    gate_graph::{Gate, GateData, GateGraph},
    hints::{
        BigUintDivideHint, BigUintModPowHint, HintRegistry, ModInverseHint, Secp256k1EndosplitHint,
    },
};

pub mod opcode;

pub use opcode::Opcode;

pub mod assert_eq;
pub mod assert_eq_cond;
pub mod assert_false;
pub mod assert_non_zero;
pub mod assert_true;
pub mod assert_zero;
pub mod band;
pub mod biguint_divide_hint;
pub mod biguint_mod_pow_hint;
pub mod bor;
pub mod bxor;
pub mod bxor_multi;
pub mod fax;
pub mod iadd32;
pub mod iadd_cin_cout;
pub mod icmp_eq;
pub mod icmp_ult;
pub mod imul;
pub mod isub_bin_bout;
pub mod mod_inverse_hint;
pub mod rotr;
pub mod rotr32;
pub mod sar;
pub mod secp256k1_endosplit_hint;
pub mod select;
pub mod shl;
pub mod shr;
pub mod sll32;
pub mod smul;
pub mod sra32;
pub mod srl32;

pub fn constrain(gate: Gate, graph: &GateGraph, builder: &mut ConstraintBuilder) {
    let data = &graph.gates[gate];
    match data.opcode {
        Opcode::Band => band::constrain(gate, data, builder),
        Opcode::Bxor => bxor::constrain(gate, data, builder),
        Opcode::BxorMulti => bxor_multi::constrain(gate, data, builder),
        Opcode::Bor => bor::constrain(gate, data, builder),
        Opcode::Fax => fax::constrain(gate, data, builder),
        Opcode::Select => select::constrain(gate, data, builder),
        Opcode::IaddCinCout => iadd_cin_cout::constrain(gate, data, builder),
        Opcode::Iadd32 => iadd32::constrain(gate, data, builder),
        Opcode::IsubBinBout => isub_bin_bout::constrain(gate, data, builder),
        Opcode::Sll32 => sll32::constrain(gate, data, builder),
        Opcode::Srl32 => srl32::constrain(gate, data, builder),
        Opcode::Rotr32 => rotr32::constrain(gate, data, builder),
        Opcode::Sra32 => sra32::constrain(gate, data, builder),
        Opcode::Rotr => rotr::constrain(gate, data, builder),
        Opcode::AssertEq => assert_eq::constrain(gate, data, builder),
        Opcode::AssertZero => assert_zero::constrain(gate, data, builder),
        Opcode::AssertNonZero => assert_non_zero::constrain(gate, data, builder),
        Opcode::AssertFalse => assert_false::constrain(gate, data, builder),
        Opcode::AssertTrue => assert_true::constrain(gate, data, builder),
        Opcode::AssertEqCond => assert_eq_cond::constrain(gate, data, builder),
        Opcode::Imul => imul::constrain(gate, data, builder),
        Opcode::Smul => smul::constrain(gate, data, builder),
        Opcode::IcmpUlt => icmp_ult::constrain(gate, data, builder),
        Opcode::IcmpEq => icmp_eq::constrain(gate, data, builder),
        Opcode::Shr => shr::constrain(gate, data, builder),
        Opcode::Shl => shl::constrain(gate, data, builder),
        Opcode::Sar => sar::constrain(gate, data, builder),
        // Hints do not introduce constraints
        Opcode::BigUintDivideHint => (),
        Opcode::BigUintModPowHint => (),
        Opcode::ModInverseHint => (),
        Opcode::Secp256k1EndosplitHint => (),
    }
}

/// Emit bytecode for a single gate
pub fn emit_gate_bytecode(
    gate: Gate,
    data: &GateData,
    graph: &GateGraph,
    builder: &mut BytecodeBuilder,
    wire_to_reg: impl Fn(crate::compiler::gate_graph::Wire) -> u32 + Copy,
    hint_registry: &mut HintRegistry,
) {
    match data.opcode {
        Opcode::Band => band::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Bxor => bxor::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::BxorMulti => bxor_multi::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Bor => bor::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Fax => fax::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Select => select::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::IaddCinCout => iadd_cin_cout::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Iadd32 => iadd32::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::IsubBinBout => isub_bin_bout::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Sll32 => sll32::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Srl32 => srl32::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Rotr32 => rotr32::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Sra32 => sra32::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Rotr => rotr::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::AssertEq => {
            let assertion_path = graph.assertion_names[gate];
            assert_eq::emit_eval_bytecode(gate, data, assertion_path, builder, wire_to_reg)
        }
        Opcode::AssertZero => {
            let assertion_path = graph.assertion_names[gate];
            assert_zero::emit_eval_bytecode(gate, data, assertion_path, builder, wire_to_reg)
        }
        Opcode::AssertNonZero => {
            let assertion_path = graph.assertion_names[gate];
            assert_non_zero::emit_eval_bytecode(gate, data, assertion_path, builder, wire_to_reg)
        }
        Opcode::AssertEqCond => {
            let assertion_path = graph.assertion_names[gate];
            assert_eq_cond::emit_eval_bytecode(gate, data, assertion_path, builder, wire_to_reg)
        }
        Opcode::AssertFalse => {
            let assertion_path = graph.assertion_names[gate];
            assert_false::emit_eval_bytecode(gate, data, assertion_path, builder, wire_to_reg)
        }
        Opcode::AssertTrue => {
            let assertion_path = graph.assertion_names[gate];
            assert_true::emit_eval_bytecode(gate, data, assertion_path, builder, wire_to_reg)
        }
        Opcode::Imul => imul::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Smul => smul::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::IcmpUlt => icmp_ult::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::IcmpEq => icmp_eq::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Shr => shr::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Shl => shl::emit_eval_bytecode(gate, data, builder, wire_to_reg),
        Opcode::Sar => sar::emit_eval_bytecode(gate, data, builder, wire_to_reg),

        // Hint-based gates
        Opcode::ModInverseHint => {
            let hint_id = hint_registry.register(Box::new(ModInverseHint::new()));
            mod_inverse_hint::emit_eval_bytecode(gate, data, builder, wire_to_reg, hint_id)
        }
        Opcode::BigUintModPowHint => {
            let hint_id = hint_registry.register(Box::new(BigUintModPowHint::new()));
            biguint_mod_pow_hint::emit_eval_bytecode(gate, data, builder, wire_to_reg, hint_id)
        }
        Opcode::BigUintDivideHint => {
            let hint_id = hint_registry.register(Box::new(BigUintDivideHint::new()));
            biguint_divide_hint::emit_eval_bytecode(gate, data, builder, wire_to_reg, hint_id)
        }
        Opcode::Secp256k1EndosplitHint => {
            let hint_id = hint_registry.register(Box::new(Secp256k1EndosplitHint::new()));
            secp256k1_endosplit_hint::emit_eval_bytecode(gate, data, builder, wire_to_reg, hint_id)
        }
    }
}
