// Copyright 2025 Irreducible Inc.
//! BigUint modular exponentiation.
//!
//! Given `base`, `exp` and `modulus` represented by little-endian arrays of
//! 64-bit limbs returns `(base^exp) % modulus`.
//!
//! Shape is determined by number of limbs in `base`, `exp` and `modulus`.
//! There are `base.len() + exp.len() + modulus.len()` inputs and
//! `modulus.len()` outputs.
//!
//! # Algorithm
//!
//! Performs the modular exponentiation.
//!
//! # Constraints
//!
//! No constraints are generated! This is a hint - a deterministic computation that happens
//! only on the prover side. The result should be additionally constrained with bignum circuits.

use crate::compiler::{
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape(dimensions: &[usize]) -> OpcodeShape {
    let [n_base_limbs, n_exp_limbs, n_modulus_limbs] = dimensions else {
        unreachable!()
    };
    OpcodeShape {
        const_in: &[],
        n_in: *n_base_limbs + *n_exp_limbs + *n_modulus_limbs,
        n_out: *n_modulus_limbs,
        n_aux: 0,
        n_scratch: 0,
        n_imm: 0,
    }
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
    hint_id: u32,
) {
    let GateParam {
        inputs, outputs, ..
    } = data.gate_param();

    let input_regs: Vec<u32> = inputs.iter().map(|&wire| wire_to_reg(wire)).collect();

    let output_regs: Vec<u32> = outputs.iter().map(|&wire| wire_to_reg(wire)).collect();

    builder.emit_hint(hint_id, &data.dimensions, &input_regs, &output_regs);
}
