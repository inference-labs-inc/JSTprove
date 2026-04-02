// Copyright 2025 Irreducible Inc.
//! BigUint modular inverse
//!
//! Given `base` and `modulus` represented by little-endian arrays of 64-bit limbs returns
//! `(quotient, inverse)` such as `base * inverse = 1 + quotient * modulus`.
//! If `base` and `modulus` are not coprime then both `quotient` and `inverse` are set to zero.
//!
//! Shape is determined by number of limbs in `base` and `modulus`.
//! There are `base.len() + modulus.len()` inputs and `2 * modulus.len()` outputs.
//!
//! # Algorithm
//!
//! Performs the extended Euclidean algorithm.
//!
//! # Constraints
//!
//! No constraints are generated! This is a hint - a deterministic computation that happens only
//! on the prover side. The result should be additionally constrained by checking that
//! `base * inverse = 1 + quotient * modulus` using bignum circuits.

use crate::compiler::{
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape(dimensions: &[usize]) -> OpcodeShape {
    let [base_limbs_len, modulus_limbs_len] = dimensions else {
        unreachable!()
    };
    OpcodeShape {
        const_in: &[],
        n_in: *base_limbs_len + *modulus_limbs_len,
        n_out: 2 * *modulus_limbs_len,
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
