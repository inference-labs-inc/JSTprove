// Copyright 2025 Irreducible Inc.
//! BigUint division.
//!
//! Given `dividend` and `divisor`, returns `(quotient, remainder)`, numbers represented
//! by little-endian arrays of 64-bit limbs. It holds that `quotient.len() == dividend.len()`
//! and `remainder.len() == divisor.len()`.
//!
//! Shape is determined by the number of limbs in `dividend` and `divisor`.
//! There are `dividend.len() + divisor.len()` inputs & outputs.
//!
//! # Algorithm
//!
//! Performs the long division. Returns zero quotient & remainder in case of division by zero.
//!
//! # Constraints
//!
//! No constraints are generated! This is a hint - a deterministic computation that happens only
//! on the prover side. The result should be additionally constrained by checking that
//! `remainder + divisor * quotient == dividend` using bignum circuits.

use crate::compiler::{
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape(dimensions: &[usize]) -> OpcodeShape {
    let [dividend_limbs_len, divisor_limbs_len] = dimensions else {
        unreachable!()
    };
    OpcodeShape {
        const_in: &[],
        n_in: *dividend_limbs_len + *divisor_limbs_len,
        n_out: *dividend_limbs_len + *divisor_limbs_len,
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
