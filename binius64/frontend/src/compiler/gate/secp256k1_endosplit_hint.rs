// Copyright 2025 Irreducible Inc.
//! Secp256k1 endomorphism split
//!
//! The curve has an endomorphism `λ (x, y) = (βx, y)` where `λ³=1 (mod n)`
//! and `β³=1 (mod p)` (`n` being the scalar field modulus and `p` coordinate field one).
//!
//! For a 256-bit scalar `k` it is possible to split it into `k1` and `k2` such that
//! `k1 + λ k2 = k (mod n)` and both `k1` and `k2` are no farther than `2^128` from zero.
//!
//! The `k` scalar is represented by four 64-bit limbs in little endian order. The return value is
//! quadruple of `(k1_neg, k2_neg, k1_abs, k2_abs)` where `k1_neg` and `k2_neg` are MSB-bools
//! indicating whether `k1_abs` or `k2_abs`, respectively, should be negated. `k1_abs` and `k2_abs`
//! are at most 128 bits and are represented with two 64-bit limbs. When `k` cannot be represented
//! in this way (any valid scalar can, so it has to be modulus or above) both  `k1_abs` and `k2_abs`
//! are assigned zero values.
//!
//! This is a hint - a deterministic computation that happens only on the prover side.
//! The result should be additionally constrained by using bignum circuits to check that
//! `k1 + λ k2 = k (mod n)`.
//!
//! The method used here comes straight from libsecp256k1, follow the link for derivation:
//! <https://github.com/bitcoin-core/secp256k1/blob/master/src/scalar_impl.h#L92-L141>

use crate::compiler::{
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[],
        n_in: 4,
        n_out: 6,
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
