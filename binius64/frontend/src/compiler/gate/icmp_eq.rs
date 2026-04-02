// Copyright 2025 Irreducible Inc.
//! 64-bit equality test that returns an MSB-bool indicating equality.
//!
//! Returns a wire whose value, as an MSB-bool, is true if `x == y` and false otherwise.
//! It is undefined what the NON-most-significant bits of the output wire will be.
//!
//! # Algorithm
//!
//! The gate exploits the property that when adding `all-1` to a value:
//! - If the value is 0: `0 + all-1 = all-1` with no carry out (MSB of cout = 0)
//! - If the value is non-zero: `value + all-1` wraps around with carry out (MSB of cout = 1)
//!
//! 1. Compute `diff = x ⊕ y` (which is 0 iff x == y)
//! 2. Compute carry bits `cout` from `diff + all-1` using the constraint: `(x ⊕ y ⊕ cin) ∧ (all-1 ⊕
//!    cin) = cin ⊕ cout` where `cin = cout << 1`
//! 3. The MSB of `cout` indicates the comparison result:
//!    - MSB = 0: no carry out, meaning `diff = 0`, so `x == y`
//!    - MSB = 1: carry out occurred, meaning `diff ≠ 0`, so `x ≠ y`
//! 4. Negate the MSB: `out_wire ≔ cout ⊕ 0x8000000000000000`.
//!
//! we do this in a slightly more roundabout way, in order to make things work with the builder.
//! since `out_wire` and `cout` will differ only in their MSB, it's valid to take
//! `cin ≔ out_wire << 1`, instead of `cin ≔ cout << 1`. thus, we never materialize cout.
//! in the one place we need it, we inline the expression `out_wire ⊕ 0x8000000000000000` instead.
//!
//! of course, in theory, we could have instead done out_wire ≔ cout ⊕ 0xFFFFFFFFFFFFFFFF.
//! this still would have been correct, as far as the MSB of `out_wire` was concerned.
//! the problem with doing that is that we then would have had to put `cin ≔ ¬out-wire << 1`.
//! doubling up on the negation and the shift caused a problem, so this works out quite nicely.
//!
//! # Constraints
//!
//! The gate generates one AND constraint:
//! 1. Carry propagation: `(x ⊕ y ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout`

use binius_core::word::Word;

use crate::compiler::{
    constraint_builder::{ConstraintBuilder, sll, xor2, xor3},
    gate::opcode::OpcodeShape,
    gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
    OpcodeShape {
        const_in: &[Word::ALL_ONE, Word::MSB_ONE, Word::ZERO], // Need zero constant for cin
        n_in: 2,
        n_out: 1,
        n_aux: 1,     // carry-out register used in eval bytecode
        n_scratch: 2, // Need 2 scratch registers for intermediate computations
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
    let [all_one, msb_one, _zero] = constants else {
        unreachable!()
    };
    let [x, y] = inputs else { unreachable!() };
    let [out_wire] = outputs else { unreachable!() };

    let cin = sll(*out_wire, 1);

    // Constraint 1: Constrain carry-out
    // (x ⊕ y ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout
    builder
        .and()
        .a(xor3(*x, *y, cin))
        .b(xor2(*all_one, cin))
        .c(xor3(cin, *out_wire, *msb_one))
        .build();
}

pub fn emit_eval_bytecode(
    _gate: Gate,
    data: &GateData,
    builder: &mut crate::compiler::eval_form::BytecodeBuilder,
    wire_to_reg: impl Fn(Wire) -> u32,
) {
    let GateParam {
        constants,
        inputs,
        outputs,
        aux,
        scratch,
        ..
    } = data.gate_param();
    let [all_one, msb_one, zero] = constants else {
        unreachable!()
    };
    let [x, y] = inputs else { unreachable!() };
    let [out_wire] = outputs else { unreachable!() };
    let [cout] = aux else { unreachable!() };
    let [scratch_diff, scratch_sum_unused] = scratch else {
        unreachable!()
    };

    // Compute diff = x ^ y
    builder.emit_bxor(wire_to_reg(*scratch_diff), wire_to_reg(*x), wire_to_reg(*y));

    // Compute carry bits from all_one + diff
    builder.emit_iadd_cin_cout(
        wire_to_reg(*scratch_sum_unused), // sum (unused)
        wire_to_reg(*cout),               // cout
        wire_to_reg(*all_one),            // all_one
        wire_to_reg(*scratch_diff),       // diff
        wire_to_reg(*zero),               // cin = 0
    );

    // Invert: out_wire = out_wire ^ msb_one
    builder.emit_bxor(
        wire_to_reg(*out_wire),
        wire_to_reg(*cout),
        wire_to_reg(*msb_one),
    );
}
